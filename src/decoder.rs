//! RLNC Decoder — in-place pivot-based Gaussian elimination over GF(2⁸).
//!
//! Working rows are recycled from a free-list (no heap alloc in the steady-state
//! receive path). Pivot reorder after back-substitution is done in-place via
//! swaps (no second matrix allocation).

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::aligned::AlignedBuffer;
use crate::encoder::CodedPacket;
use crate::error::RlncError;
use crate::field::tables::{EXP, LOG};
use crate::kernel;

/// RLNC Decoder.
#[cfg(feature = "alloc")]
pub struct Decoder {
    generation_size: usize,
    symbol_size: usize,
    /// Augmented rows: `[coefficients (k bytes) | payload (symbol_size bytes)]`.
    rows: Vec<AlignedBuffer>,
    /// Recycled working rows for `receive` (same length as matrix rows).
    free_rows: Vec<AlignedBuffer>,
    pivot_col: Vec<Option<usize>>,
    rank: usize,
    decoded: bool,
}

#[cfg(feature = "alloc")]
impl Decoder {
    /// Create a new decoder for a generation.
    pub fn new(generation_size: usize, symbol_size: usize) -> Result<Self, RlncError> {
        if generation_size == 0 || symbol_size == 0 {
            return Err(RlncError::InvalidParameters);
        }
        let k = generation_size;
        let row_len = k + symbol_size;
        let rows = (0..k).map(|_| AlignedBuffer::zeroed(row_len)).collect();
        // Pre-seed one free row so first `receive` need not allocate.
        let free_rows = vec![AlignedBuffer::zeroed(row_len)];
        Ok(Decoder {
            generation_size,
            symbol_size,
            rows,
            free_rows,
            pivot_col: vec![None; k],
            rank: 0,
            decoded: false,
        })
    }

    /// Generation size (`k`).
    pub fn generation_size(&self) -> usize {
        self.generation_size
    }
    /// Symbol size in bytes.
    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }
    /// Current rank (innovative packets received).
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// True when rank == `generation_size`.
    pub fn is_complete(&self) -> bool {
        self.rank == self.generation_size
    }

    fn row_len(&self) -> usize {
        self.generation_size + self.symbol_size
    }

    /// Take a working row from the free-list (or allocate if empty).
    fn take_work_row(&mut self) -> AlignedBuffer {
        let mut row = self
            .free_rows
            .pop()
            .unwrap_or_else(|| AlignedBuffer::zeroed(self.row_len()));
        row.as_mut_slice().fill(0);
        row
    }

    /// Receive a coded packet.
    ///
    /// Returns `true` if the packet was innovative (increased rank).
    ///
    /// Steady-state hot path: **no heap allocation** (free-list row + in-place
    /// [`kernel::scale_inplace`] with SIMD).
    pub fn receive(&mut self, pkt: CodedPacket) -> Result<bool, RlncError> {
        let k = self.generation_size;
        let n = self.symbol_size;

        if pkt.coefficients.len() != k || pkt.payload.len() != n {
            return Err(RlncError::PacketSizeMismatch {
                expected_coeffs: k,
                got_coeffs: pkt.coefficients.len(),
                expected_payload: n,
                got_payload: pkt.payload.len(),
            });
        }

        if self.is_complete() {
            return Ok(false);
        }

        let mut row = self.take_work_row();
        row.as_mut_slice()[..k].copy_from_slice(pkt.coefficients.as_slice());
        row.as_mut_slice()[k..].copy_from_slice(pkt.payload.as_slice());

        // Forward elimination — pivot rows and `row` are different buffers.
        for r in 0..self.rank {
            let Some(col) = self.pivot_col[r] else {
                continue;
            };
            let coeff = row.as_slice()[col];
            if coeff == 0 {
                continue;
            }
            kernel::axpy(coeff, self.rows[r].as_slice(), row.as_mut_slice());
        }

        let new_pivot = row.as_slice()[..k].iter().position(|&b| b != 0);
        let Some(pivot_col) = new_pivot else {
            // Linearly dependent — recycle row
            self.free_rows.push(row);
            return Ok(false);
        };

        // Normalise pivot to 1 — SIMD in-place scale.
        let pivot_val = row.as_slice()[pivot_col];
        if pivot_val != 1 {
            let inv = EXP[255 - LOG[pivot_val as usize] as usize];
            if inv != 1 {
                kernel::scale_inplace(inv, row.as_mut_slice());
            }
        }

        // Install into matrix; previous placeholder row can be recycled.
        let old = core::mem::replace(&mut self.rows[self.rank], row);
        self.free_rows.push(old);
        self.pivot_col[self.rank] = Some(pivot_col);
        self.rank += 1;
        self.decoded = false;

        Ok(true)
    }

    /// Attempt to decode. Returns `Some(symbols)` when rank == `generation_size`.
    pub fn decode(&mut self) -> Result<Option<Vec<Vec<u8>>>, RlncError> {
        if !self.is_complete() {
            return Ok(None);
        }
        if self.decoded {
            return Ok(Some(self.extract_symbols()));
        }

        let k = self.generation_size;

        // Back-substitution — split_at_mut, no allocation
        for r in (0..k).rev() {
            let Some(col) = self.pivot_col[r] else {
                continue;
            };
            for r2 in 0..r {
                let coeff = self.rows[r2].as_slice()[col];
                if coeff == 0 {
                    continue;
                }
                let (lo, hi) = self.rows.split_at_mut(r);
                let pivot_slice: &[u8] = hi[0].as_slice();
                kernel::axpy(coeff, pivot_slice, lo[r2].as_mut_slice());
            }
        }

        // In-place permutation so that row i has pivot column i (cycle following).
        // pivot_col[r] was the pivot of the r-th innovative row before reorder.
        self.permute_rows_to_identity_pivots();
        self.decoded = true;

        Ok(Some(self.extract_symbols()))
    }

    /// In-place reorder: selection-sort rows by pivot column (O(k²), k small).
    /// After full rank, row `i` has pivot column `i`. No second matrix allocation.
    fn permute_rows_to_identity_pivots(&mut self) {
        let k = self.generation_size;
        for i in 0..k {
            let mut best = i;
            let mut best_col = self.pivot_col[i].unwrap_or(usize::MAX);
            for j in (i + 1)..k {
                let c = self.pivot_col[j].unwrap_or(usize::MAX);
                if c < best_col {
                    best = j;
                    best_col = c;
                }
            }
            if best != i {
                self.rows.swap(i, best);
                self.pivot_col.swap(i, best);
            }
        }
        for i in 0..k {
            self.pivot_col[i] = Some(i);
        }
    }

    fn extract_symbols(&self) -> Vec<Vec<u8>> {
        let k = self.generation_size;
        let n = self.symbol_size;
        self.rows
            .iter()
            .map(|row| row.as_slice()[k..k + n].to_vec())
            .collect()
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;
    use crate::encoder::{Encoder, SimpleRng};

    fn make_source(k: usize, n: usize) -> Vec<Vec<u8>> {
        (0..k)
            .map(|i| (0..n).map(|j| (i * 7 + j * 3) as u8).collect())
            .collect()
    }

    #[test]
    fn encode_decode_round_trip() {
        let k = 4usize;
        let n = 64usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        let mut rng = SimpleRng::new(0xDEAD_BEEF);

        let mut innovative = 0;
        for _ in 0..k + 2 {
            let pkt = enc.encode_random(&refs, &mut rng).unwrap();
            if dec.receive(pkt).unwrap() {
                innovative += 1;
            }
        }
        assert_eq!(innovative, k);
        assert!(dec.is_complete());

        let decoded = dec.decode().unwrap().unwrap();
        assert_eq!(decoded.len(), k);
        for i in 0..k {
            assert_eq!(decoded[i], source[i], "symbol {i} mismatch");
        }
    }

    #[test]
    fn systematic_decode() {
        let k = 3usize;
        let n = 32usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        for i in 0..k {
            let pkt = enc.encode_systematic(&refs, i).unwrap();
            assert!(dec.receive(pkt).unwrap());
        }
        assert!(dec.is_complete());
        let decoded = dec.decode().unwrap().unwrap();
        for i in 0..k {
            assert_eq!(decoded[i], source[i]);
        }
    }

    #[test]
    fn redundant_packet_ignored() {
        let k = 2usize;
        let n = 8usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        let pkt0 = enc.encode_systematic(&refs, 0).unwrap();
        let pkt0_dup = enc.encode_systematic(&refs, 0).unwrap();
        assert!(dec.receive(pkt0).unwrap());
        assert!(!dec.receive(pkt0_dup).unwrap());
        assert_eq!(dec.rank(), 1);
    }

    #[test]
    fn decoder_rows_are_aligned() {
        use crate::aligned::ALIGN;
        let k = 4usize;
        let n = 128usize;
        let dec = Decoder::new(k, n).unwrap();
        for (i, row) in dec.rows.iter().enumerate() {
            assert_eq!(
                row.as_ptr() as usize % ALIGN,
                0,
                "decoder row {i} not {ALIGN}-byte aligned"
            );
        }
    }

    #[test]
    fn free_list_recycles_on_redundant() {
        let k = 2usize;
        let n = 16usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        let free_before = dec.free_rows.len();
        let p = enc.encode_systematic(&refs, 0).unwrap();
        assert!(dec.receive(p).unwrap());
        let p2 = enc.encode_systematic(&refs, 0).unwrap();
        assert!(!dec.receive(p2).unwrap());
        // Redundant receive returns row to free list
        assert!(dec.free_rows.len() >= free_before);
    }

    #[test]
    fn new_rejects_zero_params() {
        assert!(Decoder::new(0, 8).is_err());
        assert!(Decoder::new(4, 0).is_err());
    }

    #[test]
    fn receive_rejects_packet_size_mismatch() {
        let mut dec = Decoder::new(2, 4).unwrap();
        let bad = CodedPacket::from_slices(&[1], &[1, 2, 3, 4]); // wrong coeff len
        let err = dec.receive(bad).unwrap_err();
        match err {
            crate::error::RlncError::PacketSizeMismatch {
                expected_coeffs: 2,
                got_coeffs: 1,
                expected_payload: 4,
                got_payload: 4,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn decode_none_when_incomplete() {
        let k = 3usize;
        let n = 8usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        // Only one systematic packet
        let pkt = enc.encode_systematic(&refs, 0).unwrap();
        assert!(dec.receive(pkt).unwrap());
        assert!(!dec.is_complete());
        assert_eq!(dec.rank(), 1);
        let out = dec.decode().unwrap();
        assert!(out.is_none(), "decode must be None before full rank");
    }

    #[test]
    fn receive_after_complete_returns_false() {
        let k = 2usize;
        let n = 8usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        for i in 0..k {
            assert!(dec
                .receive(enc.encode_systematic(&refs, i).unwrap())
                .unwrap());
        }
        assert!(dec.is_complete());
        let extra = enc.encode_systematic(&refs, 0).unwrap();
        assert!(!dec.receive(extra).unwrap());
    }
}
