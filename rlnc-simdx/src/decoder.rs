//! RLNC Decoder — in-place pivot-based Gaussian elimination over GF(2⁸).
//!
//! Coefficients and payloads are stored separately. Incoming coefficients are
//! reduced first so dependent packets never touch their large payload, while
//! innovative packet buffers move directly into column-ordered pivot storage.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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
    coefficient_rows: Vec<AlignedBuffer>,
    payload_rows: Vec<AlignedBuffer>,
    pivot_col: Vec<Option<usize>>,
    elimination_factors: Vec<u8>,
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
        let mut coefficient_rows = Vec::new();
        let mut payload_rows = Vec::new();
        let mut pivot_col = Vec::new();
        let mut elimination_factors = Vec::new();
        coefficient_rows
            .try_reserve_exact(k)
            .map_err(|_| RlncError::InvalidParameters)?;
        payload_rows
            .try_reserve_exact(k)
            .map_err(|_| RlncError::InvalidParameters)?;
        pivot_col
            .try_reserve_exact(k)
            .map_err(|_| RlncError::InvalidParameters)?;
        elimination_factors
            .try_reserve_exact(k)
            .map_err(|_| RlncError::InvalidParameters)?;
        pivot_col.resize(k, None);
        elimination_factors.resize(k, 0);
        Ok(Decoder {
            generation_size,
            symbol_size,
            coefficient_rows,
            payload_rows,
            pivot_col,
            elimination_factors,
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

    /// Receive a coded packet.
    ///
    /// Returns `true` if the packet was innovative (increased rank).
    ///
    /// Innovative packet allocations are moved directly into pivot storage.
    /// Dependent packets are rejected after coefficient-only reduction, without
    /// payload-sized arithmetic or additional allocation.
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

        let CodedPacket {
            mut coefficients,
            mut payload,
        } = pkt;

        // Reduce the small coefficient vector first and record the operations.
        // A dependent packet can then be rejected without touching its payload.
        for r in 0..self.rank {
            let Some(col) = self.pivot_col[r] else {
                continue;
            };
            let coeff = coefficients.as_slice()[col];
            self.elimination_factors[r] = coeff;
            if coeff == 0 {
                continue;
            }
            // SAFETY: stored and incoming packet buffers are distinct owned
            // allocations and the suffix lengths match.
            unsafe {
                kernel::axpy_unchecked(
                    coeff,
                    &self.coefficient_rows[r].as_slice()[col..],
                    &mut coefficients.as_mut_slice()[col..],
                );
            }
        }

        let new_pivot = coefficients.as_slice().iter().position(|&b| b != 0);
        let Some(pivot_col) = new_pivot else {
            return Ok(false);
        };

        for r in 0..self.rank {
            let coeff = self.elimination_factors[r];
            if coeff != 0 {
                // SAFETY: stored and incoming payloads are distinct and have
                // the decoder's validated symbol length.
                unsafe {
                    kernel::axpy_unchecked(
                        coeff,
                        self.payload_rows[r].as_slice(),
                        payload.as_mut_slice(),
                    );
                }
            }
        }

        let pivot_val = coefficients.as_slice()[pivot_col];
        if pivot_val != 1 {
            let inv = EXP[255 - LOG[pivot_val as usize] as usize];
            kernel::scale_inplace(inv, &mut coefficients.as_mut_slice()[pivot_col..]);
            kernel::scale_inplace(inv, payload.as_mut_slice());
        }

        // Keep pivot rows ordered. Moving AlignedBuffer handles is cheap and
        // makes the full-rank pivot order the identity, avoiding a decode-time
        // O(k^2) selection sort and preserving the echelon invariant.
        let insert_at = self.pivot_col[..self.rank]
            .iter()
            .position(|&col| col.is_some_and(|col| col > pivot_col))
            .unwrap_or(self.rank);

        self.coefficient_rows.push(coefficients);
        self.payload_rows.push(payload);
        self.pivot_col[self.rank] = Some(pivot_col);
        for i in (insert_at..self.rank).rev() {
            self.coefficient_rows.swap(i, i + 1);
            self.payload_rows.swap(i, i + 1);
            self.pivot_col.swap(i, i + 1);
        }
        self.rank += 1;
        self.decoded = false;

        debug_assert!(self.pivot_col[..self.rank]
            .windows(2)
            .all(|pair| pair[0] < pair[1]));

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

        // There are k distinct, ordered pivots drawn from k columns, so their
        // only possible full-rank order is 0..k.
        debug_assert!(self
            .pivot_col
            .iter()
            .enumerate()
            .all(|(col, &pivot)| pivot == Some(col)));

        // Back-substitution — split_at_mut, no allocation
        for r in (0..k).rev() {
            let Some(col) = self.pivot_col[r] else {
                continue;
            };
            let (coefficients_above, pivot_coefficients) = self.coefficient_rows.split_at_mut(r);
            let coefficient_suffix = &pivot_coefficients[0].as_slice()[col..];
            let (payloads_above, pivot_payloads) = self.payload_rows.split_at_mut(r);
            let pivot_payload = pivot_payloads[0].as_slice();
            for r2 in 0..r {
                let coeff = coefficients_above[r2].as_slice()[col];
                if coeff == 0 {
                    continue;
                }
                // SAFETY: split_at_mut separates the pivot and destination
                // rows; corresponding coefficient suffixes and payloads match.
                unsafe {
                    kernel::axpy_unchecked(
                        coeff,
                        coefficient_suffix,
                        &mut coefficients_above[r2].as_mut_slice()[col..],
                    );
                    kernel::axpy_unchecked(coeff, pivot_payload, payloads_above[r2].as_mut_slice());
                }
            }
        }

        self.decoded = true;

        Ok(Some(self.extract_symbols()))
    }

    fn extract_symbols(&self) -> Vec<Vec<u8>> {
        self.payload_rows
            .iter()
            .map(AlignedBuffer::to_vec)
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
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let encoder = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        assert!(dec
            .receive(encoder.encode_systematic(&refs, 0).unwrap())
            .unwrap());
        for (i, row) in dec.coefficient_rows.iter().enumerate() {
            assert_eq!(
                row.as_ptr() as usize % ALIGN,
                0,
                "decoder coefficient row {i} not {ALIGN}-byte aligned"
            );
        }
        for (i, row) in dec.payload_rows.iter().enumerate() {
            assert_eq!(
                row.as_ptr() as usize % ALIGN,
                0,
                "decoder payload row {i} not {ALIGN}-byte aligned"
            );
        }
    }

    #[test]
    fn redundant_packet_does_not_add_storage() {
        let k = 2usize;
        let n = 16usize;
        let source = make_source(k, n);
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let enc = Encoder::new(k, n).unwrap();
        let mut dec = Decoder::new(k, n).unwrap();
        let p = enc.encode_systematic(&refs, 0).unwrap();
        assert!(dec.receive(p).unwrap());
        let rows_before = dec.payload_rows.len();
        let p2 = enc.encode_systematic(&refs, 0).unwrap();
        assert!(!dec.receive(p2).unwrap());
        assert_eq!(dec.payload_rows.len(), rows_before);
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
