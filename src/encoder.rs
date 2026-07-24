//! RLNC Encoder — produces random linear coded packets over GF(2⁸).
//!
//! `CodedPacket.payload` is backed by an [`AlignedBuffer`] so that all
//! intermediate re-encoding and decoding operations on the payload always
//! pass 64-byte aligned pointers to the SIMD kernels.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::aligned::AlignedBuffer;
use crate::error::RlncError;
use crate::kernel;

/// A single coded packet: GF(2⁸) coefficient vector + 64-byte-aligned payload.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct CodedPacket {
    /// Length == `generation_size`.  The i-th byte is the coefficient
    /// applied to source symbol i.
    pub coefficients: AlignedBuffer,
    /// Length == `symbol_size`, 64-byte aligned.
    pub payload: AlignedBuffer,
}

#[cfg(feature = "alloc")]
impl CodedPacket {
    /// Coefficient vector as a byte slice.
    #[inline]
    pub fn coefficients_slice(&self) -> &[u8] {
        self.coefficients.as_slice()
    }

    /// Payload as a byte slice.
    #[inline]
    pub fn payload_slice(&self) -> &[u8] {
        self.payload.as_slice()
    }

    /// Copy coefficients and payload into owned `Vec`s (for FFI / unaligned consumers).
    pub fn into_vecs(self) -> (Vec<u8>, Vec<u8>) {
        (self.coefficients.into_vec(), self.payload.into_vec())
    }

    /// Build a packet from coefficient and payload slices (copies into aligned buffers).
    pub fn from_slices(coefficients: &[u8], payload: &[u8]) -> Self {
        Self {
            coefficients: AlignedBuffer::from_slice(coefficients),
            payload: AlignedBuffer::from_slice(payload),
        }
    }

    /// Build a packet from owned `Vec`s (copies into aligned buffers).
    pub fn from_vecs(coefficients: Vec<u8>, payload: Vec<u8>) -> Self {
        Self {
            coefficients: AlignedBuffer::from_slice(&coefficients),
            payload: AlignedBuffer::from_slice(&payload),
        }
    }
}

/// Simple LFSR-based PRNG for coding coefficients — **not a CSPRNG**.
///
/// Use only for RLNC coefficient generation in non-adversarial settings.
/// For adversarial environments, inject entropy from a cryptographic RNG
/// outside this crate. See the crate-level security warning.
pub struct SimpleRng(u64);

impl SimpleRng {
    /// Create a new RNG from `seed`.
    pub fn new(seed: u64) -> Self {
        SimpleRng(seed.wrapping_add(1))
    }
    /// Next pseudorandom byte.
    pub fn next_u8(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u8
    }
    /// Fill `buf` with pseudorandom bytes.
    pub fn fill(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.next_u8();
        }
    }
}

/// RLNC Encoder.
#[cfg(feature = "alloc")]
pub struct Encoder {
    generation_size: usize,
    symbol_size: usize,
}

#[cfg(feature = "alloc")]
impl Encoder {
    /// Create a new encoder.
    ///
    /// - `generation_size` (`k`): number of source symbols per generation.
    /// - `symbol_size` (`n`): bytes per source symbol.
    pub fn new(generation_size: usize, symbol_size: usize) -> Result<Self, RlncError> {
        if generation_size == 0 || symbol_size == 0 {
            return Err(RlncError::InvalidParameters);
        }
        Ok(Encoder {
            generation_size,
            symbol_size,
        })
    }

    /// Generation size (`k`).
    pub fn generation_size(&self) -> usize {
        self.generation_size
    }
    /// Symbol size in bytes (`n`).
    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }

    /// Encode one random-linear coded packet.
    ///
    /// Uses cache-blocked multi-source AXPY ([`kernel::axpy_multi`]) for
    /// better DRAM behaviour when `k` is moderate and `symbol_size` is large.
    ///
    /// `source` must contain exactly `generation_size` slices, each of
    /// length `symbol_size`.  `rng` provides the random coefficients.
    pub fn encode_random(
        &self,
        source: &[&[u8]],
        rng: &mut SimpleRng,
    ) -> Result<CodedPacket, RlncError> {
        self.validate_source(source)?;

        let k = self.generation_size;
        let n = self.symbol_size;

        let mut coeffs = AlignedBuffer::zeroed(k);
        let mut payload = AlignedBuffer::zeroed(n);

        rng.fill(coeffs.as_mut_slice());

        // Avoid all-zero coefficient vector (bounded retries — M4)
        let mut retries = 0u32;
        while coeffs.as_slice().iter().all(|&c| c == 0) {
            retries += 1;
            if retries >= 100 {
                coeffs.as_mut_slice()[0] = 1;
                break;
            }
            rng.fill(coeffs.as_mut_slice());
        }

        kernel::axpy_multi(coeffs.as_slice(), source, payload.as_mut_slice());

        Ok(CodedPacket {
            coefficients: coeffs,
            payload,
        })
    }

    /// Produce a **systematic** coded packet for source symbol `index`.
    pub fn encode_systematic(
        &self,
        source: &[&[u8]],
        index: usize,
    ) -> Result<CodedPacket, RlncError> {
        self.validate_source(source)?;
        if index >= self.generation_size {
            return Err(RlncError::IndexOutOfRange {
                index,
                max: self.generation_size,
            });
        }

        let mut coeffs = AlignedBuffer::zeroed(self.generation_size);
        coeffs.as_mut_slice()[index] = 1;
        let payload = AlignedBuffer::from_slice(source[index]);

        Ok(CodedPacket {
            coefficients: coeffs,
            payload,
        })
    }

    fn validate_source(&self, source: &[&[u8]]) -> Result<(), RlncError> {
        if source.len() != self.generation_size {
            return Err(RlncError::SourceCountMismatch {
                expected: self.generation_size,
                got: source.len(),
            });
        }
        for s in source {
            if s.len() != self.symbol_size {
                return Err(RlncError::SourceSizeMismatch {
                    expected: self.symbol_size,
                    got: s.len(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;
    use crate::aligned::ALIGN;

    #[test]
    fn systematic_packet_is_copy() {
        let k = 4usize;
        let n = 8usize;
        let symbols: Vec<Vec<u8>> = (0..k as u8).map(|i| vec![i * 10; n]).collect();
        let refs: Vec<&[u8]> = symbols.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        for (i, symbol) in symbols.iter().enumerate() {
            let pkt = enc.encode_systematic(&refs, i).unwrap();
            assert_eq!(pkt.payload.as_slice(), symbol.as_slice());
            assert_eq!(pkt.coefficients.as_slice()[i], 1);
            assert!(pkt
                .coefficients
                .as_slice()
                .iter()
                .enumerate()
                .all(|(j, &c)| if j == i { c == 1 } else { c == 0 }));
        }
    }

    #[test]
    fn random_encode_non_zero_payload() {
        let k = 4usize;
        let n = 16usize;
        let symbols: Vec<Vec<u8>> = (1u8..=k as u8).map(|i| vec![i; n]).collect();
        let refs: Vec<&[u8]> = symbols.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        let mut rng = SimpleRng::new(42);
        let pkt = enc.encode_random(&refs, &mut rng).unwrap();
        assert_eq!(pkt.payload.len(), n);
        assert_eq!(pkt.coefficients.len(), k);
    }

    #[test]
    fn coded_packet_payload_is_aligned() {
        let k = 4usize;
        let n = 128usize;
        let symbols: Vec<Vec<u8>> = (0..k as u8).map(|i| vec![i; n]).collect();
        let refs: Vec<&[u8]> = symbols.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        let mut rng = SimpleRng::new(0xBEEF);
        let pkt = enc.encode_random(&refs, &mut rng).unwrap();
        assert_eq!(
            pkt.payload.as_ptr() as usize % ALIGN,
            0,
            "CodedPacket.payload must be {ALIGN}-byte aligned"
        );
        assert_eq!(
            pkt.coefficients.as_ptr() as usize % ALIGN,
            0,
            "CodedPacket.coefficients must be {ALIGN}-byte aligned"
        );
    }

    #[test]
    fn coded_packet_into_from_vecs_roundtrip() {
        let pkt = CodedPacket::from_slices(&[1, 2, 3, 4], &[9, 8, 7, 6]);
        let (c, p) = pkt.into_vecs();
        assert_eq!(c, vec![1, 2, 3, 4]);
        assert_eq!(p, vec![9, 8, 7, 6]);
        let pkt2 = CodedPacket::from_vecs(c, p);
        assert_eq!(pkt2.coefficients_slice(), &[1, 2, 3, 4]);
        assert_eq!(pkt2.payload_slice(), &[9, 8, 7, 6]);
    }

    #[test]
    fn new_rejects_zero_params() {
        assert!(Encoder::new(0, 16).is_err());
        assert!(Encoder::new(4, 0).is_err());
        assert!(matches!(
            Encoder::new(0, 0),
            Err(crate::error::RlncError::InvalidParameters)
        ));
    }

    #[test]
    fn encode_rejects_source_count_mismatch() {
        let enc = Encoder::new(2, 4).unwrap();
        let s0 = [1u8; 4];
        let mut rng = SimpleRng::new(1);
        let err = enc.encode_random(&[&s0], &mut rng).unwrap_err();
        match err {
            crate::error::RlncError::SourceCountMismatch {
                expected: 2,
                got: 1,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn encode_rejects_source_size_mismatch() {
        let enc = Encoder::new(2, 4).unwrap();
        let s0 = [1u8; 4];
        let s1 = [2u8; 3];
        let mut rng = SimpleRng::new(1);
        let err = enc.encode_random(&[&s0, &s1], &mut rng).unwrap_err();
        match err {
            crate::error::RlncError::SourceSizeMismatch {
                expected: 4,
                got: 3,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn systematic_index_out_of_range() {
        let enc = Encoder::new(2, 4).unwrap();
        let s0 = [1u8; 4];
        let s1 = [2u8; 4];
        let err = enc.encode_systematic(&[&s0, &s1], 2).unwrap_err();
        match err {
            crate::error::RlncError::IndexOutOfRange { index: 2, max: 2 } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}
