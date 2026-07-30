//! RLNC Recoder — re-encodes already-coded packets into new coded packets.
//!
//! All output buffers are [`AlignedBuffer`]-backed, ensuring the recoded
//! payload passes 64-byte aligned pointers to the SIMD kernels.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::aligned::AlignedBuffer;
use crate::encoder::{CodedPacket, SimpleRng};
use crate::error::RlncError;
use crate::kernel;

/// RLNC Recoder.
pub struct Recoder;

impl Recoder {
    /// Produce a new coded packet from a set of already-coded packets.
    ///
    /// All input packets must have the same coefficient length and payload length.
    #[cfg(feature = "alloc")]
    pub fn recode(coded: &[CodedPacket], rng: &mut SimpleRng) -> Result<CodedPacket, RlncError> {
        if coded.is_empty() {
            return Err(RlncError::InvalidParameters);
        }

        let k = coded[0].coefficients.len();
        let n = coded[0].payload.len();

        for pkt in coded {
            if pkt.coefficients.len() != k || pkt.payload.len() != n {
                return Err(RlncError::PacketSizeMismatch {
                    expected_coeffs: k,
                    got_coeffs: pkt.coefficients.len(),
                    expected_payload: n,
                    got_payload: pkt.payload.len(),
                });
            }
        }

        let mut out_coeffs = AlignedBuffer::zeroed(k);
        let mut out_payload = AlignedBuffer::zeroed(n);

        let mut recode_coeffs = alloc::vec![0u8; coded.len()];
        rng.fill(&mut recode_coeffs);
        // Bounded retries — M4
        let mut retries = 0u32;
        while recode_coeffs.iter().all(|&c| c == 0) {
            retries += 1;
            if retries >= 100 {
                recode_coeffs[0] = 1;
                break;
            }
            rng.fill(&mut recode_coeffs);
        }

        let mut sources = Vec::with_capacity(coded.len());
        sources.extend(coded.iter().map(|packet| packet.coefficients.as_slice()));
        kernel::axpy_multi(&recode_coeffs, &sources, out_coeffs.as_mut_slice());
        sources.clear();
        sources.extend(coded.iter().map(|packet| packet.payload.as_slice()));
        kernel::axpy_multi(&recode_coeffs, &sources, out_payload.as_mut_slice());

        Ok(CodedPacket {
            coefficients: out_coeffs,
            payload: out_payload,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;
    use crate::decoder::Decoder;
    use crate::encoder::Encoder;

    #[test]
    fn recode_then_decode() {
        let k = 4usize;
        let n = 32usize;
        let source: alloc::vec::Vec<alloc::vec::Vec<u8>> = (0..k)
            .map(|i| alloc::vec![(i as u8).wrapping_mul(17); n])
            .collect();
        let refs: alloc::vec::Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();

        let enc = Encoder::new(k, n).unwrap();
        let mut rng = SimpleRng::new(0x1234_5678);

        // Full-rank source set: mix systematic + random for reliable span
        let mut coded: alloc::vec::Vec<CodedPacket> = (0..k)
            .map(|i| enc.encode_systematic(&refs, i).unwrap())
            .collect();
        for _ in 0..k {
            coded.push(enc.encode_random(&refs, &mut rng).unwrap());
        }

        let mut dec = Decoder::new(k, n).unwrap();
        let mut recode_rng = SimpleRng::new(0xABCD_EF01);
        // Enough recodes to finish with high probability; fail hard if not
        for _ in 0..k * 4 {
            let recoded = Recoder::recode(&coded, &mut recode_rng).unwrap();
            let _ = dec.receive(recoded);
            if dec.is_complete() {
                break;
            }
        }

        assert!(
            dec.is_complete(),
            "recode stream never reached full rank (rank={})",
            dec.rank()
        );
        let decoded = dec.decode().unwrap().expect("decode after complete");
        assert_eq!(decoded.len(), k);
        for i in 0..k {
            assert_eq!(decoded[i], source[i], "symbol {i} mismatch after recode");
        }
    }

    #[test]
    fn recode_empty_is_invalid() {
        let mut rng = SimpleRng::new(1);
        let err = Recoder::recode(&[], &mut rng).unwrap_err();
        assert_eq!(err, RlncError::InvalidParameters);
    }

    #[test]
    fn recode_size_mismatch() {
        let mut rng = SimpleRng::new(2);
        let a = CodedPacket::from_slices(&[1, 0], &[9, 9]);
        let b = CodedPacket::from_slices(&[0, 1, 0], &[8, 8]); // wrong coeff len
        let err = Recoder::recode(&[a, b], &mut rng).unwrap_err();
        match err {
            RlncError::PacketSizeMismatch { .. } => {}
            other => panic!("expected PacketSizeMismatch, got {other:?}"),
        }
    }
}
