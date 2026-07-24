//! Error types for the `rlnc-simdx` crate.

/// Errors returned by RLNC operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlncError {
    /// `generation_size` or `symbol_size` was zero.
    InvalidParameters,
    /// A coded packet had wrong coefficient or payload length.
    PacketSizeMismatch {
        /// Expected coefficient vector length (`generation_size`).
        expected_coeffs: usize,
        /// Actual coefficient vector length received.
        got_coeffs: usize,
        /// Expected payload length (`symbol_size`).
        expected_payload: usize,
        /// Actual payload length received.
        got_payload: usize,
    },
    /// The source slice count does not match `generation_size`.
    SourceCountMismatch {
        /// Expected number of source symbols.
        expected: usize,
        /// Actual number of slices provided.
        got: usize,
    },
    /// A source symbol had the wrong byte length.
    SourceSizeMismatch {
        /// Expected bytes per symbol.
        expected: usize,
        /// Actual slice length.
        got: usize,
    },
    /// Systematic index out of range.
    IndexOutOfRange {
        /// Requested systematic symbol index.
        index: usize,
        /// Exclusive upper bound (`generation_size`).
        max: usize,
    },
}

impl core::fmt::Display for RlncError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameters => {
                write!(f, "generation_size and symbol_size must both be > 0")
            }
            Self::PacketSizeMismatch {
                expected_coeffs,
                got_coeffs,
                expected_payload,
                got_payload,
            } => write!(
                f,
                "packet size mismatch: expected ({expected_coeffs} coeffs, {expected_payload} payload), got ({got_coeffs}, {got_payload})"
            ),
            Self::SourceCountMismatch { expected, got } => {
                write!(f, "expected {expected} source symbols, got {got}")
            }
            Self::SourceSizeMismatch { expected, got } => {
                write!(
                    f,
                    "source symbol size mismatch: expected {expected} bytes, got {got}"
                )
            }
            Self::IndexOutOfRange { index, max } => {
                write!(f, "systematic index {index} out of range [0, {max})")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RlncError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_variants_non_empty() {
        let samples = [
            RlncError::InvalidParameters,
            RlncError::PacketSizeMismatch {
                expected_coeffs: 4,
                got_coeffs: 2,
                expected_payload: 8,
                got_payload: 1,
            },
            RlncError::SourceCountMismatch {
                expected: 4,
                got: 3,
            },
            RlncError::SourceSizeMismatch {
                expected: 16,
                got: 8,
            },
            RlncError::IndexOutOfRange { index: 5, max: 4 },
        ];
        for e in samples {
            let s = format!("{e}");
            assert!(!s.is_empty(), "empty Display for {e:?}");
        }
    }
}
