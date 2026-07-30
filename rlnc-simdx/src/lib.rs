//! # rlnc-simdx — Random Linear Network Coding over GF(2⁸)
//!
//! High-performance, `no_std`-compatible RLNC implementation with maximum
//! SIMD acceleration across all major architectures.
//!
//! Crate: **`rlnc-simdx`** · use as `use rlnc_simdx::...`.
//!
//! ## ⚠️ Security & cryptography warning
//!
//! **This is a network-coding acceleration library, not a cryptography library.**
//!
//! - **Do not** use it to “encrypt” or obfuscate confidential data.
//! - Field arithmetic is **not constant-time** — no side-channel resistance.
//! - Built-in [`SimpleRng`] is a fast LFSR for coding coefficients, **not** a CSPRNG.
//! - Authenticate and protect coded packets with external crypto (TLS, AEAD, etc.).
//!
//! ## Kernel safety (public API)
//!
//! - [`kernel::axpy`] / [`kernel::scale`]: equal lengths **and** non-overlapping
//!   buffers are **asserted in release** (cheap pointer-range check).
//! - For in-place scale use [`kernel::scale_inplace`].
//! - Raw SIMD tier functions are **crate-private**; only the safe wrappers above
//!   are part of the supported public surface.
//!
//! ## SIMD Tier Hierarchy
//!
//! With `std`, the library selects the best available kernel at runtime and
//! caches that choice. Without `std`, it selects from compile-time target features.
//!
//! | Tier | Feature flags                   | Width   | Throughput |
//! |------|---------------------------------|---------|------------|
//! | 1    | GFNI + AVX-512BW                | 512-bit | ~64 B/cy   |
//! | 2    | GFNI + AVX2                     | 256-bit | ~32 B/cy   |
//! | 3    | GFNI + SSE4.2                   | 128-bit | ~16 B/cy   |
//! | 4    | AVX-512BW + SSSE3               | 512-bit | ~11 B/cy   |
//! | 5    | AVX2 + SSSE3                    | 256-bit | ~5 B/cy    |
//! | 6    | SSSE3                           | 128-bit | ~3 B/cy    |
//! | 7    | NEON (`AArch64`)                | 128-bit | ~3 B/cy    |
//! | 8    | WASM SIMD128                    | 128-bit | ~3 B/cy    |
//! | 9    | Scalar (all targets)            | 1 byte  | ~0.3 B/cy  |
//!
//! SVE is experimental, has no production kernel, and is not selected by dispatch.
//!
//! ## Feature Flags
//!
//! | Flag              | Default | Effect                                       |
//! |-------------------|---------|----------------------------------------------|
//! | `alloc`           | on      | Enables heap-backed RLNC APIs and `GfMatrix` |
//! | `std`             | on      | Enables runtime CPU dispatch; implies alloc  |
//! | `bench-internals` | off     | Unstable scalar/direct-tier benchmark APIs   |
//!
//! ## Quick Start
//!
//! ```rust
//! use rlnc_simdx::{Encoder, Decoder, SimpleRng};
//!
//! let k = 4;   // generation size (number of source symbols)
//! let n = 128; // symbol size (bytes)
//!
//! // Source data
//! let source: Vec<Vec<u8>> = (0..k).map(|i| vec![i as u8; n]).collect();
//! let refs: Vec<&[u8]> = source.iter().map(|v| v.as_slice()).collect();
//!
//! // Encode
//! let enc = Encoder::new(k, n).unwrap();
//! let mut rng = SimpleRng::new(42);
//! let packets: Vec<_> = (0..k+2)
//!     .map(|_| enc.encode_random(&refs, &mut rng).unwrap())
//!     .collect();
//!
//! // Decode
//! let mut dec = Decoder::new(k, n).unwrap();
//! for pkt in packets {
//!     dec.receive(pkt).unwrap();
//!     if dec.is_complete() { break; }
//! }
//! let decoded = dec.decode().unwrap().unwrap();
//! assert_eq!(decoded[0], source[0]);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
// The API predates pedantic documentation/must-use coverage. These narrowly
// scoped compatibility allows keep the strict workspace gate useful without
// forcing unrelated public-API churn in this remediation pass.
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_pass_by_value)]

// Pull in alloc crate for no_std + alloc builds.
#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(feature = "alloc")]
pub mod aligned;
#[cfg(feature = "alloc")]
pub mod decoder;
#[cfg(feature = "alloc")]
pub mod encoder;
pub mod error;
pub mod field;
pub mod kernel;
#[cfg(feature = "alloc")]
pub mod matrix;
#[cfg(feature = "alloc")]
pub mod recoder;

// Re-export the most commonly used types at crate root.
pub use error::RlncError;
pub use field::Gf8;

#[cfg(feature = "alloc")]
pub use aligned::AlignedBuffer;
#[cfg(feature = "alloc")]
pub use decoder::Decoder;
#[cfg(feature = "alloc")]
pub use encoder::{CodedPacket, Encoder, SimpleRng};
#[cfg(feature = "alloc")]
pub use matrix::GfMatrix;
#[cfg(feature = "alloc")]
pub use recoder::Recoder;

/// Returns the name of the SIMD kernel tier active for this build.
///
/// Useful for diagnostics and benchmarking.
///
/// # Example
/// ```rust
/// println!("Active kernel: {}", rlnc_simdx::active_kernel());
/// ```
#[must_use]
pub fn active_kernel() -> &'static str {
    kernel::active_kernel_name()
}
