//! `x86`/`x86_64` SIMD kernel module.
//!
//! The parent dispatch layer selects from these crate-private tiers.

pub(crate) mod avx2_ssse3;
pub(crate) mod avx512_ssse3;
pub(crate) mod gfni_avx2;
pub(crate) mod gfni_avx512;
pub(crate) mod gfni_sse;
pub(crate) mod ssse3;
