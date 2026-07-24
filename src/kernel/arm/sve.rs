//! ARM SVE kernel — **experimental / not production-ready**.
//!
//! ## Status (expert review)
//!
//! The previous SVE nibble implementation used a 256-entry table load that is
//! **incorrect** for typical VL (16–64 bytes): `svtbl_u8` only indexes within
//! the table **vector length**, not a full 256-byte map.
//!
//! Until a correct VL-agnostic nibble kernel is implemented and CI-tested on
//! SVE hardware, this module **does not** provide a working AXPY and is
//! **not** selected by dispatch (AArch64 always uses NEON).
//!
//! Do not call these stubs from production code.

/// Placeholder — SVE AXPY is not production-ready.
///
/// # Panics
/// Always panics / unimplemented.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn axpy_sve(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!(
        "SVE AXPY is experimental and disabled; use NEON dispatch. \
         See plans/expert_review.md P1 SVE."
    );
}

/// Placeholder — SVE scale is not production-ready.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn scale_sve(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("SVE scale is experimental and disabled; use NEON dispatch.");
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "sve")))]
/// Stub when SVE is not available at compile time.
pub(crate) fn axpy_sve(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("SVE not available on this target")
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "sve")))]
/// Stub when SVE is not available at compile time.
pub(crate) fn scale_sve(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("SVE not available on this target")
}
