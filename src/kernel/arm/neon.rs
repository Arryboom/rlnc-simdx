//! ARM NEON kernel (AArch64) — nibble-split with `vqtbl1q_u8`.
//!
//! On modern AArch64, `vld1q_u8` / `vst1q_u8` are alignment-agnostic for
//! throughput; a single load/store path is used (no dual aligned branch).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// AXPY using NEON `vqtbl1q_u8` nibble-split (AArch64).
///
/// # Safety
/// Requires `neon` target feature (always present on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn axpy_neon(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        axpy_neon_xor(x, y);
        return;
    }

    let (lo_arr, hi_arr) = crate::kernel::scalar::make_nibble_tables(c);
    let lo_tbl = vld1q_u8(lo_arr.as_ptr());
    let hi_tbl = vld1q_u8(hi_arr.as_ptr());
    let mask_lo = vdupq_n_u8(0x0F);

    let len = x.len();
    let mut i = 0usize;

    while i + 64 <= len {
        let x0 = vld1q_u8(x.as_ptr().add(i));
        let x1 = vld1q_u8(x.as_ptr().add(i + 16));
        let x2 = vld1q_u8(x.as_ptr().add(i + 32));
        let x3 = vld1q_u8(x.as_ptr().add(i + 48));
        let y0 = vld1q_u8(y.as_ptr().add(i));
        let y1 = vld1q_u8(y.as_ptr().add(i + 16));
        let y2 = vld1q_u8(y.as_ptr().add(i + 32));
        let y3 = vld1q_u8(y.as_ptr().add(i + 48));

        vst1q_u8(
            y.as_mut_ptr().add(i),
            veorq_u8(
                y0,
                veorq_u8(
                    vqtbl1q_u8(lo_tbl, vandq_u8(x0, mask_lo)),
                    vqtbl1q_u8(hi_tbl, vshrq_n_u8(x0, 4)),
                ),
            ),
        );
        vst1q_u8(
            y.as_mut_ptr().add(i + 16),
            veorq_u8(
                y1,
                veorq_u8(
                    vqtbl1q_u8(lo_tbl, vandq_u8(x1, mask_lo)),
                    vqtbl1q_u8(hi_tbl, vshrq_n_u8(x1, 4)),
                ),
            ),
        );
        vst1q_u8(
            y.as_mut_ptr().add(i + 32),
            veorq_u8(
                y2,
                veorq_u8(
                    vqtbl1q_u8(lo_tbl, vandq_u8(x2, mask_lo)),
                    vqtbl1q_u8(hi_tbl, vshrq_n_u8(x2, 4)),
                ),
            ),
        );
        vst1q_u8(
            y.as_mut_ptr().add(i + 48),
            veorq_u8(
                y3,
                veorq_u8(
                    vqtbl1q_u8(lo_tbl, vandq_u8(x3, mask_lo)),
                    vqtbl1q_u8(hi_tbl, vshrq_n_u8(x3, 4)),
                ),
            ),
        );
        i += 64;
    }
    while i + 16 <= len {
        let xv = vld1q_u8(x.as_ptr().add(i));
        let yv = vld1q_u8(y.as_ptr().add(i));
        vst1q_u8(
            y.as_mut_ptr().add(i),
            veorq_u8(
                yv,
                veorq_u8(
                    vqtbl1q_u8(lo_tbl, vandq_u8(xv, mask_lo)),
                    vqtbl1q_u8(hi_tbl, vshrq_n_u8(xv, 4)),
                ),
            ),
        );
        i += 16;
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
}

/// Pure XOR path for `c == 1`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_neon_xor(x: &[u8], y: &mut [u8]) {
    let len = x.len();
    let mut i = 0usize;
    while i + 64 <= len {
        let x0 = vld1q_u8(x.as_ptr().add(i));
        let x1 = vld1q_u8(x.as_ptr().add(i + 16));
        let x2 = vld1q_u8(x.as_ptr().add(i + 32));
        let x3 = vld1q_u8(x.as_ptr().add(i + 48));
        let y0 = vld1q_u8(y.as_ptr().add(i));
        let y1 = vld1q_u8(y.as_ptr().add(i + 16));
        let y2 = vld1q_u8(y.as_ptr().add(i + 32));
        let y3 = vld1q_u8(y.as_ptr().add(i + 48));
        vst1q_u8(y.as_mut_ptr().add(i), veorq_u8(y0, x0));
        vst1q_u8(y.as_mut_ptr().add(i + 16), veorq_u8(y1, x1));
        vst1q_u8(y.as_mut_ptr().add(i + 32), veorq_u8(y2, x2));
        vst1q_u8(y.as_mut_ptr().add(i + 48), veorq_u8(y3, x3));
        i += 64;
    }
    while i + 16 <= len {
        let xv = vld1q_u8(x.as_ptr().add(i));
        let yv = vld1q_u8(y.as_ptr().add(i));
        vst1q_u8(y.as_mut_ptr().add(i), veorq_u8(yv, xv));
        i += 16;
    }
    crate::kernel::scalar::xor_assign(&x[i..], &mut y[i..]);
}

/// Scale using NEON `vqtbl1q_u8` (AArch64).
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn scale_neon(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        y.copy_from_slice(x);
        return;
    }

    let (lo_arr, hi_arr) = crate::kernel::scalar::make_nibble_tables(c);
    let lo_tbl = vld1q_u8(lo_arr.as_ptr());
    let hi_tbl = vld1q_u8(hi_arr.as_ptr());
    let mask_lo = vdupq_n_u8(0x0F);

    let len = x.len();
    let mut i = 0usize;

    while i + 16 <= len {
        let xv = vld1q_u8(x.as_ptr().add(i));
        vst1q_u8(
            y.as_mut_ptr().add(i),
            veorq_u8(
                vqtbl1q_u8(lo_tbl, vandq_u8(xv, mask_lo)),
                vqtbl1q_u8(hi_tbl, vshrq_n_u8(xv, 4)),
            ),
        );
        i += 16;
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
}

/// In-place scale using NEON nibble-split.
///
/// # Safety
/// Requires `neon`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn scale_inplace_neon(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let (lo_arr, hi_arr) = crate::kernel::scalar::make_nibble_tables(c);
    let lo_tbl = vld1q_u8(lo_arr.as_ptr());
    let hi_tbl = vld1q_u8(hi_arr.as_ptr());
    let mask_lo = vdupq_n_u8(0x0F);
    let len = y.len();
    let mut i = 0usize;
    while i + 16 <= len {
        let yv = vld1q_u8(y.as_ptr().add(i));
        vst1q_u8(
            y.as_mut_ptr().add(i),
            veorq_u8(
                vqtbl1q_u8(lo_tbl, vandq_u8(yv, mask_lo)),
                vqtbl1q_u8(hi_tbl, vshrq_n_u8(yv, 4)),
            ),
        );
        i += 16;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
}

// Stubs for non-aarch64 builds
#[cfg(not(target_arch = "aarch64"))]
/// NEON AXPY stub (non-AArch64).
pub(crate) fn axpy_neon(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("NEON is only available on AArch64")
}

#[cfg(not(target_arch = "aarch64"))]
/// NEON scale stub (non-AArch64).
pub(crate) fn scale_neon(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("NEON is only available on AArch64")
}

#[cfg(not(target_arch = "aarch64"))]
/// NEON in-place scale stub (non-AArch64).
pub(crate) fn scale_inplace_neon(_c: u8, _y: &mut [u8]) {
    unimplemented!("NEON is only available on AArch64")
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_scalar() {
        let c = 0x57u8;
        let x: Vec<u8> = (0u8..=255).collect();
        let mut y_simd = vec![0x33u8; 256];
        let mut y_scalar = vec![0x33u8; 256];
        unsafe {
            axpy_neon(c, &x, &mut y_simd);
        }
        crate::kernel::scalar::axpy(c, &x, &mut y_scalar);
        assert_eq!(y_simd, y_scalar);
    }

    #[test]
    fn axpy_c_one_is_xor() {
        let x: Vec<u8> = (0u8..64).collect();
        let mut y = vec![0xAAu8; 64];
        let mut y_ref = y.clone();
        unsafe {
            axpy_neon(1, &x, &mut y);
        }
        crate::kernel::scalar::axpy(1, &x, &mut y_ref);
        assert_eq!(y, y_ref);
    }
}
