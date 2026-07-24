//! SSSE3 nibble-split kernel (Tier 6 — 128-bit `pshufb`).
//!
//! Uses `_mm_load_si128` / `_mm_store_si128` when buffers are 16-byte aligned.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernel::scalar::make_nibble_tables;

#[inline(always)]
fn both_aligned16(x: &[u8], y: &[u8]) -> bool {
    (x.as_ptr() as usize | y.as_ptr() as usize) & 15 == 0
}

/// AXPY using SSSE3 `pshufb`, aligned or unaligned.
///
/// # Safety
/// Requires `ssse3` target feature.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn axpy_ssse3(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        let len = x.len();
        let mut i = 0usize;
        while i + 16 <= len {
            let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(y.as_mut_ptr().add(i) as *mut __m128i, _mm_xor_si128(yv, xv));
            i += 16;
        }
        crate::kernel::scalar::xor_assign(&x[i..], &mut y[i..]);
        return;
    }

    let (lo_arr, hi_arr) = make_nibble_tables(c);
    let lo_tbl = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
    let hi_tbl = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
    let mask_lo = _mm_set1_epi8(0x0Fi8);

    let len = x.len();
    let mut i = 0usize;

    macro_rules! axpy128_ssse3 {
        ($load:ident, $store:ident) => {
            while i + 16 <= len {
                let xv = $load(x.as_ptr().add(i) as *const __m128i);
                let yv = $load(y.as_ptr().add(i) as *const __m128i);
                let xlo = _mm_and_si128(xv, mask_lo);
                let xhi = _mm_and_si128(_mm_srli_epi16(xv, 4), mask_lo);
                let mul =
                    _mm_xor_si128(_mm_shuffle_epi8(lo_tbl, xlo), _mm_shuffle_epi8(hi_tbl, xhi));
                $store(
                    y.as_mut_ptr().add(i) as *mut __m128i,
                    _mm_xor_si128(yv, mul),
                );
                i += 16;
            }
        };
    }

    if both_aligned16(x, y) {
        axpy128_ssse3!(_mm_load_si128, _mm_store_si128);
    } else {
        axpy128_ssse3!(_mm_loadu_si128, _mm_storeu_si128);
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
}

/// Scale using SSSE3 `pshufb`, aligned or unaligned.
///
/// # Safety
/// Requires `ssse3` target feature.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn scale_ssse3(c: u8, x: &[u8], y: &mut [u8]) {
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

    let (lo_arr, hi_arr) = make_nibble_tables(c);
    let lo_tbl = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
    let hi_tbl = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
    let mask_lo = _mm_set1_epi8(0x0Fi8);

    let len = x.len();
    let mut i = 0usize;

    if both_aligned16(x, y) {
        while i + 16 <= len {
            let xv = _mm_load_si128(x.as_ptr().add(i) as *const __m128i);
            let xlo = _mm_and_si128(xv, mask_lo);
            let xhi = _mm_and_si128(_mm_srli_epi16(xv, 4), mask_lo);
            _mm_store_si128(
                y.as_mut_ptr().add(i) as *mut __m128i,
                _mm_xor_si128(_mm_shuffle_epi8(lo_tbl, xlo), _mm_shuffle_epi8(hi_tbl, xhi)),
            );
            i += 16;
        }
    } else {
        while i + 16 <= len {
            let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            let xlo = _mm_and_si128(xv, mask_lo);
            let xhi = _mm_and_si128(_mm_srli_epi16(xv, 4), mask_lo);
            _mm_storeu_si128(
                y.as_mut_ptr().add(i) as *mut __m128i,
                _mm_xor_si128(_mm_shuffle_epi8(lo_tbl, xlo), _mm_shuffle_epi8(hi_tbl, xhi)),
            );
            i += 16;
        }
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
}

/// In-place scale using SSSE3 nibble-split.
///
/// # Safety
/// Requires `ssse3`.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn scale_inplace_ssse3(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let (lo_arr, hi_arr) = make_nibble_tables(c);
    let lo_tbl = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
    let hi_tbl = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
    let mask_lo = _mm_set1_epi8(0x0Fi8);
    let len = y.len();
    let mut i = 0usize;
    while i + 16 <= len {
        let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        let ylo = _mm_and_si128(yv, mask_lo);
        let yhi = _mm_and_si128(_mm_srli_epi16(yv, 4), mask_lo);
        _mm_storeu_si128(
            y.as_mut_ptr().add(i) as *mut __m128i,
            _mm_xor_si128(_mm_shuffle_epi8(lo_tbl, ylo), _mm_shuffle_epi8(hi_tbl, yhi)),
        );
        i += 16;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn axpy_matches_scalar() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        let c = 0x53u8;
        let x: Vec<u8> = (0u8..=255).collect();
        let mut y_simd = vec![0xAAu8; 256];
        let mut y_scalar = vec![0xAAu8; 256];
        unsafe {
            axpy_ssse3(c, &x, &mut y_simd);
        }
        crate::kernel::scalar::axpy(c, &x, &mut y_scalar);
        assert_eq!(y_simd, y_scalar);
    }
}
