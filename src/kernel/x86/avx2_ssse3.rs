//! AVX2 + SSSE3 nibble-split kernel (Tier 5 — 256-bit `vpshufb`).
//!
//! Uses aligned load/store when both buffers are 32-byte aligned.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernel::scalar::make_nibble_tables;

#[inline(always)]
fn both_aligned32(x: &[u8], y: &[u8]) -> bool {
    (x.as_ptr() as usize | y.as_ptr() as usize) & 31 == 0
}

/// AXPY using AVX2 `vpshufb`, aligned or unaligned.
///
/// # Safety
/// Requires `avx2` and `ssse3` target features.
#[target_feature(enable = "avx2,ssse3")]
pub(crate) unsafe fn axpy_avx2_ssse3(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        let len = x.len();
        let mut i = 0usize;
        while i + 32 <= len {
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let yv = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                y.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_xor_si256(yv, xv),
            );
            i += 32;
        }
        crate::kernel::scalar::xor_assign(&x[i..], &mut y[i..]);
        _mm256_zeroupper();
        return;
    }

    let (lo_arr, hi_arr) = make_nibble_tables(c);
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo_arr.as_ptr() as *const __m128i));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi_arr.as_ptr() as *const __m128i));
    let mask_lo = _mm256_set1_epi8(0x0Fi8);

    let len = x.len();
    let mut i = 0usize;

    macro_rules! axpy256_ssse3 {
        ($load:ident, $store:ident) => {
            while i + 32 <= len {
                let xv = $load(x.as_ptr().add(i) as *const __m256i);
                let yv = $load(y.as_ptr().add(i) as *const __m256i);
                let xlo = _mm256_and_si256(xv, mask_lo);
                let xhi = _mm256_and_si256(_mm256_srli_epi16(xv, 4), mask_lo);
                let mul = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo_tbl, xlo),
                    _mm256_shuffle_epi8(hi_tbl, xhi),
                );
                $store(
                    y.as_mut_ptr().add(i) as *mut __m256i,
                    _mm256_xor_si256(yv, mul),
                );
                i += 32;
            }
        };
    }

    if both_aligned32(x, y) {
        axpy256_ssse3!(_mm256_load_si256, _mm256_store_si256);
    } else {
        axpy256_ssse3!(_mm256_loadu_si256, _mm256_storeu_si256);
    }

    // 16-byte SSSE3 tail
    while i + 16 <= len {
        let lo128 = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
        let hi128 = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
        let mask128 = _mm_set1_epi8(0x0Fi8);
        let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
        let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        let xlo = _mm_and_si128(xv, mask128);
        let xhi = _mm_and_si128(_mm_srli_epi16(xv, 4), mask128);
        let res = _mm_xor_si128(
            yv,
            _mm_xor_si128(_mm_shuffle_epi8(lo128, xlo), _mm_shuffle_epi8(hi128, xhi)),
        );
        _mm_storeu_si128(y.as_mut_ptr().add(i) as *mut __m128i, res);
        i += 16;
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
    _mm256_zeroupper();
}

/// Scale using AVX2 `vpshufb`, aligned or unaligned.
///
/// # Safety
/// Requires `avx2` and `ssse3` target features.
#[target_feature(enable = "avx2,ssse3")]
pub(crate) unsafe fn scale_avx2_ssse3(c: u8, x: &[u8], y: &mut [u8]) {
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
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo_arr.as_ptr() as *const __m128i));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi_arr.as_ptr() as *const __m128i));
    let mask_lo = _mm256_set1_epi8(0x0Fi8);

    let len = x.len();
    let mut i = 0usize;

    if both_aligned32(x, y) {
        while i + 32 <= len {
            let xv = _mm256_load_si256(x.as_ptr().add(i) as *const __m256i);
            let xlo = _mm256_and_si256(xv, mask_lo);
            let xhi = _mm256_and_si256(_mm256_srli_epi16(xv, 4), mask_lo);
            _mm256_store_si256(
                y.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo_tbl, xlo),
                    _mm256_shuffle_epi8(hi_tbl, xhi),
                ),
            );
            i += 32;
        }
    } else {
        while i + 32 <= len {
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let xlo = _mm256_and_si256(xv, mask_lo);
            let xhi = _mm256_and_si256(_mm256_srli_epi16(xv, 4), mask_lo);
            _mm256_storeu_si256(
                y.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo_tbl, xlo),
                    _mm256_shuffle_epi8(hi_tbl, xhi),
                ),
            );
            i += 32;
        }
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
    _mm256_zeroupper();
}

/// In-place scale using AVX2 nibble-split.
///
/// # Safety
/// Requires `avx2` and `ssse3`.
#[target_feature(enable = "avx2,ssse3")]
pub(crate) unsafe fn scale_inplace_avx2_ssse3(c: u8, y: &mut [u8]) {
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
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo_arr.as_ptr() as *const __m128i));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi_arr.as_ptr() as *const __m128i));
    let mask_lo = _mm256_set1_epi8(0x0Fi8);
    let len = y.len();
    let mut i = 0usize;
    while i + 32 <= len {
        let yv = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        let ylo = _mm256_and_si256(yv, mask_lo);
        let yhi = _mm256_and_si256(_mm256_srli_epi16(yv, 4), mask_lo);
        _mm256_storeu_si256(
            y.as_mut_ptr().add(i) as *mut __m256i,
            _mm256_xor_si256(
                _mm256_shuffle_epi8(lo_tbl, ylo),
                _mm256_shuffle_epi8(hi_tbl, yhi),
            ),
        );
        i += 32;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
    _mm256_zeroupper();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let c = 0x7Fu8;
        let x: Vec<u8> = (0u8..=255).collect();
        let mut y_simd = vec![0x55u8; 256];
        let mut y_scalar = vec![0x55u8; 256];
        unsafe {
            axpy_avx2_ssse3(c, &x, &mut y_simd);
        }
        crate::kernel::scalar::axpy(c, &x, &mut y_scalar);
        assert_eq!(y_simd, y_scalar);
    }
}
