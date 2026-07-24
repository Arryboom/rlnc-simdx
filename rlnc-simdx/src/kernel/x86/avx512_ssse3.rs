//! AVX-512BW + SSSE3 nibble-split kernel (Tier 4 — 512-bit `vpshufb`).
//!
//! Uses aligned 512-bit load/store when buffers are 64-byte aligned.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernel::scalar::make_nibble_tables;

#[inline(always)]
fn both_aligned64(x: &[u8], y: &[u8]) -> bool {
    (x.as_ptr() as usize | y.as_ptr() as usize) & 63 == 0
}

/// AXPY using AVX-512BW `vpshufb`, aligned or unaligned.
///
/// # Safety
/// Requires `avx512f`, `avx512bw`, and `ssse3` target features.
#[target_feature(enable = "avx512f,avx512bw,ssse3")]
pub(crate) unsafe fn axpy_avx512_ssse3(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        let len = x.len();
        let mut i = 0usize;
        while i + 64 <= len {
            let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
            let yv = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
            _mm512_storeu_si512(y.as_mut_ptr().add(i) as *mut _, _mm512_xor_si512(yv, xv));
            i += 64;
        }
        crate::kernel::scalar::xor_assign(&x[i..], &mut y[i..]);
        return;
    }

    let (lo_arr, hi_arr) = make_nibble_tables(c);
    let lo128 = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
    let hi128 = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
    let lo_tbl = _mm512_broadcast_i32x4(lo128);
    let hi_tbl = _mm512_broadcast_i32x4(hi128);
    let mask_lo = _mm512_set1_epi8(0x0Fi8);

    let len = x.len();
    let mut i = 0usize;

    macro_rules! axpy512_ssse3 {
        ($load:ident, $store:ident) => {
            while i + 64 <= len {
                let xv = $load(x.as_ptr().add(i) as *const _);
                let yv = $load(y.as_ptr().add(i) as *const _);
                let xlo = _mm512_and_si512(xv, mask_lo);
                let xhi = _mm512_and_si512(_mm512_srli_epi16(xv, 4), mask_lo);
                let mul = _mm512_xor_si512(
                    _mm512_shuffle_epi8(lo_tbl, xlo),
                    _mm512_shuffle_epi8(hi_tbl, xhi),
                );
                $store(y.as_mut_ptr().add(i) as *mut _, _mm512_xor_si512(yv, mul));
                i += 64;
            }
        };
    }

    if both_aligned64(x, y) {
        axpy512_ssse3!(_mm512_load_si512, _mm512_store_si512);
    } else {
        axpy512_ssse3!(_mm512_loadu_si512, _mm512_storeu_si512);
    }

    // 32-byte AVX2 tail
    if i + 32 <= len {
        use super::avx2_ssse3::axpy_avx2_ssse3;
        axpy_avx2_ssse3(c, &x[i..i + 32], &mut y[i..i + 32]);
        i += 32;
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
}

/// Scale using AVX-512BW `vpshufb`, aligned or unaligned.
///
/// # Safety
/// Requires `avx512f`, `avx512bw`, and `ssse3` target features.
#[target_feature(enable = "avx512f,avx512bw,ssse3")]
pub(crate) unsafe fn scale_avx512_ssse3(c: u8, x: &[u8], y: &mut [u8]) {
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
    let lo128 = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
    let hi128 = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
    let lo_tbl = _mm512_broadcast_i32x4(lo128);
    let hi_tbl = _mm512_broadcast_i32x4(hi128);
    let mask_lo = _mm512_set1_epi8(0x0Fi8);

    let len = x.len();
    let mut i = 0usize;

    if both_aligned64(x, y) {
        while i + 64 <= len {
            let xv = _mm512_load_si512(x.as_ptr().add(i) as *const _);
            let xlo = _mm512_and_si512(xv, mask_lo);
            let xhi = _mm512_and_si512(_mm512_srli_epi16(xv, 4), mask_lo);
            _mm512_store_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_xor_si512(
                    _mm512_shuffle_epi8(lo_tbl, xlo),
                    _mm512_shuffle_epi8(hi_tbl, xhi),
                ),
            );
            i += 64;
        }
    } else {
        while i + 64 <= len {
            let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
            let xlo = _mm512_and_si512(xv, mask_lo);
            let xhi = _mm512_and_si512(_mm512_srli_epi16(xv, 4), mask_lo);
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_xor_si512(
                    _mm512_shuffle_epi8(lo_tbl, xlo),
                    _mm512_shuffle_epi8(hi_tbl, xhi),
                ),
            );
            i += 64;
        }
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
}

/// In-place scale using AVX-512BW nibble-split.
///
/// # Safety
/// Requires `avx512f`, `avx512bw`, `ssse3`.
#[target_feature(enable = "avx512f,avx512bw,ssse3")]
pub(crate) unsafe fn scale_inplace_avx512_ssse3(c: u8, y: &mut [u8]) {
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
    let lo128 = _mm_loadu_si128(lo_arr.as_ptr() as *const __m128i);
    let hi128 = _mm_loadu_si128(hi_arr.as_ptr() as *const __m128i);
    let lo_tbl = _mm512_broadcast_i32x4(lo128);
    let hi_tbl = _mm512_broadcast_i32x4(hi128);
    let mask_lo = _mm512_set1_epi8(0x0Fi8);
    let len = y.len();
    let mut i = 0usize;
    while i + 64 <= len {
        let yv = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
        let ylo = _mm512_and_si512(yv, mask_lo);
        let yhi = _mm512_and_si512(_mm512_srli_epi16(yv, 4), mask_lo);
        _mm512_storeu_si512(
            y.as_mut_ptr().add(i) as *mut _,
            _mm512_xor_si512(
                _mm512_shuffle_epi8(lo_tbl, ylo),
                _mm512_shuffle_epi8(hi_tbl, yhi),
            ),
        );
        i += 64;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let c = 0xA3u8;
        let x: Vec<u8> = (0u8..=255).collect();
        let mut y_simd = vec![0x77u8; 256];
        let mut y_scalar = vec![0x77u8; 256];
        unsafe {
            axpy_avx512_ssse3(c, &x, &mut y_simd);
        }
        crate::kernel::scalar::axpy(c, &x, &mut y_scalar);
        assert_eq!(y_simd, y_scalar);
    }
}
