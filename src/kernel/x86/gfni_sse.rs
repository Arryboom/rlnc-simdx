//! GFNI + SSE kernel (Tier 3 — 128-bit `GF2P8MULB`).
//!
//! Uses `_mm_load_si128` / `_mm_store_si128` when buffers are 16-byte aligned.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
fn both_aligned16(x: &[u8], y: &[u8]) -> bool {
    (x.as_ptr() as usize | y.as_ptr() as usize) & 15 == 0
}

/// AXPY using GFNI `gf2p8mulb` (128-bit), aligned or unaligned.
///
/// # Safety
/// Requires `gfni` and `sse4.2` target features.
#[target_feature(enable = "gfni,sse4.2")]
pub(crate) unsafe fn axpy_gfni_sse(c: u8, x: &[u8], y: &mut [u8]) {
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

    let c_vec = _mm_set1_epi8(c as i8);
    let len = x.len();
    let mut i = 0usize;

    macro_rules! axpy128 {
        ($load:ident, $store:ident) => {
            while i + 64 <= len {
                let x0 = $load(x.as_ptr().add(i) as *const __m128i);
                let x1 = $load(x.as_ptr().add(i + 16) as *const __m128i);
                let x2 = $load(x.as_ptr().add(i + 32) as *const __m128i);
                let x3 = $load(x.as_ptr().add(i + 48) as *const __m128i);
                let y0 = $load(y.as_ptr().add(i) as *const __m128i);
                let y1 = $load(y.as_ptr().add(i + 16) as *const __m128i);
                let y2 = $load(y.as_ptr().add(i + 32) as *const __m128i);
                let y3 = $load(y.as_ptr().add(i + 48) as *const __m128i);
                $store(
                    y.as_mut_ptr().add(i) as *mut __m128i,
                    _mm_xor_si128(y0, _mm_gf2p8mul_epi8(x0, c_vec)),
                );
                $store(
                    y.as_mut_ptr().add(i + 16) as *mut __m128i,
                    _mm_xor_si128(y1, _mm_gf2p8mul_epi8(x1, c_vec)),
                );
                $store(
                    y.as_mut_ptr().add(i + 32) as *mut __m128i,
                    _mm_xor_si128(y2, _mm_gf2p8mul_epi8(x2, c_vec)),
                );
                $store(
                    y.as_mut_ptr().add(i + 48) as *mut __m128i,
                    _mm_xor_si128(y3, _mm_gf2p8mul_epi8(x3, c_vec)),
                );
                i += 64;
            }
            while i + 16 <= len {
                let xv = $load(x.as_ptr().add(i) as *const __m128i);
                let yv = $load(y.as_ptr().add(i) as *const __m128i);
                $store(
                    y.as_mut_ptr().add(i) as *mut __m128i,
                    _mm_xor_si128(yv, _mm_gf2p8mul_epi8(xv, c_vec)),
                );
                i += 16;
            }
        };
    }

    if both_aligned16(x, y) {
        axpy128!(_mm_load_si128, _mm_store_si128);
    } else {
        axpy128!(_mm_loadu_si128, _mm_storeu_si128);
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
}

/// Scale using GFNI `gf2p8mulb` (128-bit), aligned or unaligned.
///
/// # Safety
/// Requires `gfni` and `sse4.2` target features.
#[target_feature(enable = "gfni,sse4.2")]
pub(crate) unsafe fn scale_gfni_sse(c: u8, x: &[u8], y: &mut [u8]) {
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

    let c_vec = _mm_set1_epi8(c as i8);
    let len = x.len();
    let mut i = 0usize;

    if both_aligned16(x, y) {
        while i + 16 <= len {
            let xv = _mm_load_si128(x.as_ptr().add(i) as *const __m128i);
            _mm_store_si128(
                y.as_mut_ptr().add(i) as *mut __m128i,
                _mm_gf2p8mul_epi8(xv, c_vec),
            );
            i += 16;
        }
    } else {
        while i + 16 <= len {
            let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(
                y.as_mut_ptr().add(i) as *mut __m128i,
                _mm_gf2p8mul_epi8(xv, c_vec),
            );
            i += 16;
        }
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
}

/// In-place scale using GFNI (128-bit).
///
/// # Safety
/// Requires `gfni` and `sse4.2`.
#[target_feature(enable = "gfni,sse4.2")]
pub(crate) unsafe fn scale_inplace_gfni_sse(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let c_vec = _mm_set1_epi8(c as i8);
    let len = y.len();
    let mut i = 0usize;
    while i + 16 <= len {
        let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(
            y.as_mut_ptr().add(i) as *mut __m128i,
            _mm_gf2p8mul_epi8(yv, c_vec),
        );
        i += 16;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_scalar() {
        if !is_x86_feature_detected!("gfni") {
            return;
        }
        let c = 0x03u8;
        let x: Vec<u8> = (0u8..=255).collect();
        let mut y_simd = vec![0xFFu8; 256];
        let mut y_scalar = vec![0xFFu8; 256];
        unsafe {
            axpy_gfni_sse(c, &x, &mut y_simd);
        }
        crate::kernel::scalar::axpy(c, &x, &mut y_scalar);
        assert_eq!(y_simd, y_scalar);
    }
}
