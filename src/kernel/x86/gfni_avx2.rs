//! GFNI + AVX2 kernel (Tier 2 — 256-bit `GF2P8MULB`).
//!
//! Selects `_mm256_load_si256` / `_mm256_store_si256` when both buffers are
//! 32-byte aligned, falling back to the unaligned variants otherwise.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
fn both_aligned32(x: &[u8], y: &[u8]) -> bool {
    (x.as_ptr() as usize | y.as_ptr() as usize) & 31 == 0
}

/// AXPY using GFNI `vgf2p8mulb` (256-bit), aligned or unaligned.
///
/// # Safety
/// Requires `gfni` and `avx2` target features.
#[target_feature(enable = "gfni,avx2")]
pub(crate) unsafe fn axpy_gfni_avx2(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        // Pure XOR — skip GFNI multiply
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

    let c_vec = _mm256_set1_epi8(c as i8);
    let len = x.len();
    let mut i = 0usize;

    macro_rules! axpy256 {
        ($load:ident, $store:ident) => {
            while i + 128 <= len {
                let x0 = $load(x.as_ptr().add(i) as *const __m256i);
                let x1 = $load(x.as_ptr().add(i + 32) as *const __m256i);
                let x2 = $load(x.as_ptr().add(i + 64) as *const __m256i);
                let x3 = $load(x.as_ptr().add(i + 96) as *const __m256i);
                let y0 = $load(y.as_ptr().add(i) as *const __m256i);
                let y1 = $load(y.as_ptr().add(i + 32) as *const __m256i);
                let y2 = $load(y.as_ptr().add(i + 64) as *const __m256i);
                let y3 = $load(y.as_ptr().add(i + 96) as *const __m256i);
                $store(
                    y.as_mut_ptr().add(i) as *mut __m256i,
                    _mm256_xor_si256(y0, _mm256_gf2p8mul_epi8(x0, c_vec)),
                );
                $store(
                    y.as_mut_ptr().add(i + 32) as *mut __m256i,
                    _mm256_xor_si256(y1, _mm256_gf2p8mul_epi8(x1, c_vec)),
                );
                $store(
                    y.as_mut_ptr().add(i + 64) as *mut __m256i,
                    _mm256_xor_si256(y2, _mm256_gf2p8mul_epi8(x2, c_vec)),
                );
                $store(
                    y.as_mut_ptr().add(i + 96) as *mut __m256i,
                    _mm256_xor_si256(y3, _mm256_gf2p8mul_epi8(x3, c_vec)),
                );
                i += 128;
            }
            while i + 32 <= len {
                let xv = $load(x.as_ptr().add(i) as *const __m256i);
                let yv = $load(y.as_ptr().add(i) as *const __m256i);
                $store(
                    y.as_mut_ptr().add(i) as *mut __m256i,
                    _mm256_xor_si256(yv, _mm256_gf2p8mul_epi8(xv, c_vec)),
                );
                i += 32;
            }
        };
    }

    if both_aligned32(x, y) {
        axpy256!(_mm256_load_si256, _mm256_store_si256);
    } else {
        axpy256!(_mm256_loadu_si256, _mm256_storeu_si256);
    }

    // 16-byte GFNI SSE tail
    if i + 16 <= len {
        let c128 = _mm_set1_epi8(c as i8);
        let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
        let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(
            y.as_mut_ptr().add(i) as *mut __m128i,
            _mm_xor_si128(yv, _mm_gf2p8mul_epi8(xv, c128)),
        );
        i += 16;
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
    _mm256_zeroupper();
}

/// Scale using GFNI `vgf2p8mulb` (256-bit), aligned or unaligned.
///
/// # Safety
/// Requires `gfni` and `avx2` target features.
#[target_feature(enable = "gfni,avx2")]
pub(crate) unsafe fn scale_gfni_avx2(c: u8, x: &[u8], y: &mut [u8]) {
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

    let c_vec = _mm256_set1_epi8(c as i8);
    let len = x.len();
    let mut i = 0usize;

    if both_aligned32(x, y) {
        while i + 32 <= len {
            let xv = _mm256_load_si256(x.as_ptr().add(i) as *const __m256i);
            _mm256_store_si256(
                y.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_gf2p8mul_epi8(xv, c_vec),
            );
            i += 32;
        }
    } else {
        while i + 32 <= len {
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                y.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_gf2p8mul_epi8(xv, c_vec),
            );
            i += 32;
        }
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
    _mm256_zeroupper();
}

/// In-place scale using GFNI (256-bit). Sequential VL load/mul/store — full alias OK.
///
/// # Safety
/// Requires `gfni` and `avx2`.
#[target_feature(enable = "gfni,avx2")]
pub(crate) unsafe fn scale_inplace_gfni_avx2(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let c_vec = _mm256_set1_epi8(c as i8);
    let len = y.len();
    let mut i = 0usize;
    while i + 32 <= len {
        let yv = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(
            y.as_mut_ptr().add(i) as *mut __m256i,
            _mm256_gf2p8mul_epi8(yv, c_vec),
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
        if !is_x86_feature_detected!("gfni") || !is_x86_feature_detected!("avx2") {
            return;
        }
        let c = 0xE3u8;
        let x: Vec<u8> = (0u8..=255).collect();
        let mut y_simd = vec![0x11u8; 256];
        let mut y_scalar = vec![0x11u8; 256];
        unsafe {
            axpy_gfni_avx2(c, &x, &mut y_simd);
        }
        crate::kernel::scalar::axpy(c, &x, &mut y_scalar);
        assert_eq!(y_simd, y_scalar);
    }
}
