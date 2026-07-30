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
        if i + 16 <= len {
            let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(y.as_mut_ptr().add(i) as *mut __m128i, _mm_xor_si128(yv, xv));
            i += 16;
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

/// Fused multi-source AXPY using one destination load/store per vector.
///
/// # Safety
/// Requires `gfni` and `avx2`; validated sources must match and not overlap `y`.
#[target_feature(enable = "gfni,avx2")]
pub(crate) unsafe fn axpy_multi_gfni_avx2(coeffs: &[u8], sources: &[&[u8]], y: &mut [u8]) {
    debug_assert_eq!(coeffs.len(), sources.len());
    debug_assert!(sources.iter().all(|source| source.len() == y.len()));
    let len = y.len();
    let mut i = 0usize;
    while i + 32 <= len {
        let mut acc = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        for source_index in 0..coeffs.len() {
            let coefficient = *coeffs.get_unchecked(source_index);
            if coefficient == 0 {
                continue;
            }
            let source = *sources.get_unchecked(source_index);
            let value = _mm256_loadu_si256(source.as_ptr().add(i) as *const __m256i);
            let product = if coefficient == 1 {
                value
            } else {
                _mm256_gf2p8mul_epi8(value, _mm256_set1_epi8(coefficient as i8))
            };
            acc = _mm256_xor_si256(acc, product);
        }
        _mm256_storeu_si256(y.as_mut_ptr().add(i) as *mut __m256i, acc);
        i += 32;
    }
    if i + 16 <= len {
        let mut acc = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        for source_index in 0..coeffs.len() {
            let coefficient = *coeffs.get_unchecked(source_index);
            if coefficient == 0 {
                continue;
            }
            let source = *sources.get_unchecked(source_index);
            let value = _mm_loadu_si128(source.as_ptr().add(i) as *const __m128i);
            let product = if coefficient == 1 {
                value
            } else {
                _mm_gf2p8mul_epi8(value, _mm_set1_epi8(coefficient as i8))
            };
            acc = _mm_xor_si128(acc, product);
        }
        _mm_storeu_si128(y.as_mut_ptr().add(i) as *mut __m128i, acc);
        i += 16;
    }
    for source_index in 0..coeffs.len() {
        crate::kernel::scalar::axpy(
            *coeffs.get_unchecked(source_index),
            &sources.get_unchecked(source_index)[i..],
            &mut y[i..],
        );
    }
    _mm256_zeroupper();
}

/// Vectorized GF dot product.
///
/// # Safety
/// Requires `gfni` and `avx2`.
#[target_feature(enable = "gfni,avx2")]
pub(crate) unsafe fn dot_gfni_avx2(a: &[u8], b: &[u8]) -> u8 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= a.len() {
        let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        acc = _mm256_xor_si256(acc, _mm256_gf2p8mul_epi8(av, bv));
        i += 32;
    }
    let mut lanes = [0u8; 32];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let result = lanes
        .iter()
        .copied()
        .fold(crate::kernel::scalar::dot(&a[i..], &b[i..]), |x, y| x ^ y);
    _mm256_zeroupper();
    result
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

    if i + 16 <= len {
        let c128 = _mm_set1_epi8(c as i8);
        let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(
            y.as_mut_ptr().add(i) as *mut __m128i,
            _mm_gf2p8mul_epi8(xv, c128),
        );
        i += 16;
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
    if i + 16 <= len {
        let c128 = _mm_set1_epi8(c as i8);
        let yv = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(
            y.as_mut_ptr().add(i) as *mut __m128i,
            _mm_gf2p8mul_epi8(yv, c128),
        );
        i += 16;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
    _mm256_zeroupper();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AlignedBuffer;

    const TAIL_LENGTHS: [usize; 9] = [15, 16, 17, 31, 32, 33, 63, 64, 65];

    fn check_operations(x: &[u8], initial: &[u8], aligned: bool) {
        for c in [1u8, 0xE3] {
            let mut expected = initial.to_vec();
            crate::kernel::scalar::axpy(c, x, &mut expected);
            if aligned {
                let mut y = AlignedBuffer::from_slice(initial);
                unsafe { axpy_gfni_avx2(c, x, &mut y) };
                assert_eq!(y.as_slice(), expected, "axpy c={c:#x}");
            } else {
                let mut backing = vec![0u8; initial.len() + 1];
                backing[1..].copy_from_slice(initial);
                unsafe { axpy_gfni_avx2(c, x, &mut backing[1..]) };
                assert_eq!(&backing[1..], expected, "axpy c={c:#x}");
            }
        }

        let c = 0xE3;
        let mut expected = vec![0u8; x.len()];
        crate::kernel::scalar::scale(c, x, &mut expected);
        if aligned {
            let mut y = AlignedBuffer::zeroed(x.len());
            unsafe { scale_gfni_avx2(c, x, &mut y) };
            assert_eq!(y.as_slice(), expected, "scale");
        } else {
            let mut backing = vec![0xA5u8; x.len() + 1];
            unsafe { scale_gfni_avx2(c, x, &mut backing[1..]) };
            assert_eq!(&backing[1..], expected, "scale");
        }

        let mut expected = initial.to_vec();
        crate::kernel::scalar::scale_inplace(c, &mut expected);
        if aligned {
            let mut y = AlignedBuffer::from_slice(initial);
            unsafe { scale_inplace_gfni_avx2(c, &mut y) };
            assert_eq!(y.as_slice(), expected, "scale_inplace");
        } else {
            let mut backing = vec![0u8; initial.len() + 1];
            backing[1..].copy_from_slice(initial);
            unsafe { scale_inplace_gfni_avx2(c, &mut backing[1..]) };
            assert_eq!(&backing[1..], expected, "scale_inplace");
        }
    }

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

    #[test]
    fn vector_tails_match_scalar_at_boundaries() {
        if !is_x86_feature_detected!("gfni") || !is_x86_feature_detected!("avx2") {
            return;
        }
        for len in TAIL_LENGTHS {
            let source: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(29)).collect();
            let initial: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
                .collect();
            let aligned_source = AlignedBuffer::from_slice(&source);
            check_operations(&aligned_source, &initial, true);

            let mut unaligned_source = vec![0u8; len + 1];
            unaligned_source[1..].copy_from_slice(&source);
            assert_ne!(unaligned_source[1..].as_ptr() as usize & 31, 0);
            check_operations(&unaligned_source[1..], &initial, false);
        }
    }
}
