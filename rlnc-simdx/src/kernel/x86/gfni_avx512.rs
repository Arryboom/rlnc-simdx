//! GFNI + AVX-512 kernel (Tier 1 — fastest, 512-bit `GF2P8MULB`).
//!
//! Automatically selects aligned (`_mm512_load_si512`) or unaligned
//! (`_mm512_loadu_si512`) intrinsics based on runtime pointer check.
//! When both `x` and `y` are 64-byte aligned (guaranteed by `AlignedBuffer`
//! and the crate's internal data structures), the aligned path is taken,
//! eliminating any possible cache-line-split penalty.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
fn both_aligned64(x: &[u8], y: &[u8]) -> bool {
    (x.as_ptr() as usize | y.as_ptr() as usize) & 63 == 0
}

/// AXPY using GFNI `vgf2p8mulb` (512-bit), aligned or unaligned.
///
/// # Safety
/// Requires `gfni`, `avx512f`, `avx512bw` target features.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn axpy_gfni_avx512(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        axpy_gfni_avx512_xor(x, y);
        return;
    }

    let c_vec = _mm512_set1_epi8(c as i8);
    let len = x.len();
    let mut i = 0usize;

    if both_aligned64(x, y) {
        // ── Aligned path ────────────────────────────────────────────────────
        while i + 256 <= len {
            let x0 = _mm512_load_si512(x.as_ptr().add(i) as *const _);
            let x1 = _mm512_load_si512(x.as_ptr().add(i + 64) as *const _);
            let x2 = _mm512_load_si512(x.as_ptr().add(i + 128) as *const _);
            let x3 = _mm512_load_si512(x.as_ptr().add(i + 192) as *const _);
            let y0 = _mm512_load_si512(y.as_ptr().add(i) as *const _);
            let y1 = _mm512_load_si512(y.as_ptr().add(i + 64) as *const _);
            let y2 = _mm512_load_si512(y.as_ptr().add(i + 128) as *const _);
            let y3 = _mm512_load_si512(y.as_ptr().add(i + 192) as *const _);
            _mm512_store_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_xor_si512(y0, _mm512_gf2p8mul_epi8(x0, c_vec)),
            );
            _mm512_store_si512(
                y.as_mut_ptr().add(i + 64) as *mut _,
                _mm512_xor_si512(y1, _mm512_gf2p8mul_epi8(x1, c_vec)),
            );
            _mm512_store_si512(
                y.as_mut_ptr().add(i + 128) as *mut _,
                _mm512_xor_si512(y2, _mm512_gf2p8mul_epi8(x2, c_vec)),
            );
            _mm512_store_si512(
                y.as_mut_ptr().add(i + 192) as *mut _,
                _mm512_xor_si512(y3, _mm512_gf2p8mul_epi8(x3, c_vec)),
            );
            i += 256;
        }
        while i + 64 <= len {
            let xv = _mm512_load_si512(x.as_ptr().add(i) as *const _);
            let yv = _mm512_load_si512(y.as_ptr().add(i) as *const _);
            _mm512_store_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_xor_si512(yv, _mm512_gf2p8mul_epi8(xv, c_vec)),
            );
            i += 64;
        }
    } else {
        // ── Unaligned path ───────────────────────────────────────────────────
        while i + 256 <= len {
            let x0 = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
            let x1 = _mm512_loadu_si512(x.as_ptr().add(i + 64) as *const _);
            let x2 = _mm512_loadu_si512(x.as_ptr().add(i + 128) as *const _);
            let x3 = _mm512_loadu_si512(x.as_ptr().add(i + 192) as *const _);
            let y0 = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
            let y1 = _mm512_loadu_si512(y.as_ptr().add(i + 64) as *const _);
            let y2 = _mm512_loadu_si512(y.as_ptr().add(i + 128) as *const _);
            let y3 = _mm512_loadu_si512(y.as_ptr().add(i + 192) as *const _);
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_xor_si512(y0, _mm512_gf2p8mul_epi8(x0, c_vec)),
            );
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i + 64) as *mut _,
                _mm512_xor_si512(y1, _mm512_gf2p8mul_epi8(x1, c_vec)),
            );
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i + 128) as *mut _,
                _mm512_xor_si512(y2, _mm512_gf2p8mul_epi8(x2, c_vec)),
            );
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i + 192) as *mut _,
                _mm512_xor_si512(y3, _mm512_gf2p8mul_epi8(x3, c_vec)),
            );
            i += 256;
        }
        while i + 64 <= len {
            let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
            let yv = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_xor_si512(yv, _mm512_gf2p8mul_epi8(xv, c_vec)),
            );
            i += 64;
        }
    }

    // 32-byte GFNI+AVX2 tail
    if i + 32 <= len {
        let c256 = _mm256_set1_epi8(c as i8);
        let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
        let yv = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(
            y.as_mut_ptr().add(i) as *mut __m256i,
            _mm256_xor_si256(yv, _mm256_gf2p8mul_epi8(xv, c256)),
        );
        i += 32;
    }

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
}

/// Fused multi-source AXPY using one destination load/store per vector.
///
/// # Safety
/// Requires `gfni`, `avx512f`, and `avx512bw`; all sources must match the
/// destination length and be disjoint from it.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn axpy_multi_gfni_avx512(coeffs: &[u8], sources: &[&[u8]], y: &mut [u8]) {
    debug_assert_eq!(coeffs.len(), sources.len());
    debug_assert!(sources.iter().all(|source| source.len() == y.len()));
    let len = y.len();
    let mut i = 0usize;

    while i + 64 <= len {
        let mut acc = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
        for source_index in 0..coeffs.len() {
            let coefficient = *coeffs.get_unchecked(source_index);
            if coefficient == 0 {
                continue;
            }
            let source = *sources.get_unchecked(source_index);
            let value = _mm512_loadu_si512(source.as_ptr().add(i) as *const _);
            let product = if coefficient == 1 {
                value
            } else {
                _mm512_gf2p8mul_epi8(value, _mm512_set1_epi8(coefficient as i8))
            };
            acc = _mm512_xor_si512(acc, product);
        }
        _mm512_storeu_si512(y.as_mut_ptr().add(i) as *mut _, acc);
        i += 64;
    }
    if i + 32 <= len {
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
}

/// Vectorized GF dot product.
///
/// # Safety
/// Requires `gfni`, `avx512f`, and `avx512bw`.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn dot_gfni_avx512(a: &[u8], b: &[u8]) -> u8 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm512_setzero_si512();
    let mut i = 0usize;
    while i + 64 <= a.len() {
        let av = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let bv = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, bv));
        i += 64;
    }
    let mut lanes = [0u8; 64];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut _, acc);
    lanes
        .iter()
        .copied()
        .fold(crate::kernel::scalar::dot(&a[i..], &b[i..]), |x, y| x ^ y)
}

/// Pure XOR path for coefficient `c == 1`.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
unsafe fn axpy_gfni_avx512_xor(x: &[u8], y: &mut [u8]) {
    let len = x.len();
    let mut i = 0usize;
    if both_aligned64(x, y) {
        while i + 64 <= len {
            let xv = _mm512_load_si512(x.as_ptr().add(i) as *const _);
            let yv = _mm512_load_si512(y.as_ptr().add(i) as *const _);
            _mm512_store_si512(y.as_mut_ptr().add(i) as *mut _, _mm512_xor_si512(yv, xv));
            i += 64;
        }
    } else {
        while i + 64 <= len {
            let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
            let yv = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
            _mm512_storeu_si512(y.as_mut_ptr().add(i) as *mut _, _mm512_xor_si512(yv, xv));
            i += 64;
        }
    }
    if i + 32 <= len {
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
}

/// Scale using GFNI `vgf2p8mulb` (512-bit), aligned or unaligned.
///
/// # Safety
/// Requires `gfni`, `avx512f`, `avx512bw` target features.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn scale_gfni_avx512(c: u8, x: &[u8], y: &mut [u8]) {
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

    let c_vec = _mm512_set1_epi8(c as i8);
    let len = x.len();
    let mut i = 0usize;

    if both_aligned64(x, y) {
        while i + 64 <= len {
            let xv = _mm512_load_si512(x.as_ptr().add(i) as *const _);
            _mm512_store_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_gf2p8mul_epi8(xv, c_vec),
            );
            i += 64;
        }
    } else {
        while i + 64 <= len {
            let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_gf2p8mul_epi8(xv, c_vec),
            );
            i += 64;
        }
    }

    if i + 32 <= len {
        let c256 = _mm256_set1_epi8(c as i8);
        let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(
            y.as_mut_ptr().add(i) as *mut __m256i,
            _mm256_gf2p8mul_epi8(xv, c256),
        );
        i += 32;
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
}

/// In-place scale: `y[i] = c * y[i]` using GFNI (512-bit).
///
/// Sequential VL-width load/mul/store is safe for full alias (same buffer).
///
/// # Safety
/// Requires `gfni`, `avx512f`, `avx512bw`.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn scale_inplace_gfni_avx512(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let c_vec = _mm512_set1_epi8(c as i8);
    let len = y.len();
    let mut i = 0usize;
    let aligned = (y.as_ptr() as usize) & 63 == 0;
    if aligned {
        while i + 64 <= len {
            let yv = _mm512_load_si512(y.as_ptr().add(i) as *const _);
            _mm512_store_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_gf2p8mul_epi8(yv, c_vec),
            );
            i += 64;
        }
    } else {
        while i + 64 <= len {
            let yv = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
            _mm512_storeu_si512(
                y.as_mut_ptr().add(i) as *mut _,
                _mm512_gf2p8mul_epi8(yv, c_vec),
            );
            i += 64;
        }
    }
    if i + 32 <= len {
        let c256 = _mm256_set1_epi8(c as i8);
        let yv = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(
            y.as_mut_ptr().add(i) as *mut __m256i,
            _mm256_gf2p8mul_epi8(yv, c256),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AlignedBuffer;

    const TAIL_LENGTHS: [usize; 9] = [15, 16, 17, 31, 32, 33, 63, 64, 65];

    fn check_operations(x: &[u8], initial: &[u8], aligned: bool) {
        for c in [1u8, 0xB7] {
            let mut y = AlignedBuffer::from_slice(initial);
            if !aligned {
                let mut backing = vec![0u8; initial.len() + 1];
                backing[1..].copy_from_slice(initial);
                let mut expected = initial.to_vec();
                crate::kernel::scalar::axpy(c, x, &mut expected);
                unsafe { axpy_gfni_avx512(c, x, &mut backing[1..]) };
                assert_eq!(&backing[1..], expected, "axpy c={c:#x}");
            } else {
                let mut expected = initial.to_vec();
                crate::kernel::scalar::axpy(c, x, &mut expected);
                unsafe { axpy_gfni_avx512(c, x, &mut y) };
                assert_eq!(y.as_slice(), expected, "axpy c={c:#x}");
            }
        }

        let c = 0xB7;
        let mut scaled = AlignedBuffer::zeroed(x.len());
        let mut expected = vec![0u8; x.len()];
        crate::kernel::scalar::scale(c, x, &mut expected);
        if aligned {
            unsafe { scale_gfni_avx512(c, x, &mut scaled) };
            assert_eq!(scaled.as_slice(), expected, "scale");
        } else {
            let mut backing = vec![0xA5u8; x.len() + 1];
            unsafe { scale_gfni_avx512(c, x, &mut backing[1..]) };
            assert_eq!(&backing[1..], expected, "scale");
        }

        if aligned {
            let mut inplace = AlignedBuffer::from_slice(initial);
            let mut expected = initial.to_vec();
            crate::kernel::scalar::scale_inplace(c, &mut expected);
            unsafe { scale_inplace_gfni_avx512(c, &mut inplace) };
            assert_eq!(inplace.as_slice(), expected, "scale_inplace");
        } else {
            let mut backing = vec![0u8; initial.len() + 1];
            backing[1..].copy_from_slice(initial);
            let mut expected = initial.to_vec();
            crate::kernel::scalar::scale_inplace(c, &mut expected);
            unsafe { scale_inplace_gfni_avx512(c, &mut backing[1..]) };
            assert_eq!(&backing[1..], expected, "scale_inplace");
        }
    }

    #[test]
    fn axpy_matches_scalar() {
        if !is_x86_feature_detected!("gfni")
            || !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
        {
            return;
        }
        let c = 0xB7u8;
        // Test both aligned and unaligned paths
        let x: Vec<u8> = (0u8..=255).collect();
        for offset in [0usize, 1, 7, 15] {
            let x_src = &x[offset..offset.saturating_add(128).min(256)];
            let len = x_src.len();
            let mut y_simd = vec![0x22u8; len];
            let mut y_scalar = vec![0x22u8; len];
            unsafe {
                axpy_gfni_avx512(c, x_src, &mut y_simd);
            }
            crate::kernel::scalar::axpy(c, x_src, &mut y_scalar);
            assert_eq!(y_simd, y_scalar, "offset={offset}");
        }
    }

    #[test]
    fn aligned_path_taken_for_aligned_buffers() {
        if !is_x86_feature_detected!("gfni")
            || !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
        {
            return;
        }

        let src = AlignedBuffer::from_slice(&[0x42u8; 256]);
        let mut dst = AlignedBuffer::zeroed(256);
        assert!(
            both_aligned64(&src, &dst),
            "AlignedBuffer must be 64-byte aligned"
        );
        unsafe {
            axpy_gfni_avx512(0x03, &src, &mut dst);
        }
        let mut expected = vec![0u8; 256];
        crate::kernel::scalar::axpy(0x03, &src, &mut expected);
        assert_eq!(dst.as_slice(), expected.as_slice());
    }

    #[test]
    fn vector_tails_match_scalar_at_boundaries() {
        if !is_x86_feature_detected!("gfni")
            || !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
        {
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
            assert_ne!(unaligned_source[1..].as_ptr() as usize & 63, 0);
            check_operations(&unaligned_source[1..], &initial, false);
        }
    }
}
