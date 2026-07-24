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

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
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
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AlignedBuffer;

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
}
