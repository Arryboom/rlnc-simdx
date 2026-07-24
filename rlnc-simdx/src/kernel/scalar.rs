//! Pure-Rust scalar fallback kernel.

// These tiny primitives are deliberately forced inline in SIMD tail paths.
#![allow(clippy::inline_always)]
//!
//! Works on any target, `no_std` clean.  Uses the log/exp table for
//! GF(2⁸) multiplication — one byte at a time.

use crate::field::tables::{EXP, LOG};

/// Precompute two 16-entry nibble tables for coefficient `c`.
///
/// For the nibble-split approach used by all SIMD tiers:
/// - `lo[j] = c * j`       for j in 0..16  (low  nibble contribution)
/// - `hi[j] = c * (j<<4)`  for j in 0..16  (high nibble contribution)
///
/// These are also used by SIMD kernels to initialize their shuffle tables.
#[inline]
#[must_use]
pub fn make_nibble_tables(c: u8) -> ([u8; 16], [u8; 16]) {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    if c == 0 {
        return (lo, hi);
    }
    let log_c = LOG[c as usize] as usize;
    for j in 1usize..16 {
        // lo[j] = c * j  (j has no high bit set, so j < 16)
        lo[j] = EXP[log_c + LOG[j as u8 as usize] as usize];
        // hi[j] = c * (j << 4)
        let jhi = (j << 4) as u8;
        if jhi != 0 {
            hi[j] = EXP[log_c + LOG[jhi as usize] as usize];
        }
    }
    (lo, hi)
}

/// GF(2⁸) multiply two scalars.
#[inline(always)]
#[must_use]
// Reference helper used by tests and exposed benchmark internals; production
// kernels call specialized vector/scalar loops directly.
#[allow(dead_code)]
pub fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    EXP[LOG[a as usize] as usize + LOG[b as usize] as usize]
}

// ---------------------------------------------------------------------------
// Core vector kernels — scalar implementation
// ---------------------------------------------------------------------------

/// `y[i] ^= c * x[i]`  over GF(2⁸).
///
/// This is the AXPY ("a·x plus y") primitive that drives all
/// encoding and Gaussian-elimination operations.
pub fn axpy(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    if c == 0 {
        return;
    }
    if c == 1 {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi ^= xi;
        }
        return;
    }
    let log_c = LOG[c as usize] as usize;
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        if xi != 0 {
            *yi ^= EXP[log_c + LOG[xi as usize] as usize];
        }
    }
}

/// `y[i] = c * x[i]`  over GF(2⁸)  (scale, no accumulation).
pub fn scale(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len(), "scale: length mismatch");
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
    let log_c = LOG[c as usize] as usize;
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = if xi != 0 {
            EXP[log_c + LOG[xi as usize] as usize]
        } else {
            0
        };
    }
}

/// In-place scale: `y[i] = c * y[i]`.
pub fn scale_inplace(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let log_c = LOG[c as usize] as usize;
    for yi in y.iter_mut() {
        if *yi != 0 {
            *yi = EXP[log_c + LOG[*yi as usize] as usize];
        }
    }
}

/// Pure XOR: `y[i] ^= x[i]`  (AXPY with `c = 1`).
#[inline]
pub fn xor_assign(x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi ^= xi;
    }
}

/// `sum(a[i] * b[i])`  over GF(2⁸)  (dot product).
#[must_use]
pub fn dot(a: &[u8], b: &[u8]) -> u8 {
    debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc: u8 = 0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai != 0 && bi != 0 {
            acc ^= EXP[LOG[ai as usize] as usize + LOG[bi as usize] as usize];
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_c_zero_noop() {
        let x = [0x12u8, 0x34, 0x56];
        let mut y = [0xAAu8, 0xBB, 0xCC];
        let expected = y;
        axpy(0, &x, &mut y);
        assert_eq!(y, expected);
    }

    #[test]
    fn axpy_c_one_is_xor() {
        let x = [0x12u8, 0x34, 0x56];
        let mut y = [0xAAu8, 0xBB, 0xCC];
        axpy(1, &x, &mut y);
        assert_eq!(y, [0xAAu8 ^ 0x12, 0xBB ^ 0x34, 0xCC ^ 0x56]);
    }

    #[test]
    fn axpy_known_vector() {
        // Simple 4-byte test: y[i] ^= 0x03 * x[i]
        let c = 0x03u8;
        let x = [0x01u8, 0x02, 0x03, 0x04];
        let mut y = [0u8; 4];
        axpy(c, &x, &mut y);
        for i in 0..4 {
            assert_eq!(y[i], gf_mul(c, x[i]));
        }
    }

    #[test]
    fn scale_vs_axpy() {
        let c = 0x53u8;
        let x: Vec<u8> = (1u8..=64).collect();
        let mut by_scale = vec![0u8; 64];
        let mut by_axpy = vec![0u8; 64];
        scale(c, &x, &mut by_scale);
        axpy(c, &x, &mut by_axpy);
        assert_eq!(by_scale, by_axpy);
    }

    #[test]
    fn dot_product() {
        let a = [0x01u8, 0x02, 0x03];
        let b = [0x01u8, 0x02, 0x03];
        // dot(a,b) = a[0]*b[0] ^ a[1]*b[1] ^ a[2]*b[2]
        let expected = gf_mul(1, 1) ^ gf_mul(2, 2) ^ gf_mul(3, 3);
        assert_eq!(dot(&a, &b), expected);
    }

    #[test]
    fn nibble_tables_correctness() {
        let c = 0x53u8;
        let (lo, hi) = make_nibble_tables(c);
        for x in 0u8..=255 {
            let xlo = (x & 0x0F) as usize;
            let xhi = ((x >> 4) & 0x0F) as usize;
            let result = lo[xlo] ^ hi[xhi];
            assert_eq!(result, gf_mul(c, x), "c=0x{c:02x}, x=0x{x:02x}");
        }
    }
}
