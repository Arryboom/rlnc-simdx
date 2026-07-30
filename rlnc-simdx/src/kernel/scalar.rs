//! Pure-Rust scalar fallback kernel.

// These tiny primitives are deliberately forced inline in SIMD tail paths.
#![allow(clippy::inline_always)]
//!
//! Works on any target, `no_std` clean.  Uses the log/exp table for
//! GF(2⁸) multiplication — one byte at a time.

use crate::field::tables::{EXP, LOG};

const fn gf_mul_polynomial(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    let mut bit = 0;
    while bit < 8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let high = a & 0x80;
        a <<= 1;
        if high != 0 {
            a ^= 0x1B;
        }
        b >>= 1;
        bit += 1;
    }
    product
}

const fn build_nibble_tables() -> [([u8; 16], [u8; 16]); 256] {
    let mut tables = [([0u8; 16], [0u8; 16]); 256];
    let mut coefficient = 0usize;
    while coefficient < 256 {
        let mut nibble = 0usize;
        while nibble < 16 {
            tables[coefficient].0[nibble] = gf_mul_polynomial(coefficient as u8, nibble as u8);
            tables[coefficient].1[nibble] =
                gf_mul_polynomial(coefficient as u8, (nibble as u8) << 4);
            nibble += 1;
        }
        coefficient += 1;
    }
    tables
}

/// Pre-generated fixed-coefficient tables. Keeping this 8 KiB object in
/// read-only data avoids rebuilding 30 field products at every SIMD call.
static NIBBLE_TABLES: [([u8; 16], [u8; 16]); 256] = build_nibble_tables();

/// Precompute two 16-entry nibble tables for coefficient `c`.
///
/// For the nibble-split approach used by all SIMD tiers:
/// - `lo[j] = c * j`       for j in 0..16  (low  nibble contribution)
/// - `hi[j] = c * (j<<4)`  for j in 0..16  (high nibble contribution)
///
/// These are also used by SIMD kernels to initialize their shuffle tables.
#[inline]
#[must_use]
#[allow(dead_code)]
pub fn make_nibble_tables(c: u8) -> ([u8; 16], [u8; 16]) {
    NIBBLE_TABLES[c as usize]
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
    let (lo, hi) = &NIBBLE_TABLES[c as usize];
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi ^= lo[(xi & 0x0F) as usize] ^ hi[(xi >> 4) as usize];
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
    let (lo, hi) = &NIBBLE_TABLES[c as usize];
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = lo[(xi & 0x0F) as usize] ^ hi[(xi >> 4) as usize];
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
    let (lo, hi) = &NIBBLE_TABLES[c as usize];
    for yi in y.iter_mut() {
        *yi = lo[(*yi & 0x0F) as usize] ^ hi[(*yi >> 4) as usize];
    }
}

/// Pure XOR: `y[i] ^= x[i]`  (AXPY with `c = 1`).
#[inline]
#[allow(dead_code)]
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
        for c in 0u8..=255 {
            let (lo, hi) = make_nibble_tables(c);
            for x in 0u8..=255 {
                let xlo = (x & 0x0F) as usize;
                let xhi = ((x >> 4) & 0x0F) as usize;
                let result = lo[xlo] ^ hi[xhi];
                assert_eq!(result, gf_mul(c, x), "c=0x{c:02x}, x=0x{x:02x}");
            }
        }
    }
}
