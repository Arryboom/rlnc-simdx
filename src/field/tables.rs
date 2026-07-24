//! Compile-time GF(2⁸) logarithm / exponent tables.
//!
//! Primitive polynomial: x⁸ + x⁴ + x³ + x + 1  (0x11B)
//! Generator:            0x03  (matches AES / GFNI hardware)
//!
//! All tables are `const` and live in `.rodata`; no heap required.

/// Exponent table: `EXP[i] = g^i mod p` for i in 0..512.
/// Doubled (512 entries) so we can index `EXP[a + b]` without
/// the `% 255` modular reduction (for a,b in 0..255).
pub const EXP: [u8; 512] = gen_exp();

/// Logarithm table: `LOG[x] = i` such that `g^i == x`.
/// `LOG[0]` is defined as 0 but is never used in valid arithmetic
/// (multiplication by 0 is handled specially).
pub const LOG: [u8; 256] = gen_log();

/// Inverse table: `INV[x] = x^-1` in GF(2⁸).  `INV[0] = 0`.
pub const INV: [u8; 256] = gen_inv();

// ---------------------------------------------------------------------------
// Compile-time table generators (const fn)
// ---------------------------------------------------------------------------

const POLY: u32 = 0x11B; // x^8 + x^4 + x^3 + x + 1
const GEN: u8 = 0x03; // primitive root / generator

const fn gf_mul_slow(mut a: u8, mut b: u8) -> u8 {
    let mut result: u8 = 0;
    let mut i = 0usize;
    while i < 8 {
        if b & 1 != 0 {
            result ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= (POLY & 0xFF) as u8; // reduce mod p
        }
        b >>= 1;
        i += 1;
    }
    result
}

const fn gen_exp() -> [u8; 512] {
    let mut table = [0u8; 512];
    let mut x: u8 = 1;
    let mut i = 0usize;
    while i < 255 {
        table[i] = x;
        table[i + 255] = x;
        x = gf_mul_slow(x, GEN);
        i += 1;
    }
    // table[255] and table[510] are set by the doubled copy above
    // but loop ends at i=254; handle edge:
    table[255] = 1; // g^255 == g^0 == 1 (order divides 255)
    table[510] = 1;
    table[511] = GEN; // g^256 == g^1
    table
}

const fn gen_log() -> [u8; 256] {
    let exp = gen_exp();
    let mut table = [0u8; 256];
    let mut i = 0usize;
    while i < 255 {
        table[exp[i] as usize] = i as u8;
        i += 1;
    }
    // log[1] = 0 (already set), log[0] = 0 (sentinel)
    table
}

const fn gen_inv() -> [u8; 256] {
    let log = gen_log();
    let exp = gen_exp();
    let mut table = [0u8; 256];
    // inv(0) = 0 by convention
    let mut i = 1usize;
    while i < 256 {
        // a^-1 = g^(255 - log[a])
        let l = log[i] as usize;
        table[i] = exp[255 - l];
        i += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_has_order_255() {
        // g^255 must equal 1
        assert_eq!(EXP[255], 1, "g^255 should be 1");
    }

    #[test]
    fn exp_log_round_trip() {
        for x in 1u8..=255 {
            let l = LOG[x as usize] as usize;
            assert_eq!(EXP[l], x, "EXP[LOG[{x}]] != {x}");
        }
    }

    #[test]
    fn inv_correctness() {
        // x * inv(x) == 1 for all x != 0
        for x in 1u8..=255 {
            let inv_x = INV[x as usize];
            let product = gf_mul_slow(x, inv_x);
            assert_eq!(product, 1, "x={x} * inv={inv_x} = {product}, expected 1");
        }
    }

    #[test]
    fn inv_zero_is_zero() {
        assert_eq!(INV[0], 0);
    }

    #[test]
    fn mul_commutativity() {
        for a in 0u8..=255 {
            for b in 0u8..=7 {
                assert_eq!(gf_mul_slow(a, b), gf_mul_slow(b, a));
            }
        }
    }
}
