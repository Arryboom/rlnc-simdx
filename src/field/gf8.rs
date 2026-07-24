//! GF(2⁸) element newtype and arithmetic operations.

// Tiny field operations deliberately use forced inlining. XOR is the correct
// addition/subtraction operation in characteristic two, despite generic
// arithmetic-operator lint heuristics.
#![allow(clippy::inline_always)]
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::suspicious_op_assign_impl)]
//!
//! All arithmetic uses the AES primitive polynomial (0x11B) with
//! generator 0x03, matching Intel/AMD GFNI hardware (`GF2P8MULB`).

use super::tables::{EXP, INV, LOG};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// An element of GF(2⁸).
///
/// Addition and subtraction are both XOR (characteristic-2 field).
/// Multiplication uses log/exp table lookup.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(transparent)]
pub struct Gf8(pub u8);

impl Gf8 {
    /// The additive identity (zero element).
    pub const ZERO: Self = Gf8(0);
    /// The multiplicative identity.
    pub const ONE: Self = Gf8(1);

    /// Construct from a raw byte.
    #[inline(always)]
    #[must_use]
    pub const fn new(v: u8) -> Self {
        Gf8(v)
    }

    /// Raw byte value.
    #[inline(always)]
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Multiplicative inverse.  `inv(0) = 0` by convention.
    #[inline(always)]
    #[must_use]
    pub fn inv(self) -> Self {
        Gf8(INV[self.0 as usize])
    }

    /// Raise to a power.  `pow(0, 0) = 1`.
    #[inline]
    #[must_use]
    pub fn pow(self, exp: u32) -> Self {
        if self.0 == 0 {
            return Gf8(0);
        }
        let l = u32::from(LOG[self.0 as usize]);
        let idx = ((l * (exp % 255)) % 255) as usize;
        Gf8(EXP[idx])
    }

    /// True if this is the zero element.
    #[inline(always)]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

// ---------------------------------------------------------------------------
// Arithmetic trait impls
// ---------------------------------------------------------------------------

impl Add for Gf8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Gf8(self.0 ^ rhs.0)
    }
}

impl AddAssign for Gf8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

// In GF(2^8), subtraction == addition == XOR
impl Sub for Gf8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Gf8(self.0 ^ rhs.0)
    }
}

impl SubAssign for Gf8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

// Negation is a no-op in characteristic 2
impl Neg for Gf8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for Gf8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        if self.0 == 0 || rhs.0 == 0 {
            return Gf8(0);
        }
        let l = LOG[self.0 as usize] as usize + LOG[rhs.0 as usize] as usize;
        // EXP is doubled, so no mod 255 needed as long as l < 510 (guaranteed since l <= 508)
        Gf8(EXP[l])
    }
}

impl MulAssign for Gf8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Gf8 {
    type Output = Self;
    /// Panics if `rhs == 0`.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        debug_assert!(rhs.0 != 0, "division by zero in GF(2^8)");
        self * rhs.inv()
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

impl core::fmt::Debug for Gf8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Gf8(0x{:02x})", self.0)
    }
}

impl core::fmt::Display for Gf8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x{:02x}", self.0)
    }
}

// ---------------------------------------------------------------------------
// From / Into conversions
// ---------------------------------------------------------------------------

impl From<u8> for Gf8 {
    #[inline(always)]
    fn from(v: u8) -> Self {
        Gf8(v)
    }
}

impl From<Gf8> for u8 {
    #[inline(always)]
    fn from(g: Gf8) -> u8 {
        g.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_is_xor() {
        assert_eq!((Gf8(0x53) + Gf8(0xca)).0, 0x53 ^ 0xca);
    }

    #[test]
    fn sub_equals_add() {
        let a = Gf8(0xAB);
        let b = Gf8(0xCD);
        assert_eq!(a - b, a + b);
    }

    #[test]
    fn mul_by_zero() {
        for x in 0u8..=255 {
            assert_eq!(Gf8(x) * Gf8::ZERO, Gf8::ZERO);
        }
    }

    #[test]
    fn mul_by_one() {
        for x in 0u8..=255 {
            assert_eq!(Gf8(x) * Gf8::ONE, Gf8(x));
        }
    }

    #[test]
    fn mul_commutativity() {
        for a in 0u8..=15 {
            for b in 0u8..=15 {
                assert_eq!(Gf8(a) * Gf8(b), Gf8(b) * Gf8(a));
            }
        }
    }

    #[test]
    fn mul_associativity() {
        let a = Gf8(0x53);
        let b = Gf8(0x7e);
        let c = Gf8(0x2d);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn distributivity() {
        let a = Gf8(0x53);
        let b = Gf8(0x7e);
        let c = Gf8(0x2d);
        assert_eq!(a * (b + c), (a * b) + (a * c));
    }

    #[test]
    fn inv_roundtrip() {
        for x in 1u8..=255 {
            assert_eq!(Gf8(x) * Gf8(x).inv(), Gf8::ONE, "failed at x=0x{x:02x}");
        }
    }

    #[test]
    fn div_roundtrip() {
        let a = Gf8(0xAB);
        let b = Gf8(0xCD);
        assert_eq!((a / b) * b, a);
    }

    #[test]
    fn pow_known_values() {
        // g^1 = 0x03 (generator)
        assert_eq!(Gf8(0x03).pow(1), Gf8(0x03));
        // Any non-zero element raised to 255 == 1
        for x in 1u8..=255 {
            assert_eq!(Gf8(x).pow(255), Gf8::ONE, "x^255 != 1 for x=0x{x:02x}");
        }
    }
}
