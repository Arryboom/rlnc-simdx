//! GF(2⁸) finite field arithmetic.
//!
//! Primitive polynomial: x⁸ + x⁴ + x³ + x + 1  (0x11B, AES polynomial)
//! Generator: 0x03

pub mod gf8;
pub(crate) mod tables;

pub use gf8::Gf8;
