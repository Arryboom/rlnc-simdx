//! WASM SIMD128 kernel (Tier 8 — `i8x16_swizzle` nibble-split).
//!
//! `i8x16_swizzle` is the WebAssembly SIMD equivalent of `pshufb`:
//! a byte-shuffle where out-of-range indices (≥ 16) produce 0, which
//! is exactly correct for nibble-split (nibbles are always 0..15).
//!
//! Processes 16 bytes per iteration.

#[cfg(target_family = "wasm")]
use core::arch::wasm32::*;

/// AXPY using WASM SIMD128 `i8x16_swizzle` (nibble-split).
#[cfg(target_family = "wasm")]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn axpy_wasm(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        // Pure XOR
        let len = x.len();
        let mut i = 0usize;
        while i + 16 <= len {
            let xv = v128_load(x.as_ptr().add(i) as *const v128);
            let yv = v128_load(y.as_ptr().add(i) as *const v128);
            v128_store(y.as_mut_ptr().add(i) as *mut v128, v128_xor(yv, xv));
            i += 16;
        }
        crate::kernel::scalar::xor_assign(&x[i..], &mut y[i..]);
        return;
    }

    let (lo_arr, hi_arr) = crate::kernel::scalar::make_nibble_tables(c);
    let lo_tbl = v128_load(lo_arr.as_ptr() as *const v128);
    let hi_tbl = v128_load(hi_arr.as_ptr() as *const v128);
    let mask_lo = i8x16_splat(0x0F_u8 as i8);

    let len = x.len();
    let mut i = 0usize;

    while i + 16 <= len {
        let xv = v128_load(x.as_ptr().add(i) as *const v128);
        let yv = v128_load(y.as_ptr().add(i) as *const v128);

        let xlo = v128_and(xv, mask_lo);
        let xhi = u8x16_shr(xv, 4);

        let rlo = i8x16_swizzle(lo_tbl, xlo);
        let rhi = i8x16_swizzle(hi_tbl, xhi);

        let mul = v128_xor(rlo, rhi);
        let res = v128_xor(yv, mul);

        v128_store(y.as_mut_ptr().add(i) as *mut v128, res);
        i += 16;
    }

    crate::kernel::scalar::axpy(c, &x[i..], &mut y[i..]);
}

/// Scale using WASM SIMD128 `i8x16_swizzle`.
#[cfg(target_family = "wasm")]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn scale_wasm(c: u8, x: &[u8], y: &mut [u8]) {
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

    let (lo_arr, hi_arr) = crate::kernel::scalar::make_nibble_tables(c);
    let lo_tbl = v128_load(lo_arr.as_ptr() as *const v128);
    let hi_tbl = v128_load(hi_arr.as_ptr() as *const v128);
    let mask_lo = i8x16_splat(0x0F_u8 as i8);

    let len = x.len();
    let mut i = 0usize;

    while i + 16 <= len {
        let xv = v128_load(x.as_ptr().add(i) as *const v128);
        let xlo = v128_and(xv, mask_lo);
        let xhi = u8x16_shr(xv, 4);
        let res = v128_xor(i8x16_swizzle(lo_tbl, xlo), i8x16_swizzle(hi_tbl, xhi));
        v128_store(y.as_mut_ptr().add(i) as *mut v128, res);
        i += 16;
    }

    crate::kernel::scalar::scale(c, &x[i..], &mut y[i..]);
}

/// In-place scale using WASM SIMD128.
#[cfg(target_family = "wasm")]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn scale_inplace_wasm(c: u8, y: &mut [u8]) {
    if c == 0 {
        for yi in y.iter_mut() {
            *yi = 0;
        }
        return;
    }
    if c == 1 {
        return;
    }
    let (lo_arr, hi_arr) = crate::kernel::scalar::make_nibble_tables(c);
    let lo_tbl = v128_load(lo_arr.as_ptr() as *const v128);
    let hi_tbl = v128_load(hi_arr.as_ptr() as *const v128);
    let mask_lo = i8x16_splat(0x0F_u8 as i8);
    let len = y.len();
    let mut i = 0usize;
    while i + 16 <= len {
        let yv = v128_load(y.as_ptr().add(i) as *const v128);
        let ylo = v128_and(yv, mask_lo);
        let yhi = u8x16_shr(yv, 4);
        let res = v128_xor(i8x16_swizzle(lo_tbl, ylo), i8x16_swizzle(hi_tbl, yhi));
        v128_store(y.as_mut_ptr().add(i) as *mut v128, res);
        i += 16;
    }
    crate::kernel::scalar::scale_inplace(c, &mut y[i..]);
}

// Stubs for non-WASM builds (referenced only from no_std wasm compile paths)
#[cfg(not(target_family = "wasm"))]
#[allow(dead_code)]
/// WASM AXPY stub (non-wasm targets).
pub(crate) fn axpy_wasm(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("WASM SIMD128 is only available on wasm32")
}

#[cfg(not(target_family = "wasm"))]
#[allow(dead_code)]
/// WASM scale stub (non-wasm targets).
pub(crate) fn scale_wasm(_c: u8, _x: &[u8], _y: &mut [u8]) {
    unimplemented!("WASM SIMD128 is only available on wasm32")
}

#[cfg(not(target_family = "wasm"))]
#[allow(dead_code)]
/// WASM in-place scale stub (non-wasm targets).
pub(crate) fn scale_inplace_wasm(_c: u8, _y: &mut [u8]) {
    unimplemented!("WASM SIMD128 is only available on wasm32")
}
