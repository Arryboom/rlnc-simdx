//! Kernel dispatch layer — **safe public API** + runtime CPU feature detection.
//!
//! ## For application code (safe, no `unsafe`)
//!
//! Use only:
//!
//! - [`axpy`] — `y[i] ^= c * x[i]` (equal lengths; **non-overlapping** buffers)
//! - [`scale`] — `y[i] = c * x[i]` (equal lengths; **non-overlapping**; use
//!   [`scale_inplace`] for in-place)
//! - [`scale_inplace`] — `y[i] = c * y[i]`
//! - [`axpy_multi`] — blocked multi-source AXPY for encoding
//! - [`dot`] — GF(2⁸) dot product (scalar implementation)
//! - [`active_kernel_name`] — which SIMD tier is active
//!
//! Length and overlap are **asserted in release** on the safe wrappers.
//! Tier modules (`x86`, `arm`, `wasm`) are **`pub(crate)`** — not part of the
//! external API.
//!
//! ## Dispatch
//!
//! All SIMD variants are compiled in. On the **first** call to [`axpy`] /
//! [`scale`] / [`scale_inplace`] (with `std`), the best tier is selected via
//! `is_x86_feature_detected!` and cached in `OnceLock<KernelSet>`.
//!
//! ## Dispatch priority (`x86_64`)
//!
//! | Tier | Runtime check                          | Width   | Instruction     |
//! |------|----------------------------------------|---------|-----------------|
//! |  1   | gfni + avx512f + avx512bw              | 512-bit | GF2P8MULB zmm  |
//! |  2   | gfni + avx2                            | 256-bit | GF2P8MULB ymm  |
//! |  3   | gfni + sse4.2                          | 128-bit | GF2P8MULB xmm  |
//! |  4   | avx512f + avx512bw + ssse3             | 512-bit | vpshufb zmm    |
//! |  5   | avx2 + ssse3                           | 256-bit | vpshufb ymm    |
//! |  6   | ssse3                                  | 128-bit | pshufb xmm     |
//! |  7   | neon (`AArch64`)                       | 128-bit | `vqtbl1q`      |
//! |  8   | wasm simd128 (compile-time)            | 128-bit | `i8x16_swizzle` |
//! |  9   | scalar (universal fallback)            | 1 byte  | table lookup   |
//!
//! ## `no_std` targets
//!
//! `OnceLock` requires `std`. Without `std`, dispatch uses compile-time
//! `#[cfg(target_feature)]` selection (bare-metal / embedded).

// The existing cache-blocked AXPY implementation intentionally keeps its block
// size declaration adjacent to the loop. Preserve that hot-loop source shape.
#![allow(clippy::items_after_statements)]

/// Unstable scalar kernel internals exposed only for workspace benchmarks.
///
/// This module is not covered by the crate's stable API guarantees and may
/// change or disappear without a semver-major release.
#[cfg(feature = "bench-internals")]
pub mod scalar;
#[cfg(not(feature = "bench-internals"))]
pub(crate) mod scalar;

#[cfg(test)]
mod proptest;

// Tier kernels are crate-private; external code must use the safe wrappers
// below (`axpy` / `scale` / `scale_inplace`), which enforce length + overlap.
// Raw intrinsic code intentionally uses ISA wildcard imports, explicit pointer
// casts, fixed alignment masks, and aggressive inlining. Rewriting those hot
// loops solely to satisfy style lints risks code-generation regressions.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::incompatible_msrv,
    clippy::inline_always,
    clippy::ptr_as_ptr,
    clippy::unreadable_literal,
    clippy::verbose_bit_mask,
    clippy::wildcard_imports
)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm;

pub(crate) mod wasm;

// ---------------------------------------------------------------------------
// Kernel function-pointer types
// ---------------------------------------------------------------------------

/// Signature for `axpy`: `y[i] ^= c * x[i]` over GF(2⁸).
pub(crate) type AxpyFn = unsafe fn(c: u8, x: &[u8], y: &mut [u8]);
/// Signature for `scale`: `y[i] = c * x[i]` over GF(2⁸).
pub(crate) type ScaleFn = unsafe fn(c: u8, x: &[u8], y: &mut [u8]);
/// Signature for in-place scale: `y[i] = c * y[i]`.
pub(crate) type ScaleInplaceFn = unsafe fn(c: u8, y: &mut [u8]);

/// A resolved set of kernel function pointers for the detected CPU tier.
pub(crate) struct KernelSet {
    /// Best available axpy kernel.
    pub(crate) axpy: AxpyFn,
    /// Best available scale kernel.
    pub(crate) scale: ScaleFn,
    /// Best available in-place scale kernel.
    pub(crate) scale_inplace: ScaleInplaceFn,
    /// Human-readable tier name for diagnostics.
    pub(crate) name: &'static str,
}

// ---------------------------------------------------------------------------
// Runtime detection (std targets only)
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
mod runtime {
    #[cfg(target_arch = "aarch64")]
    use super::arm;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::x86;
    #[cfg(not(target_arch = "aarch64"))]
    use super::{scalar_axpy_wrapper, scalar_scale_inplace_wrapper, scalar_scale_wrapper};
    use super::{AxpyFn, KernelSet, ScaleFn, ScaleInplaceFn};
    use std::sync::OnceLock;

    static KERNEL: OnceLock<KernelSet> = OnceLock::new();

    /// Return (or initialise) the globally cached best kernel set.
    pub(crate) fn get() -> &'static KernelSet {
        KERNEL.get_or_init(detect)
    }

    /// Probe the CPU at runtime and return the highest-tier kernel set.
    fn detect() -> KernelSet {
        // ── x86 / x86_64 ────────────────────────────────────────────────────
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Tier 1 — GFNI + AVX-512
            if is_x86_feature_detected!("gfni")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
            {
                return KernelSet {
                    axpy: x86::gfni_avx512::axpy_gfni_avx512 as AxpyFn,
                    scale: x86::gfni_avx512::scale_gfni_avx512 as ScaleFn,
                    scale_inplace: x86::gfni_avx512::scale_inplace_gfni_avx512 as ScaleInplaceFn,
                    name: "gfni+avx512 (tier1)",
                };
            }

            // Tier 2 — GFNI + AVX2
            if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx2") {
                return KernelSet {
                    axpy: x86::gfni_avx2::axpy_gfni_avx2 as AxpyFn,
                    scale: x86::gfni_avx2::scale_gfni_avx2 as ScaleFn,
                    scale_inplace: x86::gfni_avx2::scale_inplace_gfni_avx2 as ScaleInplaceFn,
                    name: "gfni+avx2 (tier2)",
                };
            }

            // Tier 3 — GFNI + SSE4.2
            if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("sse4.2") {
                return KernelSet {
                    axpy: x86::gfni_sse::axpy_gfni_sse as AxpyFn,
                    scale: x86::gfni_sse::scale_gfni_sse as ScaleFn,
                    scale_inplace: x86::gfni_sse::scale_inplace_gfni_sse as ScaleInplaceFn,
                    name: "gfni+sse4.2 (tier3)",
                };
            }

            // Tier 4 — AVX-512 + SSSE3
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("ssse3")
            {
                return KernelSet {
                    axpy: x86::avx512_ssse3::axpy_avx512_ssse3 as AxpyFn,
                    scale: x86::avx512_ssse3::scale_avx512_ssse3 as ScaleFn,
                    scale_inplace: x86::avx512_ssse3::scale_inplace_avx512_ssse3 as ScaleInplaceFn,
                    name: "avx512+ssse3 (tier4)",
                };
            }

            // Tier 5 — AVX2 + SSSE3
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("ssse3") {
                return KernelSet {
                    axpy: x86::avx2_ssse3::axpy_avx2_ssse3 as AxpyFn,
                    scale: x86::avx2_ssse3::scale_avx2_ssse3 as ScaleFn,
                    scale_inplace: x86::avx2_ssse3::scale_inplace_avx2_ssse3 as ScaleInplaceFn,
                    name: "avx2+ssse3 (tier5)",
                };
            }

            // Tier 6 — SSSE3
            if is_x86_feature_detected!("ssse3") {
                return KernelSet {
                    axpy: x86::ssse3::axpy_ssse3 as AxpyFn,
                    scale: x86::ssse3::scale_ssse3 as ScaleFn,
                    scale_inplace: x86::ssse3::scale_inplace_ssse3 as ScaleInplaceFn,
                    name: "ssse3 (tier6)",
                };
            }
        }

        // ── AArch64 ─────────────────────────────────────────────────────────
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on all AArch64 — SVE is experimental (not wired).
            return KernelSet {
                axpy: arm::neon::axpy_neon as AxpyFn,
                scale: arm::neon::scale_neon as ScaleFn,
                scale_inplace: arm::neon::scale_inplace_neon as ScaleInplaceFn,
                name: "neon (tier7)",
            };
        }

        // ── WASM SIMD128 ─────────────────────────────────────────────────────
        // Runtime feature detection is unavailable on wasm32; compile-time path.

        // ── Scalar fallback ──────────────────────────────────────────────────
        // AArch64 returned the mandatory NEON kernel above, so compiling this
        // fallback there would only create unreachable-code/dead-import noise.
        #[cfg(not(target_arch = "aarch64"))]
        KernelSet {
            axpy: scalar_axpy_wrapper as AxpyFn,
            scale: scalar_scale_wrapper as ScaleFn,
            scale_inplace: scalar_scale_inplace_wrapper as ScaleInplaceFn,
            name: "scalar (tier9)",
        }
    }
}

// ---------------------------------------------------------------------------
// Thin wrappers so scalar fns have the right unsafe fn signature
// ---------------------------------------------------------------------------

#[cfg(all(feature = "std", not(target_arch = "aarch64")))]
unsafe fn scalar_axpy_wrapper(c: u8, x: &[u8], y: &mut [u8]) {
    scalar::axpy(c, x, y);
}

#[cfg(all(feature = "std", not(target_arch = "aarch64")))]
unsafe fn scalar_scale_wrapper(c: u8, x: &[u8], y: &mut [u8]) {
    scalar::scale(c, x, y);
}

#[cfg(all(feature = "std", not(target_arch = "aarch64")))]
unsafe fn scalar_scale_inplace_wrapper(c: u8, y: &mut [u8]) {
    scalar::scale_inplace(c, y);
}

// ---------------------------------------------------------------------------
// Public kernel API
// ---------------------------------------------------------------------------

/// `y[i] ^= c * x[i]`  in GF(2⁸) — the fundamental RLNC primitive.
///
/// On `std` targets the best SIMD kernel is selected at first call and
/// cached for all subsequent calls.  On `no_std` targets the kernel is
/// selected at compile time from the available `target_feature` flags.
///
/// # Panics
/// - Panics if `x.len() != y.len()`.
/// - Panics if `x` and `y` memory ranges **overlap** (including full alias).
///   Use a temporary buffer for in-place-style work; `c == 1` still requires
///   disjoint ranges.
///
/// # Aliasing
/// Buffers must be completely disjoint. Overlap is checked in **release**
/// builds (pointer compare only).
#[inline]
pub fn axpy(c: u8, x: &[u8], y: &mut [u8]) {
    assert_eq!(
        x.len(),
        y.len(),
        "rlnc_simdx::kernel::axpy: length mismatch"
    );
    assert!(
        !ranges_overlap(x.as_ptr(), x.len(), y.as_ptr(), y.len()),
        "rlnc_simdx::kernel::axpy: overlapping buffers are not allowed"
    );
    #[cfg(feature = "std")]
    // SAFETY: runtime::get() only returns a kernel verified for this CPU.
    // Length + non-overlap enforced above.
    unsafe {
        (runtime::get().axpy)(c, x, y);
    }

    #[cfg(not(feature = "std"))]
    axpy_static(c, x, y)
}

/// `y[i] = c * x[i]`  in GF(2⁸).
///
/// # Panics
/// - Panics if `x.len() != y.len()`.
/// - Panics if `x` and `y` memory ranges **overlap** (including full alias).
///   For in-place multiply use [`scale_inplace`].
///
/// # Aliasing
/// Buffers must be completely disjoint. Overlap is checked in **release**
/// builds (pointer compare only).
#[inline]
pub fn scale(c: u8, x: &[u8], y: &mut [u8]) {
    assert_eq!(
        x.len(),
        y.len(),
        "rlnc_simdx::kernel::scale: length mismatch"
    );
    assert!(
        !ranges_overlap(x.as_ptr(), x.len(), y.as_ptr(), y.len()),
        "rlnc_simdx::kernel::scale: overlapping buffers are not allowed; use scale_inplace for in-place"
    );
    #[cfg(feature = "std")]
    unsafe {
        (runtime::get().scale)(c, x, y);
    }

    #[cfg(not(feature = "std"))]
    scale_static(c, x, y)
}

/// In-place scale: `y[i] = c * y[i]`  in GF(2⁸).
///
/// Prefer this over [`scale`] when the source and destination are the same buffer
/// (e.g. pivot normalisation in Gaussian elimination). Uses the same SIMD tier
/// as [`scale`] (GFNI / nibble-split / NEON) via runtime dispatch.
#[inline]
pub fn scale_inplace(c: u8, y: &mut [u8]) {
    #[cfg(feature = "std")]
    // SAFETY: runtime::get() only returns a kernel verified for this CPU.
    unsafe {
        (runtime::get().scale_inplace)(c, y);
    }

    #[cfg(not(feature = "std"))]
    scale_inplace_static(c, y)
}

/// Multi-source fused AXPY with improved cache behaviour:
/// for each chunk of the destination, apply `y ^= c_i * source_i` for all `i`.
///
/// `coeffs.len() == sources.len()`, every `sources[i].len() == y.len()`, and
/// every source's full memory range must be completely disjoint from `y`.
/// Sources may overlap each other because they are read-only.
///
/// # Panics
/// - Panics if `coeffs.len() != sources.len()`.
/// - Panics if any source length differs from `y.len()`; the message identifies
///   the source index.
/// - Panics if any source memory range overlaps `y` (including full alias); the
///   message identifies the source index. This rule applies even when that
///   source's coefficient is zero.
#[inline]
pub fn axpy_multi(coeffs: &[u8], sources: &[&[u8]], y: &mut [u8]) {
    assert_eq!(
        coeffs.len(),
        sources.len(),
        "axpy_multi: coeffs/sources len"
    );

    // Complete every validation pass before mutating y. Lengths are checked
    // first so all ranges passed to the overlap check have the expected size.
    for (i, src) in sources.iter().enumerate() {
        assert_eq!(
            src.len(),
            y.len(),
            "axpy_multi: source[{i}] length mismatch"
        );
    }
    for (i, src) in sources.iter().enumerate() {
        assert!(
            !ranges_overlap(src.as_ptr(), src.len(), y.as_ptr(), y.len()),
            "axpy_multi: source[{i}] overlaps destination"
        );
    }

    #[cfg(feature = "std")]
    let axpy_kernel = runtime::get().axpy;

    // Cache-blocked loop interchange: keep a destination chunk hot while
    // streaming all sources (better for large symbols / many sources).
    const BLOCK: usize = 4096;
    let n = y.len();
    let mut off = 0usize;
    while off < n {
        let end = (off + BLOCK).min(n);
        for (i, &c) in coeffs.iter().enumerate() {
            if c != 0 {
                let source_block = &sources[i][off..end];
                let destination_block = &mut y[off..end];

                #[cfg(feature = "std")]
                // SAFETY: runtime dispatch verified the kernel's CPU features;
                // validation completed before mutation; disjoint full source and
                // destination ranges imply that their corresponding subranges
                // are also disjoint.
                unsafe {
                    axpy_kernel(c, source_block, destination_block);
                }

                #[cfg(not(feature = "std"))]
                axpy_static(c, source_block, destination_block);
            }
        }
        off = end;
    }
}

/// Returns true if two non-empty byte ranges share any byte (including full alias).
/// Empty ranges never overlap. Used by the **safe** public API only.
#[inline]
fn ranges_overlap(a: *const u8, a_len: usize, b: *const u8, b_len: usize) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a0 = a as usize;
    let b0 = b as usize;
    let a1 = a0 + a_len;
    let b1 = b0 + b_len;
    a0 < b1 && b0 < a1
}

/// `sum(a[i] * b[i])`  in GF(2⁸).  Always scalar.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn dot(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "rlnc_simdx::kernel::dot: length mismatch");
    scalar::dot(a, b)
}

/// Returns the name of the currently active kernel tier.
#[must_use]
pub fn active_kernel_name() -> &'static str {
    #[cfg(feature = "std")]
    {
        runtime::get().name
    }

    #[cfg(not(feature = "std"))]
    {
        active_kernel_name_static()
    }
}

// ---------------------------------------------------------------------------
// Compile-time fallback dispatch (no_std targets)
// Used when std is unavailable (embedded, WASM without SIMD, etc.)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "std"))]
#[inline]
fn axpy_static(c: u8, x: &[u8], y: &mut [u8]) {
    // WASM SIMD128
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    // SAFETY: this branch is compiled only when SIMD128 is enabled.
    return unsafe { wasm::simd128::axpy_wasm(c, x, y) };

    // AArch64 NEON (always present on aarch64)
    #[cfg(target_arch = "aarch64")]
    return unsafe { arm::neon::axpy_neon(c, x, y) };

    // x86 compile-time fallback chain
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        return unsafe { x86::gfni_avx512::axpy_gfni_avx512(c, x, y) };
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return unsafe { x86::gfni_avx2::axpy_gfni_avx2(c, x, y) };
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "ssse3",
            not(target_feature = "gfni")
        ))]
        return unsafe { x86::avx2_ssse3::axpy_avx2_ssse3(c, x, y) };
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        return unsafe { x86::ssse3::axpy_ssse3(c, x, y) };
    }

    // Universal scalar fallback
    scalar::axpy(c, x, y)
}

#[cfg(not(feature = "std"))]
#[inline]
fn scale_static(c: u8, x: &[u8], y: &mut [u8]) {
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    // SAFETY: this branch is compiled only when SIMD128 is enabled.
    return unsafe { wasm::simd128::scale_wasm(c, x, y) };

    #[cfg(target_arch = "aarch64")]
    return unsafe { arm::neon::scale_neon(c, x, y) };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        return unsafe { x86::gfni_avx512::scale_gfni_avx512(c, x, y) };
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return unsafe { x86::gfni_avx2::scale_gfni_avx2(c, x, y) };
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "ssse3",
            not(target_feature = "gfni")
        ))]
        return unsafe { x86::avx2_ssse3::scale_avx2_ssse3(c, x, y) };
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        return unsafe { x86::ssse3::scale_ssse3(c, x, y) };
    }

    scalar::scale(c, x, y)
}

#[cfg(not(feature = "std"))]
#[inline]
fn scale_inplace_static(c: u8, y: &mut [u8]) {
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    // SAFETY: this branch is compiled only when SIMD128 is enabled.
    return unsafe { wasm::simd128::scale_inplace_wasm(c, y) };

    #[cfg(target_arch = "aarch64")]
    return unsafe { arm::neon::scale_inplace_neon(c, y) };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        return unsafe { x86::gfni_avx512::scale_inplace_gfni_avx512(c, y) };
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return unsafe { x86::gfni_avx2::scale_inplace_gfni_avx2(c, y) };
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "ssse3",
            not(target_feature = "gfni")
        ))]
        return unsafe { x86::avx2_ssse3::scale_inplace_avx2_ssse3(c, y) };
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        return unsafe { x86::ssse3::scale_inplace_ssse3(c, y) };
    }

    scalar::scale_inplace(c, y)
}

#[cfg(not(feature = "std"))]
fn active_kernel_name_static() -> &'static str {
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    return "wasm-simd128 (tier8)";
    #[cfg(target_arch = "aarch64")]
    return "neon (tier7)";
    #[cfg(all(target_feature = "gfni", target_feature = "avx512f"))]
    return "gfni+avx512 (tier1)";
    #[cfg(all(
        target_feature = "gfni",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    return "gfni+avx2 (tier2)";
    #[cfg(all(
        target_feature = "avx2",
        target_feature = "ssse3",
        not(target_feature = "gfni")
    ))]
    return "avx2+ssse3 (tier5)";
    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    return "ssse3 (tier6)";
    "scalar (tier9)"
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_round_trip() {
        let c = 0x53u8;
        let x: Vec<u8> = (0u8..128).collect();
        let mut y = vec![0u8; 128];
        axpy(c, &x, &mut y);
        axpy(c, &x, &mut y);
        assert_eq!(y, vec![0u8; 128]);
    }

    #[test]
    fn scale_axpy_consistency() {
        let c = 0xC7u8;
        let x: Vec<u8> = (1u8..=64).collect();
        let mut y_scale = vec![0u8; 64];
        let mut y_axpy = vec![0u8; 64];
        scale(c, &x, &mut y_scale);
        axpy(c, &x, &mut y_axpy);
        assert_eq!(y_scale, y_axpy);
    }

    #[test]
    fn kernel_name_not_empty() {
        let name = active_kernel_name();
        assert!(!name.is_empty());
        println!("Active kernel: {name}");
    }

    #[test]
    #[cfg(feature = "std")]
    fn runtime_dispatch_selects_best() {
        // The selected kernel name should reflect what is actually available.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let name = active_kernel_name();
            if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx512f") {
                assert!(name.contains("tier1"), "Expected tier1, got: {name}");
            } else if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx2") {
                assert!(name.contains("tier2"), "Expected tier2, got: {name}");
            } else if is_x86_feature_detected!("ssse3") {
                assert!(
                    name.contains("tier3")
                        || name.contains("tier4")
                        || name.contains("tier5")
                        || name.contains("tier6"),
                    "Expected SSSE3-tier, got: {name}"
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        assert_eq!(active_kernel_name(), "neon (tier7)");

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(!active_kernel_name().is_empty());
    }
}
