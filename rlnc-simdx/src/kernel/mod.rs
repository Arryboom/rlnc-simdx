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
//! - [`axpy_multi`] — adaptive blocked/fused multi-source AXPY for encoding
//! - [`dot`] — GF(2⁸) dot product (GFNI accelerated when available)
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
    clippy::if_not_else,
    clippy::ptr_as_ptr,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::verbose_bit_mask,
    clippy::wildcard_imports
)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm;

pub(crate) mod wasm;

/// Safe direct-tier handles for workspace benchmarks.
///
/// This module exists only with `std + bench-internals`. It is not covered by
/// the crate's stable API guarantees. Handles are returned only after runtime
/// CPU-feature detection, while each call retains the public kernels' length
/// and overlap checks.
#[cfg(all(feature = "bench-internals", feature = "std"))]
pub mod bench {
    use super::ranges_overlap;

    type DirectAxpyFn = unsafe fn(c: u8, x: &[u8], y: &mut [u8]);

    /// A CPU-validated direct AXPY tier used to isolate dispatch overhead and
    /// compare implementations in benchmarks.
    #[derive(Clone, Copy)]
    pub struct DirectAxpyTier {
        name: &'static str,
        function: DirectAxpyFn,
    }

    impl DirectAxpyTier {
        /// Human-readable ISA tier name.
        #[must_use]
        pub fn name(self) -> &'static str {
            self.name
        }

        /// Run this tier through the same safety contract as [`super::axpy`].
        ///
        /// # Panics
        /// Panics when lengths differ or the source and destination overlap.
        pub fn axpy(self, c: u8, x: &[u8], y: &mut [u8]) {
            assert_eq!(
                x.len(),
                y.len(),
                "rlnc_simdx::kernel::bench::DirectAxpyTier::axpy: length mismatch"
            );
            assert!(
                !ranges_overlap(x.as_ptr(), x.len(), y.as_ptr(), y.len()),
                "rlnc_simdx::kernel::bench::DirectAxpyTier::axpy: overlapping buffers are not allowed"
            );
            // SAFETY: the constructor is private, so handles are created only
            // after checking their required CPU features. Buffer invariants
            // were validated above.
            unsafe { (self.function)(c, x, y) };
        }
    }

    /// Return every direct AXPY tier supported by the current CPU.
    #[must_use]
    pub fn available_axpy_tiers() -> Vec<DirectAxpyTier> {
        let mut tiers = Vec::new();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use super::x86;

            if std::is_x86_feature_detected!("gfni")
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
            {
                tiers.push(DirectAxpyTier {
                    name: "gfni+avx512",
                    function: x86::gfni_avx512::axpy_gfni_avx512 as DirectAxpyFn,
                });
            }
            if std::is_x86_feature_detected!("gfni") && std::is_x86_feature_detected!("avx2") {
                tiers.push(DirectAxpyTier {
                    name: "gfni+avx2",
                    function: x86::gfni_avx2::axpy_gfni_avx2 as DirectAxpyFn,
                });
            }
            if std::is_x86_feature_detected!("gfni") && std::is_x86_feature_detected!("sse4.2") {
                tiers.push(DirectAxpyTier {
                    name: "gfni+sse4.2",
                    function: x86::gfni_sse::axpy_gfni_sse as DirectAxpyFn,
                });
            }
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("ssse3")
            {
                tiers.push(DirectAxpyTier {
                    name: "avx512+ssse3",
                    function: x86::avx512_ssse3::axpy_avx512_ssse3 as DirectAxpyFn,
                });
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("ssse3") {
                tiers.push(DirectAxpyTier {
                    name: "avx2+ssse3",
                    function: x86::avx2_ssse3::axpy_avx2_ssse3 as DirectAxpyFn,
                });
            }
            if std::is_x86_feature_detected!("ssse3") {
                tiers.push(DirectAxpyTier {
                    name: "ssse3",
                    function: x86::ssse3::axpy_ssse3 as DirectAxpyFn,
                });
            }
        }

        #[cfg(target_arch = "aarch64")]
        tiers.push(DirectAxpyTier {
            name: "neon",
            function: super::arm::neon::axpy_neon as DirectAxpyFn,
        });

        #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
        tiers.push(DirectAxpyTier {
            name: "wasm-simd128",
            function: super::wasm::simd128::axpy_wasm as DirectAxpyFn,
        });

        tiers
    }
}

// ---------------------------------------------------------------------------
// Kernel function-pointer types
// ---------------------------------------------------------------------------

/// Signature for `axpy`: `y[i] ^= c * x[i]` over GF(2⁸).
#[cfg(feature = "std")]
pub(crate) type AxpyFn = unsafe fn(c: u8, x: &[u8], y: &mut [u8]);
/// Signature for `scale`: `y[i] = c * x[i]` over GF(2⁸).
#[cfg(feature = "std")]
pub(crate) type ScaleFn = unsafe fn(c: u8, x: &[u8], y: &mut [u8]);
/// Signature for in-place scale: `y[i] = c * y[i]`.
#[cfg(feature = "std")]
pub(crate) type ScaleInplaceFn = unsafe fn(c: u8, y: &mut [u8]);
/// Signature for a validated multi-source AXPY implementation.
#[cfg(feature = "std")]
pub(crate) type AxpyMultiFn = unsafe fn(coeffs: &[u8], sources: &[&[u8]], y: &mut [u8]);
/// Signature for a vectorized GF dot product.
#[cfg(feature = "std")]
pub(crate) type DotFn = unsafe fn(a: &[u8], b: &[u8]) -> u8;

/// A resolved set of kernel function pointers for the detected CPU tier.
#[cfg(feature = "std")]
pub(crate) struct KernelSet {
    /// Best available axpy kernel.
    pub(crate) axpy: AxpyFn,
    /// Best available scale kernel.
    pub(crate) scale: ScaleFn,
    /// Best available in-place scale kernel.
    pub(crate) scale_inplace: ScaleInplaceFn,
    /// Fused multi-source kernel when the selected ISA provides one.
    pub(crate) axpy_multi: Option<AxpyMultiFn>,
    /// Vectorized dot product when the selected ISA provides one.
    pub(crate) dot: Option<DotFn>,
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
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    use super::wasm;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::x86;
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    )))]
    use super::{scalar_axpy_wrapper, scalar_scale_inplace_wrapper, scalar_scale_wrapper};
    use super::{AxpyFn, KernelSet, ScaleFn, ScaleInplaceFn};
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::{AxpyMultiFn, DotFn};
    use std::sync::OnceLock;

    static KERNEL: OnceLock<KernelSet> = OnceLock::new();

    /// Return (or initialise) the globally cached best kernel set.
    pub(crate) fn get() -> &'static KernelSet {
        KERNEL.get_or_init(detect)
    }

    /// Probe the CPU at runtime and return the highest-tier kernel set.
    #[allow(clippy::too_many_lines)]
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
                    axpy_multi: Some(x86::gfni_avx512::axpy_multi_gfni_avx512 as AxpyMultiFn),
                    dot: Some(x86::gfni_avx512::dot_gfni_avx512 as DotFn),
                    name: "gfni+avx512 (tier1)",
                };
            }

            // Tier 2 — GFNI + AVX2
            if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx2") {
                return KernelSet {
                    axpy: x86::gfni_avx2::axpy_gfni_avx2 as AxpyFn,
                    scale: x86::gfni_avx2::scale_gfni_avx2 as ScaleFn,
                    scale_inplace: x86::gfni_avx2::scale_inplace_gfni_avx2 as ScaleInplaceFn,
                    axpy_multi: Some(x86::gfni_avx2::axpy_multi_gfni_avx2 as AxpyMultiFn),
                    dot: Some(x86::gfni_avx2::dot_gfni_avx2 as DotFn),
                    name: "gfni+avx2 (tier2)",
                };
            }

            // Tier 3 — GFNI + SSE4.2
            if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("sse4.2") {
                return KernelSet {
                    axpy: x86::gfni_sse::axpy_gfni_sse as AxpyFn,
                    scale: x86::gfni_sse::scale_gfni_sse as ScaleFn,
                    scale_inplace: x86::gfni_sse::scale_inplace_gfni_sse as ScaleInplaceFn,
                    axpy_multi: Some(x86::gfni_sse::axpy_multi_gfni_sse as AxpyMultiFn),
                    dot: Some(x86::gfni_sse::dot_gfni_sse as DotFn),
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
                    axpy_multi: None,
                    dot: None,
                    name: "avx512+ssse3 (tier4)",
                };
            }

            // Tier 5 — AVX2 + SSSE3
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("ssse3") {
                return KernelSet {
                    axpy: x86::avx2_ssse3::axpy_avx2_ssse3 as AxpyFn,
                    scale: x86::avx2_ssse3::scale_avx2_ssse3 as ScaleFn,
                    scale_inplace: x86::avx2_ssse3::scale_inplace_avx2_ssse3 as ScaleInplaceFn,
                    axpy_multi: None,
                    dot: None,
                    name: "avx2+ssse3 (tier5)",
                };
            }

            // Tier 6 — SSSE3
            if is_x86_feature_detected!("ssse3") {
                return KernelSet {
                    axpy: x86::ssse3::axpy_ssse3 as AxpyFn,
                    scale: x86::ssse3::scale_ssse3 as ScaleFn,
                    scale_inplace: x86::ssse3::scale_inplace_ssse3 as ScaleInplaceFn,
                    axpy_multi: None,
                    dot: None,
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
                axpy_multi: None,
                dot: None,
                name: "neon (tier7)",
            };
        }

        // ── WASM SIMD128 ─────────────────────────────────────────────────────
        // Runtime feature detection is unavailable on wasm32. SIMD128 is a
        // compile-time feature even when the crate is built with `std`.
        #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
        {
            return KernelSet {
                axpy: wasm::simd128::axpy_wasm as AxpyFn,
                scale: wasm::simd128::scale_wasm as ScaleFn,
                scale_inplace: wasm::simd128::scale_inplace_wasm as ScaleInplaceFn,
                axpy_multi: None,
                dot: None,
                name: "wasm-simd128 (tier8)",
            };
        }

        // ── Scalar fallback ──────────────────────────────────────────────────
        // AArch64 returned the mandatory NEON kernel above, so compiling this
        // fallback there would only create unreachable-code/dead-import noise.
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_family = "wasm", target_feature = "simd128")
        )))]
        KernelSet {
            axpy: scalar_axpy_wrapper as AxpyFn,
            scale: scalar_scale_wrapper as ScaleFn,
            scale_inplace: scalar_scale_inplace_wrapper as ScaleInplaceFn,
            axpy_multi: None,
            dot: None,
            name: "scalar (tier9)",
        }
    }
}

// ---------------------------------------------------------------------------
// Thin wrappers so scalar fns have the right unsafe fn signature
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "std",
    not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    ))
))]
unsafe fn scalar_axpy_wrapper(c: u8, x: &[u8], y: &mut [u8]) {
    scalar::axpy(c, x, y);
}

#[cfg(all(
    feature = "std",
    not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    ))
))]
unsafe fn scalar_scale_wrapper(c: u8, x: &[u8], y: &mut [u8]) {
    scalar::scale(c, x, y);
}

#[cfg(all(
    feature = "std",
    not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    ))
))]
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
    // SAFETY: length and non-overlap were checked above.
    unsafe { axpy_unchecked(c, x, y) }
}

/// Dispatch AXPY after the caller has established equal lengths and disjoint
/// source/destination storage. This is crate-private so public callers cannot
/// bypass the safe wrapper's validation.
#[inline]
pub(crate) unsafe fn axpy_unchecked(c: u8, x: &[u8], y: &mut [u8]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "std")]
    (runtime::get().axpy)(c, x, y);

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
    let axpy_kernel = {
        let kernels = runtime::get();
        // Full fusion trades repeated destination traffic for a dynamic
        // multi-stream inner loop. Measurements on tier1 show the latter wins
        // only once the symbol has left the small-cache regime.
        if y.len() >= 32 * 1024 && coeffs.len() >= 8 {
            if let Some(fused) = kernels.axpy_multi {
                // SAFETY: all lengths and destination overlaps were validated
                // above, and runtime dispatch verified the CPU feature set.
                unsafe { fused(coeffs, sources, y) };
                return;
            }
        }
        kernels.axpy
    };

    #[cfg(not(feature = "std"))]
    if y.len() >= 32 * 1024 && coeffs.len() >= 8 && axpy_multi_static(coeffs, sources, y) {
        return;
    }

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

/// `sum(a[i] * b[i])` in GF(2⁸).
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn dot(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "rlnc_simdx::kernel::dot: length mismatch");
    #[cfg(feature = "std")]
    if let Some(dot_kernel) = runtime::get().dot {
        // SAFETY: runtime dispatch verified the feature set and lengths match.
        return unsafe { dot_kernel(a, b) };
    }
    #[cfg(not(feature = "std"))]
    if let Some(result) = dot_static(a, b) {
        return result;
    }
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
            target_feature = "gfni",
            target_feature = "sse4.2",
            not(target_feature = "avx2")
        ))]
        return unsafe { x86::gfni_sse::axpy_gfni_sse(c, x, y) };
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "ssse3",
            not(target_feature = "gfni")
        ))]
        return unsafe { x86::avx512_ssse3::axpy_avx512_ssse3(c, x, y) };
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "ssse3",
            not(target_feature = "gfni"),
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return unsafe { x86::avx2_ssse3::axpy_avx2_ssse3(c, x, y) };
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        return unsafe { x86::ssse3::axpy_ssse3(c, x, y) };
    }

    // Universal scalar fallback
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    )))]
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
            target_feature = "gfni",
            target_feature = "sse4.2",
            not(target_feature = "avx2")
        ))]
        return unsafe { x86::gfni_sse::scale_gfni_sse(c, x, y) };
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "ssse3",
            not(target_feature = "gfni")
        ))]
        return unsafe { x86::avx512_ssse3::scale_avx512_ssse3(c, x, y) };
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "ssse3",
            not(target_feature = "gfni"),
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return unsafe { x86::avx2_ssse3::scale_avx2_ssse3(c, x, y) };
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        return unsafe { x86::ssse3::scale_ssse3(c, x, y) };
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    )))]
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
            target_feature = "gfni",
            target_feature = "sse4.2",
            not(target_feature = "avx2")
        ))]
        return unsafe { x86::gfni_sse::scale_inplace_gfni_sse(c, y) };
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "ssse3",
            not(target_feature = "gfni")
        ))]
        return unsafe { x86::avx512_ssse3::scale_inplace_avx512_ssse3(c, y) };
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "ssse3",
            not(target_feature = "gfni"),
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return unsafe { x86::avx2_ssse3::scale_inplace_avx2_ssse3(c, y) };
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        return unsafe { x86::ssse3::scale_inplace_ssse3(c, y) };
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    )))]
    scalar::scale_inplace(c, y)
}

#[cfg(not(feature = "std"))]
fn active_kernel_name_static() -> &'static str {
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    return "wasm-simd128 (tier8)";
    #[cfg(target_arch = "aarch64")]
    return "neon (tier7)";
    #[cfg(all(
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    return "gfni+avx512 (tier1)";
    #[cfg(all(
        target_feature = "gfni",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    return "gfni+avx2 (tier2)";
    #[cfg(all(
        target_feature = "gfni",
        target_feature = "sse4.2",
        not(target_feature = "avx2")
    ))]
    return "gfni+sse4.2 (tier3)";
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "ssse3",
        not(target_feature = "gfni")
    ))]
    return "avx512+ssse3 (tier4)";
    #[cfg(all(
        target_feature = "avx2",
        target_feature = "ssse3",
        not(target_feature = "gfni"),
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    return "avx2+ssse3 (tier5)";
    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    return "ssse3 (tier6)";
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    )))]
    return "scalar (tier9)";
}

#[cfg(not(feature = "std"))]
#[inline]
fn axpy_multi_static(coeffs: &[u8], sources: &[&[u8]], y: &mut [u8]) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        unsafe {
            x86::gfni_avx512::axpy_multi_gfni_avx512(coeffs, sources, y);
            return true;
        }
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        unsafe {
            x86::gfni_avx2::axpy_multi_gfni_avx2(coeffs, sources, y);
            return true;
        }
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "sse4.2",
            not(target_feature = "avx2")
        ))]
        unsafe {
            x86::gfni_sse::axpy_multi_gfni_sse(coeffs, sources, y);
            return true;
        }
    }
    let _ = (coeffs, sources, y);
    false
}

#[cfg(not(feature = "std"))]
#[inline]
fn dot_static(a: &[u8], b: &[u8]) -> Option<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        return Some(unsafe { x86::gfni_avx512::dot_gfni_avx512(a, b) });
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        return Some(unsafe { x86::gfni_avx2::dot_gfni_avx2(a, b) });
        #[cfg(all(
            target_feature = "gfni",
            target_feature = "sse4.2",
            not(target_feature = "avx2")
        ))]
        return Some(unsafe { x86::gfni_sse::dot_gfni_sse(a, b) });
    }
    let _ = (a, b);
    None
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
    fn dispatched_dot_matches_scalar_across_vector_tails() {
        for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1025] {
            let a: Vec<u8> = (0..len)
                .map(|index| (index as u8).wrapping_mul(17).wrapping_add(1))
                .collect();
            let b: Vec<u8> = (0..len)
                .map(|index| (index as u8).wrapping_mul(29).wrapping_add(3))
                .collect();
            assert_eq!(dot(&a, &b), scalar::dot(&a, &b), "len={len}");
        }
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
