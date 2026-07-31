# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [1.3.1] — 2026-07-31

### Changed

- Decoder storage now keeps coefficient rows separate from payload rows. Incoming
  coefficients are reduced before payload arithmetic, dependent packets avoid
  payload-sized work, and innovative packet buffers move directly into pivot
  storage.
- Added adaptive GFNI fused multi-source AXPY and GF(2^8) dot-product kernels
  for x86 tiers, with vector tails for AVX-512, AVX2, SSE, and SSSE3 paths.
- Fixed runtime dispatch for `std + wasm simd128` and completed no-std x86
  GFNI/AVX-512 tier selection. Scalar fixed-coefficient operations use
  pre-generated nibble tables.
- Matrix elimination skips aligned completed prefixes, and recoding uses fused
  multi-source operations for coefficient and payload rows.
- Updated the multithreaded benchmark's paired scalar/SIMD workload to use
  adaptive multi-source encoding and separated coefficient/payload decoder rows,
  matching the optimized production data flow.

### Fixed

- Safe aligned-buffer and matrix size arithmetic now rejects overflow before
  allocation; `AlignedBuffer::from_slice` no longer creates a slice over
  uninitialized storage.
- `Gf8::pow(0, 0)` returns one, and division by zero panics in release builds
  as well as debug builds.
- `SimpleRng::new(u64::MAX)` no longer enters the permanent zero state.
- Added aligned/unaligned SIMD-tail, CPU-validated direct-tier, fused primitive,
  Recoder, and staged public Decoder benchmark coverage.

## [1.2.0] — 2026-07-24

### Added

- Added the `rlnc-simdx-mt-bench` workspace package. It uses Rayon to run
  independent RLNC generations in parallel and autotunes scalar and SIMD worker
  counts separately for encode and decode across `k = 8, 16, 32` and symbol
  sizes from 64 B through 64 KiB.
- Added deterministic full-rank fixtures and benchmark correctness tests so the
  scalar and SIMD backends share identical RLNC, memory, and threading work.
- Added CI release-mode smoke coverage and included `rlnc-simdx-mt-bench`
  alongside `bench_standalone` in every native GitHub Release archive.
- Added a portable ASCII benchmark report with host/kernel metadata, streaming
  workload rows, per-operation peak-throughput and speedup summaries, and total
  elapsed time.

### Changed

- Standardized the multithreaded benchmark directory, Cargo package, executable,
  Rust import, CI job, documentation, and release archive entry on the canonical
  `rlnc-simdx-mt-bench` name.

### Fixed

- Fixed AArch64 `std` runtime dispatch compilation by importing the crate-private
  ARM kernel module inside the runtime dispatcher. AArch64 now selects mandatory
  NEON cleanly in both normal and `+sve` builds; scalar fallback code is cfg-gated
  away on AArch64, and the target-specific dispatch test asserts `neon (tier7)`.

## [1.1.0] — 2026-07-24

### Security

- **H1:** [`AlignedBuffer::new_uninit`](rlnc-simdx/src/aligned.rs) is **`pub(crate)` only** —
  prevents external safe code from reading uninitialized memory via `as_slice` / `Deref`.
- **H2:** All SIMD tier modules (`kernel::x86`, `kernel::arm`, `kernel::wasm`) and
  tier functions (`axpy_*` / `scale_*` / `scale_inplace_*`) are **`pub(crate)`**.
  External code uses only safe [`kernel::axpy`](rlnc-simdx/src/kernel/mod.rs) /
  [`scale`](rlnc-simdx/src/kernel/mod.rs) / [`scale_inplace`](rlnc-simdx/src/kernel/mod.rs).
- **H3:** Public `axpy` / `scale` assert **non-overlapping** buffers in **release**
  (pointer-range check). Overlap panics instead of silent UB.
- **M2:** [`GfMatrix::row_axpy`](rlnc-simdx/src/matrix.rs) uses `assert_ne!(dst, src)` in release.
- **M3 / M5:** Crate-level and README warnings: not cryptography, not constant-time,
  `SimpleRng` is not a CSPRNG.
- **M4:** Encoder / Recoder all-zero coefficient loops capped at **100** retries,
  then force a non-zero coefficient.

### Changed

- Project / crate renamed to **`rlnc-simdx`** (Rust import `rlnc_simdx`); author **arryboom**.
- License set to **Apache-2.0 only** (was dual MIT OR Apache-2.0).
- Corrected the declared MSRV from 1.79 to 1.89, the first stable Rust release
  providing the x86 GFNI/AVX-512 intrinsics used by the production dispatch set.
- The no-feature build is now a true zero-allocation core: `field`, `error`,
  `kernel`, and `active_kernel()` remain available, while aligned storage,
  matrix, encoder, decoder, and recoder APIs require `alloc`.
- `GfMatrix` is re-exported from the crate root whenever `alloc` is enabled.
- The no-op `runtime-dispatch` feature was removed; `std` continues to enable
  cached runtime CPU dispatch.
- `kernel::scalar` is crate-private by default. The explicitly unstable
  `bench-internals` feature exposes it only for benchmark tooling.
- Dispatch implementation aliases, `KernelSet`, raw ISA modules/functions, and
  finite-field tables are now crate-private.
- `kernel::axpy_multi` now validates every source length and full-range overlap
  before mutation, resolves runtime dispatch once, and calls the selected
  crate-private kernel directly for each non-zero source/block.
- Removed the unused `RlncError::NotEnoughPackets` variant; incomplete decode
  continues to return `Ok(None)`.
- Removed the unused host-CPUID build script and empty build-dependency section.
- The package now includes package-local README and Apache-2.0 license files.

### Added

- Automated GitHub releases now publish the source crate and native Linux, Windows,
  and macOS benchmark archives with a sorted `SHA256SUMS` manifest and per-asset hashes.
- [`kernel::scale_inplace`](rlnc-simdx/src/kernel/mod.rs) — SIMD in-place scale via dispatch.
- [`kernel::axpy_multi`](rlnc-simdx/src/kernel/mod.rs) — cache-blocked multi-source AXPY for encode.
- [`AlignedBuffer`](rlnc-simdx/src/aligned.rs) — 64-byte aligned storage; used by packets & matrix.
- [`CodedPacket`](rlnc-simdx/src/encoder.rs) FFI helpers: `into_vecs`, `from_slices`, `from_vecs`,
  `coefficients_slice`, `payload_slice`.
- Decoder free-list for working rows; in-place pivot-row reorder after back-sub.
- Property tests ([`kernel/proptest.rs`](rlnc-simdx/src/kernel/proptest.rs)): random `c` / lengths /
  unaligned offsets / aligned buffers / `scale_inplace` / `axpy_multi`.
- `axpy_multi` tests cover full/partial overlap, validation-before-mutation,
  zero coefficients, empty inputs, 4096-byte boundaries/tails, randomized
  scalar equivalence, and repeated public-dispatch equivalence.
- P0+P1 safety/API tests: overlap panics, encoder/decoder errors, recoder hard asserts
  (**91** unit tests + 3 doctests).
- Portable [`bench_standalone`](rlnc-simdx-bench/src/bin/bench_standalone.rs) binary (GB/s tables).
- [`README.md`](README.md) — usage, safety model, security warning, **reference benches**
  (AMD Ryzen 7 5800X3D, dual-channel DDR4, tier5 `avx2+ssse3`).
- Review docs under [`plans/`](plans/): architecture, expert, change, security, validation,
  perf regression, test coverage.

### Changed

- `CodedPacket` payloads / coefficients backed by `AlignedBuffer` (not `Vec`).
- `GfMatrix` uses `AlignedBuffer` with **64-byte padded row stride**.
- Encoder random path uses `axpy_multi`.
- SIMD kernels: `c == 1` XOR fast path; optional aligned load paths on x86;
  NEON single load path (ARM guidance).
- SVE marked experimental / not wired into production dispatch.
- Criterion `axpy` / standalone benches measure **scalar vs safe dispatch** only
  (tiers are crate-private).
- Public docs emphasize **safe default API** (no user `unsafe`).

### Fixed

- Decoder GE: removed `to_vec` / extra buffers on pivot AXPY and normalize path.
- Matrix `row_scale` uses SIMD `scale_inplace`.
- Various `missing_docs` / error field documentation.

### Tests

- **60** unit tests + **3** doctests (`cargo test -p rlnc-simdx`).

### Planned

- Optional size caps (`MAX_GEN` / `MAX_SYMBOL`) for DoS hardening (security M1)
- Optional CSPRNG inject for coefficients
- Production SVE kernel (when correct + CI on SVE hardware)
- SIMD `dot` product
- `serde` for `CodedPacket`
- Zero-alloc matrix view API

---

## [0.1.0] — 2026-07-23

Initial release. Full RLNC over GF(2⁸) with multi-tier SIMD on x86_64, AArch64, and WASM.

### Added

#### Core library (`rlnc-simdx`)

**GF(2⁸) field arithmetic** ([`rlnc-simdx/src/field/`](rlnc-simdx/src/field/))

- Compile-time EXP (512 doubled), LOG, INV tables; AES poly `0x11B`
- `Gf8` newtype: Add / Sub / Mul / Div / Neg / Inv / Pow

**SIMD kernel hierarchy** ([`rlnc-simdx/src/kernel/`](rlnc-simdx/src/kernel/)) — runtime
`OnceLock` dispatch on `std`; compile-time fallback on `no_std`

| Tier | Instruction | Width | CPU |
|------|-------------|-------|-----|
| 1 | `GF2P8MULB zmm` | 512-bit | Ice Lake+ / Zen4+ AVX-512 |
| 2 | `GF2P8MULB ymm` | 256-bit | Ice Lake+ / Zen4+ |
| 3 | `GF2P8MULB xmm` | 128-bit | Ice Lake+ / Zen4+ |
| 4 | `vpshufb zmm` | 512-bit | AVX-512BW |
| 5 | `vpshufb ymm` | 256-bit | AVX2 |
| 6 | `pshufb xmm` | 128-bit | SSSE3 |
| 7 | `vqtbl1q_u8` | 128-bit | AArch64 NEON |
| 8 | `i8x16_swizzle` | 128-bit | WASM SIMD128 |
| 9 | log/exp table | 1 byte | Scalar |

- Public primitives: `axpy`, `scale`, `dot`, `active_kernel_name()`
- `make_nibble_tables(c)` for nibble-split tiers

**Matrix / Encoder / Decoder / Recoder / Error** — see architecture plan.

**`no_std` (historical 0.1.0 claim):** zero mandatory deps; `alloc` / `std`
features. The originally stated MSRV 1.79 was corrected to 1.89 in 1.1.0.

#### Build / bench / CI

- Historical: the 0.1.0 package used a host-CPUID `build.rs`; it was removed in
  the subsequent remediation because no emitted `host_*` cfg was consumed.
- `rlnc-simdx-bench` Criterion suites + CI matrix (x86, aarch64, wasm, no_std, lint, MSRV)

### Fixed (0.1.0 era)

- Historical `build.rs` safe `__cpuid_count` usage
- Early decoder/recoder borrow and import cleanups

[Unreleased]: https://github.com/arryboom/rlnc-simdx/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/arryboom/rlnc-simdx/compare/v1.2.0...v1.3.1
[1.2.0]: https://github.com/arryboom/rlnc-simdx/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/arryboom/rlnc-simdx/compare/v0.1.0...v1.1.0
[0.1.0]: https://github.com/arryboom/rlnc-simdx/releases/tag/v0.1.0
