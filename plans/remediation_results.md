# Remediation Results — `rlnc-simdx`

**Date:** 2026-07-24  
**Scope:** release engineering, feature topology, public API boundary, `axpy_multi` safety/performance  
**Result:** **All requested fixes implemented; release gates pass.**

---

## Compatibility changes

- MSRV corrected from Rust 1.79 to **Rust 1.89**, the first tested stable toolchain that compiles the complete GFNI/AVX-512 production set.
- Removed the no-op `runtime-dispatch` feature; `std` controls cached runtime dispatch.
- `kernel::scalar` is private by default and exposed only by the explicitly unstable `bench-internals` feature used by `rlnc-simdx-bench`.
- Dispatch aliases, `KernelSet`, raw ISA modules/functions, and finite-field tables are crate-private.
- Removed unused `RlncError::NotEnoughPackets`; incomplete decode remains `Ok(None)`.
- The zero-feature build now exposes the allocator-free field, error, kernel, and diagnostics core; heap-backed APIs require `alloc`.

---

## `axpy_multi` safety and performance change

`kernel::axpy_multi` now:

1. validates coefficient/source count;
2. validates every source length;
3. validates every full source range is disjoint from the destination, including zero-coefficient sources;
4. completes all validation before mutating the destination;
5. resolves the runtime AXPY function pointer once under `std`;
6. preserves the existing 4096-byte cache-blocked loop order;
7. directly invokes the selected crate-private kernel for each non-zero source/block;
8. uses compile-time dispatch directly without `std`.

The public `axpy`, `scale`, `scale_inplace`, and all SIMD loop algorithms were not changed.

---

## Test coverage

Current result:

```text
91 unit tests passed
3 doctests passed
0 failures
```

New `axpy_multi` coverage includes:

- full and partial overlap rejection;
- validation-before-mutation on late length/overlap failure;
- overlap rejection even for coefficient zero;
- all-zero coefficient preservation of a non-zero destination;
- empty source/coefficient sets with empty and non-empty destinations;
- 4095, 4096, 4097, 8193, 10007, 12345-byte boundaries and tails;
- deterministic randomized scalar equivalence;
- equivalence to repeated supported public-dispatch calls.

---

## Release gates

| Gate | Result |
|------|--------|
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | Pass, zero warnings |
| `cargo test --workspace` | Pass: 91 + 3 |
| `cargo test -p rlnc-simdx --all-features` | Pass: 91 + 3 |
| `cargo check -p rlnc-simdx --no-default-features` | Pass |
| `cargo check -p rlnc-simdx --no-default-features --features alloc` | Pass |
| `cargo check -p rlnc-simdx-bench --bins` | Pass |
| Rust 1.89 all-feature check | Pass |
| Full `cargo package` verification | Pass: 31 files, 40.0 KiB compressed |
| Scalar private/default external probe | Expected E0603 (private) |
| Scalar `bench-internals` external probe | Pass |
| `git diff --check` | Pass |

Reduced-feature checks emit expected dead-code warnings for ISA tiers unavailable in those configurations; strict all-target/all-feature Clippy is clean.

---

## Benchmark protocol

- Portable release binary: `bench_standalone --quick --csv`
- Budget: 250 ms per cell
- Buffers: 64-byte aligned
- Units: SI GB/s
- Pre-remediation sample: [`pre_remediation_benchmark.csv`](pre_remediation_benchmark.csv)
- Post-remediation samples:
  - [`post_remediation_benchmark.csv`](post_remediation_benchmark.csv)
  - [`post_remediation_benchmark_run2.csv`](post_remediation_benchmark_run2.csv)
  - [`post_remediation_benchmark_run3.csv`](post_remediation_benchmark_run3.csv)
- Human-readable post run: [`post_remediation_benchmark_human.txt`](post_remediation_benchmark_human.txt)

The post run reports `gfni+avx512 (tier1)`. The pre CSV format does not embed the active tier; it was captured immediately before remediation on the same workspace/host, and runtime tier selection code was not changed. Therefore the comparison is useful as a local regression signal but is not presented as a portable cross-machine benchmark result.

---

## Performance comparison

The table compares the single pre-remediation quick sample with the median of three post-remediation quick samples.

### Raw safe-dispatch kernels

Representative cells:

| Operation | Size | Pre | Post median | Delta |
|-----------|------|-----|-------------|-------|
| AXPY | 1 KiB | 76.11 GB/s | 77.17 GB/s | +1.4% |
| AXPY | 16 KiB | 116.41 GB/s | 117.68 GB/s | +1.1% |
| AXPY | 64 KiB | 49.07 GB/s | 49.16 GB/s | +0.2% |
| AXPY | 1 MiB | 22.37 GB/s | 22.45 GB/s | +0.3% |
| SCALE | 16 KiB | 146.63 GB/s | 148.27 GB/s | +1.1% |
| SCALE | 64 KiB | 45.22 GB/s | 45.13 GB/s | -0.2% |
| SCALE | 1 MiB | 21.52 GB/s | 22.26 GB/s | +3.4% |

The quick-run raw-kernel results show **no broad performance regression**. The 256 KiB SCALE median was lower than the pre sample, but the three post samples ranged from 45.63 to 49.32 GB/s and include a value above the pre 48.91 GB/s result, indicating cache/clock variance rather than a code-path change. No SCALE implementation was modified.

### Random encode, k=16

| Symbol size | Pre | Post median | Delta |
|-------------|-----|-------------|-------|
| 1 KiB | 2.785 GB/s | 2.793 GB/s | +0.3% |
| 4 KiB | 4.235 GB/s | 4.264 GB/s | +0.7% |
| 16 KiB | 4.466 GB/s | 4.398 GB/s | -1.5% |
| 64 KiB | 4.493 GB/s | 4.793 GB/s | **+6.7%** |

Small and medium encode cells are effectively flat within quick-run variance. The intended reduction in per-block validation/dispatch overhead is visible at 64 KiB, where median throughput improved by about **6.7%**.

These measurements are regression evidence, not universal performance guarantees. Published benchmark numbers should use longer budgets, repeated medians, a recorded CPU/OS/compiler profile, and a fixed active tier.

---

## Final assessment

```text
CORE CORRECTNESS:       PASS
SAFE API CONTRACT:      PASS
ZERO-FEATURE CORE:      PASS
NO_STD + ALLOC:         PASS
CI / FMT / CLIPPY:      PASS
MSRV 1.89:              PASS
PACKAGE VERIFICATION:   PASS
RAW KERNEL REGRESSION:  NOT OBSERVED
AXPY_MULTI / ENCODE:    +6.7% at 64 KiB in local quick median
```
