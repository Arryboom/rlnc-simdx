# rlnc-simdx: Architecture Plan

> World-class Rust implementation of Random Linear Network Coding over GF(2⁸)
> `rlnc-simdx` crate — `no_std` core + maximum SIMD acceleration
> **Status:** Implemented (0.1.x line); docs aligned with security remediations (H1–H3, M2–M5)
> **Reference bench:** AMD Ryzen 7 5800X3D + dual DDR4 → tier5 `avx2+ssse3` (~55–62 GB/s AXPY @ 16 KiB–1 MiB); see [README](../README.md#benchmarks-reference-machine)

---

## 1. Project Goals

| Goal | Detail |
|------|--------|
| **Correctness** | GF(2⁸) with AES poly; scalar golden model; property + unit tests |
| **Performance** | Full SIMD hierarchy: GFNI, AVX-512, AVX2, SSSE3, NEON, WASM |
| **Portability** | `no_std + alloc`; embedded → server |
| **Safe public API** | No user `unsafe`; tier kernels **crate-private** |
| **Ergonomics** | Idiomatic Encoder / Decoder / Recoder |
| **Benchmarking** | Criterion + portable `bench_standalone` (GB/s) |

---

## 2. Workspace Layout

```
RLNC_SIMD/
├── README.md
├── CHANGELOG.md
├── Cargo.toml
├── plans/                      # architecture + reviews
├── scripts/
├── rlnc-simdx/
│   └── src/
│       ├── lib.rs              # crate docs, re-exports
│       ├── field/              # tables.rs, gf8.rs
│       ├── kernel/
│       │   ├── mod.rs          # SAFE public axpy/scale/scale_inplace + dispatch
│       │   ├── scalar.rs
│       │   ├── proptest.rs      # property tests (cfg test)
│       │   ├── x86/            # pub(crate) tiers
│       │   ├── arm/            # pub(crate); NEON prod, SVE experimental
│       │   └── wasm/           # pub(crate)
│       ├── aligned.rs          # AlignedBuffer (64 B); new_uninit is pub(crate)
│       ├── matrix.rs           # GfMatrix, padded stride, GE
│       ├── encoder.rs / decoder.rs / recoder.rs
│       └── error.rs
└── rlnc-simdx-bench/
    ├── benches/                # axpy, gf_mul, encode, decode
    └── src/bin/bench_standalone.rs
```

---

## 3. GF(2⁸) Field Arithmetic

### 3.1 Definition

- Polynomial: **0x11B** (AES) — matches hardware `GF2P8MULB`
- Generator: **0x03**
- Tables: `EXP[512]` (doubled, avoids `% 255`), `LOG[256]`, `INV[256]` — `const` / `.rodata`

### 3.2 SIMD multiply strategies

| Strategy | Hardware | Notes |
|----------|----------|-------|
| A | GFNI `GF2P8MULB` | Fastest; Ice Lake+ / Zen4+ |
| B | Nibble split + `pshufb` / `vpshufb` | SSSE3 → AVX-512 |
| C | NEON `vqtbl1q_u8` | AArch64 production path |
| D | SVE | **Experimental — not production dispatch** |
| E | WASM `i8x16.swizzle` | nibble-split |

---

## 4. SIMD Dispatch & Safety Boundary

### 4.1 Tier priority (x86_64)

```
1  gfni + avx512f + avx512bw     → GF2P8MULB zmm
2  gfni + avx2                   → GF2P8MULB ymm
3  gfni + sse4.2                 → GF2P8MULB xmm
4  avx512f + avx512bw + ssse3    → vpshufb zmm
5  avx2 + ssse3                  → vpshufb ymm
6  ssse3                         → pshufb xmm
7  aarch64 neon                  → vqtbl1q
8  wasm simd128                  → swizzle
9  scalar                        → log/exp
```

### 4.2 Public vs private

| Layer | Visibility | Responsibility |
|-------|------------|----------------|
| `kernel::axpy` / `scale` / `scale_inplace` | **`pub fn` (safe)** | Length checks; **overlap assert on axpy/scale**; CPU dispatch |
| Tier functions | **`pub(crate) unsafe fn`** | SIMD only; assume preconditions; no public export |
| Modules `x86` / `arm` / `wasm` | **`pub(crate) mod`** | Not reachable from other crates |

```rust
// Supported external / application code — no unsafe
rlnc_simdx::kernel::axpy(c, &x, &mut y);
rlnc_simdx::kernel::scale(c, &x, &mut y);
rlnc_simdx::kernel::scale_inplace(c, &mut y);
```

### 4.3 Dispatch mechanism

- **`std`:** `OnceLock<KernelSet { axpy, scale, scale_inplace, name }>` +
  `is_x86_feature_detected!` / AArch64 NEON
- **`no_std`:** compile-time `target_feature` chain → scalar fallback
- **Build scripts:** none; the historical host-CPUID script emitted unused cfgs
  and was removed. Runtime detection and target features are authoritative.

### 4.4 Safety contracts (public)

| Function | Length | Overlap |
|----------|--------|---------|
| `axpy` | Must match; release `assert_eq!` | Must **not** overlap (release `assert!`) |
| `scale` | Must match | Must **not** overlap; use `scale_inplace` for same buffer |
| `scale_inplace` | N/A (one buffer) | Full alias is intentional |
| Misuse | **Panic** (not Rust `unsafe` for the caller) | |

---

## 5. Core kernels

```rust
/// Safe public API
pub fn axpy(c: u8, x: &[u8], y: &mut [u8]);           // y ^= c * x
pub fn scale(c: u8, x: &[u8], y: &mut [u8]);          // y = c * x
pub fn scale_inplace(c: u8, y: &mut [u8]);            // y = c * y
pub fn axpy_multi(coeffs: &[u8], sources: &[&[u8]], y: &mut [u8]); // blocked encode
pub fn dot(a: &[u8], b: &[u8]) -> u8;                 // scalar for now
```

Optimizations inside tiers (crate-private): `c == 0` / `c == 1` fast paths,
4-way unroll, aligned vs unaligned loads on x86 where dual paths exist.

---

## 6. Matrix

```rust
// Conceptually: row-major, 64-byte aligned backing, padded row stride
pub struct GfMatrix {
    rows: usize,
    cols: usize,      // logical columns
    stride: usize,    // cols rounded up to 64
    data: AlignedBuffer,
}
```

- `row` / `row_mut`, `row_axpy` (**assert_ne!(dst, src)**), `row_scale` → `scale_inplace`
- Full RREF `gaussian_elimination()` → rank

---

## 7. Public RLNC API

### Encoder

- `encode_random` → `axpy_multi` + bounded all-zero coeff retries (max 100)
- `encode_systematic`
- `CodedPacket { coefficients: AlignedBuffer, payload: AlignedBuffer }`
- FFI: `into_vecs` / `from_slices` / `from_vecs` / `*_slice`

### Decoder

- Online forward elimination on `receive` (free-list working row; no GE `to_vec`)
- Normalize via `scale_inplace` (SIMD)
- `decode`: back-sub + in-place row reorder by pivot column
- Size validation → `RlncError`

### Recoder

- Linear recombine of coded packets; same RNG retry policy as encoder

### SimpleRng

- Xorshift-style LFSR for **coding coefficients only** — **not a CSPRNG**

---

## 8. Alignment

- `AlignedBuffer`: 64-byte alignment for SSE/AVX2/AVX-512
- `new_uninit` is **`pub(crate)`** (internal zero-copy fill only)
- Public construction: `zeroed` / `from_slice`
- Matrix rows start at multiples of 64 → kernels see aligned pointers on hot path

---

## 9. Benchmarking

| Tool | Role |
|------|------|
| Criterion (`rlnc-simdx-bench/benches/*`) | Statistical; use default output for thr. (not `--output-format bencher` alone) |
| `bench_standalone` | Portable EXE; GB/s; scalar vs **safe dispatch** |

Per-tier isolation benches that called raw SIMD were removed/repurposed after H2
(tiers are not public).

---

## 10. Security posture (summary)

See [`security_review.md`](security_review.md) and
[`security_review_validation.md`](security_review_validation.md).

| Item | Status |
|------|--------|
| Public safe wrappers | Implemented |
| Crate-private SIMD | Implemented (H2) |
| `new_uninit` not public | Implemented (H1) |
| Overlap checks on axpy/scale | Implemented (H3) |
| Non-crypto / non-CT docs | Implemented (M3/M5) |
| Size caps for huge k/n | Planned (M1) |

---

## 11. Testing

- Unit + doctests: field, kernels, matrix, encode/decode/recode
- Property tests: dispatch vs scalar across lengths / alignment / `c`
- CI: x86 feature matrix, aarch64, wasm check, no_std, fmt/clippy, MSRV 1.89

---

## 12. Non-goals

- Cryptographic secrecy or integrity of packets (use external AEAD/TLS)
- Constant-time arithmetic
- Production SVE until rewritten and CI-proven
- Public raw intrinsic API

---

*Architecture document maintained as the design source of truth for the workspace.*
