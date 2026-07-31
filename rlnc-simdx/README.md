# rlnc-simdx

**SIMD-accelerated Random Linear Network Coding over GF(2⁸)**

Author: **[arryboom](https://github.com/arryboom)** · MSRV 1.89 · License [Apache-2.0](LICENSE) · v1.2.1

Crate package: **`rlnc-simdx`** · Rust import: **`rlnc_simdx`**

`no_std`-friendly · zero mandatory dependencies · safe public API · automatic CPU dispatch
(GFNI / AVX-512 / AVX2 / SSSE3 / NEON / WASM SIMD128)

**~55–62 GB/s** AXPY on AMD Ryzen 7 5800X3D (dual DDR4, AVX2 path) — about **25–29×** faster than scalar.

---

## Install

```toml
# Cargo.toml
[dependencies]
rlnc-simdx = "1.2"

# From this repository's workspace root instead:
# rlnc-simdx = { path = "rlnc-simdx" }
```

The default features are `std` and `alloc`. Use `default-features = false` for the
zero-allocation core, or enable only `alloc` for heap-backed APIs without `std`.

---

## Quick start

### Encode → decode a generation

```rust
use rlnc_simdx::{Decoder, Encoder, RlncError};

fn main() -> Result<(), RlncError> {
    let k = 4;   // generation size (number of source symbols)
    let n = 128; // bytes per symbol

    let source: Vec<Vec<u8>> = (0..k).map(|i| vec![i as u8; n]).collect();
    let source_refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();

    let encoder = Encoder::new(k, n)?;
    let mut decoder = Decoder::new(k, n)?;

    // Systematic packets make this compact example deterministic. Applications
    // can use Encoder::encode_random with SimpleRng or their own coefficients.
    for index in 0..k {
        let packet = encoder.encode_systematic(&source_refs, index)?;
        decoder.receive(packet)?;
    }

    let decoded = decoder.decode()?.expect("decoder has full rank");
    assert_eq!(decoded, source);
    Ok(())
}
```

### Low-level GF kernels (still safe — no `unsafe` in your code)

```rust
use rlnc_simdx::kernel;

let x = [1u8, 2, 3, 4];
let mut y = [0u8; 4];

kernel::axpy(0x03, &x, &mut y);      // y[i] ^= c * x[i]; buffers must not overlap
kernel::scale(0x03, &x, &mut y);     // y[i]  = c * x[i]; buffers must not overlap
kernel::scale_inplace(0x03, &mut y); // y[i]  = c * y[i]
```

Check which SIMD path is active:

```rust
println!("kernel: {}", rlnc_simdx::active_kernel());
// e.g. "avx2+ssse3 (tier5)" or "gfni+avx512 (tier1)"
```

---

## What you get

| | |
|--|--|
| **Heap API (`alloc`)** | `Encoder` · `Decoder` · `Recoder` · `CodedPacket` · `GfMatrix` · `AlignedBuffer` |
| **Zero-allocation API** | `Gf8` · `RlncError` · safe `kernel` operations · `active_kernel()` |
| **Field** | GF(2⁸), AES polynomial `0x11B` (same field as Intel GFNI hardware) |
| **SIMD** | Picked automatically at runtime with `std`; compile-time selection without `std` |
| **Safety** | Public kernels check length and non-overlap in release builds and panic on misuse |
| **Dependencies** | None |
| **Tests** | 100 unit tests + 3 doctests |

SVE is experimental and intentionally not part of production dispatch. AArch64
uses the production NEON kernel. The SVE module remains crate-private until a
correct vector-length-agnostic implementation is hardware-tested.

---

## Performance

Reference machine:

| | |
|--|--|
| CPU | AMD Ryzen 7 **5800X3D** |
| RAM | Dual-channel **DDR4** |
| Kernel | `avx2+ssse3 (tier5)` (no GFNI / no AVX-512 on this SKU) |
| Tool | `bench_standalone`, SI GB/s, aligned buffers |

### AXPY `y ^= c·x` — dispatch vs scalar

| Size | Scalar | SIMD dispatch | Speedup |
|------|--------|---------------|---------|
| 1 KiB | ~2.2 GB/s | ~41 GB/s | ~19× |
| 16 KiB | ~2.2 GB/s | **~62 GB/s** | **~29×** |
| 64 KiB | ~2.2 GB/s | **~60 GB/s** | **~28×** |
| 1 MiB | ~2.2 GB/s | **~55 GB/s** | **~25×** |

### SCALE `y = c·x`

| Size | SIMD dispatch (typical) |
|------|-------------------------|
| 16 KiB | ~70 GB/s (~28× scalar) |
| 1 MiB | ~55 GB/s (~22× scalar) |

Mid sizes can vary run-to-run due to clocks and cache state. Prefer several runs
for published measurements.

### Encode (k = 16 random symbols)

| Symbol size | ≈ Time / packet | Payload throughput |
|-------------|-----------------|--------------------|
| 1 KiB | ~500 ns | ~2.0 GB/s |
| 16 KiB | ~5 µs | ~3.2 GB/s |
| 64 KiB | ~24 µs | ~2.7 GB/s |

CPUs with **GFNI** (Ice Lake+, Zen 4+) usually report `gfni+avx2` or
`gfni+avx512` and higher throughput.

### Run benchmarks

The focused multithreaded benchmark compares scalar and runtime-dispatched SIMD
encode/decode throughput using the same RLNC pipeline, separated decoder row
layout, adaptive multi-source encoding, and Rayon parallelism:

```bash
cargo run --release -p rlnc-simdx-mt-bench

# Short run, considering at most two worker threads
cargo run --release -p rlnc-simdx-mt-bench -- --quick --max-threads 2
```

It covers `k = 8, 16, 32` and symbol sizes `64 B, 1 KiB, 4 KiB, 16 KiB,
64 KiB`. Scalar and SIMD worker counts are autotuned independently for each
workload and operation. A portable ASCII metadata panel reports the active SIMD
kernel, host platform, logical CPUs, mode, tuning range, and measurement scope.
Encode and Decode use separate tables; rows stream as workloads complete, and
each table ends with peak-throughput and best-speedup summaries.
Thread-pool construction, fixture creation, and reusable worker-state allocation
are outside timed samples. See the
[`rlnc-simdx-mt-bench` guide](https://github.com/arryboom/rlnc-simdx/blob/main/rlnc-simdx-mt-bench/README.md)
for the measurement model.

The portable standalone kernel benchmark remains available:

```bash
cargo build --release -p rlnc-simdx-bench --bin bench_standalone

# Windows
target\release\bench_standalone.exe --quick

# Linux / macOS
./target/release/bench_standalone --quick
```

You can copy either release binary to another machine with the same OS and
architecture without installing Rust.

Criterion HTML reports are written under `target/criterion/`:

```bash
cargo bench -p rlnc-simdx-bench --bench axpy
cargo bench -p rlnc-simdx-bench --bench decode
```

The workspace benchmark package enables the explicitly unstable
`bench-internals` feature to compare scalar and CPU-validated direct-tier
internals with the supported safe kernel API. Applications should not enable
that feature.

---

## Safety model

```text
Your application (safe Rust)
    → Encoder / Decoder / kernel::axpy | scale | scale_inplace
           → crate-private dispatch and SIMD implementations
```

- **Do:** use equal-length, non-overlapping slices for `axpy` and `scale`.
- **Do:** use `scale_inplace` when multiplying a buffer by a scalar in place.
- **Don't:** treat this crate as encryption or constant-time cryptography.
- **Don't:** use `SimpleRng` as a CSPRNG; it is a small LFSR for coding coefficients.

---

## Build and test

```bash
cargo test --workspace
cargo test -p rlnc-simdx --all-features
cargo check -p rlnc-simdx --no-default-features                    # zero-allocation core
cargo check -p rlnc-simdx --no-default-features --features alloc   # no_std + alloc
```

### Cargo features (`rlnc-simdx`)

| Feature | Default | Meaning |
|---------|---------|---------|
| `alloc` | ✓ | Heap APIs: aligned buffers, matrix, encoder, decoder, and recoder |
| `std` | ✓ | Runtime CPU dispatch; implies `alloc` |
| `bench-internals` | | Unstable scalar/direct-tier APIs for workspace benchmarks only |

With no features, `field`, `error`, `kernel`, and `active_kernel()` remain
available and do not require a global allocator.

---

## Releasing and release verification

Maintainers create and push a signed semantic-version tag after updating the
workspace version, lockfile, changelog, and synchronized READMEs:

```bash
git tag -s v1.2.1
git push origin v1.2.1
```

The tag push runs the release workflow. A manual workflow dispatch may rerun an
existing tag idempotently; it does not create the tag. The workflow requires the
tag version to match the inherited Cargo workspace version and publishes:

- `rlnc-simdx-1.2.1.crate`
- `rlnc-simdx-bench-1.2.1-x86_64-unknown-linux-gnu.tar.gz`
- `rlnc-simdx-bench-1.2.1-x86_64-pc-windows-msvc.zip`
- `rlnc-simdx-bench-1.2.1-x86_64-apple-darwin.tar.gz`
- `SHA256SUMS`

Each native archive contains `bench_standalone` and `rlnc-simdx-mt-bench` (with
`.exe` on Windows), plus `README.md` and `LICENSE`. Download all assets into one
directory and verify them against the sorted manifest:

```bash
# Linux (or macOS with GNU coreutils)
sha256sum -c SHA256SUMS

# macOS fallback
shasum -a 256 -c SHA256SUMS
```

```powershell
# PowerShell: compare these hashes with SHA256SUMS
Get-ChildItem rlnc-simdx-* | Get-FileHash -Algorithm SHA256
```

Release notes also contain the commit SHA and a per-asset SHA-256 table. See the
[changelog](https://github.com/arryboom/rlnc-simdx/blob/main/CHANGELOG.md) for
release details.

---

## Project layout

```text
rlnc-simdx/             # publishable library package (import: rlnc_simdx)
rlnc-simdx-bench/       # Criterion suites and portable bench_standalone binary
rlnc-simdx-mt-bench/    # Rayon-autotuned scalar vs SIMD encode/decode benchmark
plans/                  # architecture, review, and historical design notes
```

Deeper design and release notes:
[architecture](https://github.com/arryboom/rlnc-simdx/blob/main/plans/architecture.md) ·
[changelog](https://github.com/arryboom/rlnc-simdx/blob/main/CHANGELOG.md)

---

## Author

**arryboom** — [https://github.com/arryboom](https://github.com/arryboom)

## License

Licensed under the [Apache License, Version 2.0](LICENSE)
([SPDX: Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)).
