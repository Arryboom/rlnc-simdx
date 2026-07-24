# Post-Rename / Pre-Release Review — `rlnc-simdx`

**Date:** 2026-07-23
**Scope:** rename integrity, Cargo packaging, public API, `no_std`, CI, docs, release gates
**Status:** **Historical review.** Its P0/P1 findings were addressed by the
subsequent remediation; result tables below record the state at review time
rather than the current tree.

## Current remediation outcome

- Release gates are green: formatting, strict all-target/all-feature Clippy,
  workspace tests, all-feature tests, zero-feature and alloc-only checks,
  benchmark binary, Rust 1.89 MSRV, full package verification, and diff check.
- Current suite: **91 unit tests + 3 doctests**, all passing.
- The zero-feature build is a real no-allocator core; heap APIs are gated by
  `alloc`.
- Dispatch internals, raw ISA kernels, finite-field tables, and scalar kernels
  are private by default. The workspace benchmark crate alone opts into the
  explicitly unstable `bench-internals` scalar exposure.
- `axpy_multi` validates all lengths and overlaps before mutation, resolves the
  selected runtime kernel once, and directly invokes it per non-zero block.
- Package verification succeeds with package-local README and Apache-2.0
  license files.
- Performance evidence and limitations are recorded in
  [`remediation_results.md`](remediation_results.md).

---

## 1. Executive verdict (historical snapshot)

| Area | Result |
|------|--------|
| Rename / workspace topology | **Pass** |
| Default / all-feature correctness | **Pass** — 83 unit + 3 doctests |
| `no_std + alloc` | **Pass** |
| Zero-feature core | **Fail** |
| Safe-only public surface claim | **Partial / misleading** |
| CI workflow | **Fail** — malformed WASM job + failing gates |
| Formatting / Clippy | **Fail** |
| Crates.io package inventory | **Buildable list, but incomplete presentation** |
| Publish decision | **Block until P0 fixed** |

### Confirmed strengths

- [`Cargo.toml`](../Cargo.toml:2), [`rlnc-simdx/Cargo.toml`](../rlnc-simdx/Cargo.toml:2), [`rlnc-simdx-bench/Cargo.toml`](../rlnc-simdx-bench/Cargo.toml:2), and [`Cargo.lock`](../Cargo.lock:388) consistently use `rlnc-simdx`.
- Rust crate/import name is correctly [`rlnc_simdx`](../rlnc-simdx/Cargo.toml:14).
- Repository, author, and Apache-2.0 metadata are consistent.
- Final stale-name/path scan returned no active old package references.
- `cargo test --workspace` passes.
- `cargo test -p rlnc-simdx --all-features` passes (83 + 3).
- `cargo check -p rlnc-simdx --no-default-features --features alloc` passes.
- Benchmark binary/package checks pass.

---

## 2. P0 — release blockers

### P0.1 — CI workflow structure is malformed

[`ci.yml`](../.github/workflows/ci.yml:112) places the `wasm_simd128` job at four-space indentation beneath `aarch64_sve`, instead of two-space indentation as a sibling job.

```yaml
# current shape (invalid GitHub Actions job schema)
aarch64_sve:
  ...
  wasm_simd128:  # interpreted as a property inside aarch64_sve
```

**Impact:** GitHub Actions may reject the workflow or never create the WASM job.

**Fix:** Move lines 113–130 to normal job indentation:

```yaml
  wasm_simd128:
    name: WASM SIMD128 (check)
    ...
```

---

### P0.2 — CI lint gate currently fails

Release-gate results:

- `cargo fmt --all -- --check` — **fails** (format drift in six files).
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — **fails first at** [`build.rs`](../rlnc-simdx/build.rs:7) (`empty_line_after_doc_comments`).

Known rustfmt drift:

- [`aligned.rs`](../rlnc-simdx/src/aligned.rs:55)
- [`encoder.rs`](../rlnc-simdx/src/encoder.rs:294)
- [`kernel/mod.rs`](../rlnc-simdx/src/kernel/mod.rs:229)
- [`matrix.rs`](../rlnc-simdx/src/matrix.rs:89)
- [`encode.rs`](../rlnc-simdx-bench/benches/encode.rs:15)
- [`bench_standalone.rs`](../rlnc-simdx-bench/src/bin/bench_standalone.rs:21)

**Impact:** Current CI `lint` job fails before release.

**Fix:** Run `cargo fmt --all`; fix build-script lint; rerun Clippy until it reaches zero warnings.

---

### P0.3 — `--no-default-features` build is broken

`cargo check -p rlnc-simdx --no-default-features` fails because alloc-dependent modules compile unconditionally in [`lib.rs`](../rlnc-simdx/src/lib.rs:88), while their types are feature-gated.

Examples:

- [`decoder.rs`](../rlnc-simdx/src/decoder.rs:12) imports `AlignedBuffer` and `CodedPacket` unconditionally.
- [`encoder.rs`](../rlnc-simdx/src/encoder.rs:12) imports `AlignedBuffer` unconditionally.
- [`matrix.rs`](../rlnc-simdx/src/matrix.rs:15) imports `AlignedBuffer` unconditionally.
- [`recoder.rs`](../rlnc-simdx/src/recoder.rs:9) imports alloc-only packet/buffer types unconditionally.

This contradicts the existing embedded CI command at [`ci.yml`](../.github/workflows/ci.yml:143).

**Decision required:**

#### Preferred: support a real zero-alloc core

- Keep field + scalar/SIMD kernels available without `alloc`.
- Gate alloc-only modules in [`lib.rs`](../rlnc-simdx/src/lib.rs:88):
  - `aligned`
  - `decoder`
  - `matrix`
  - `recoder`
- Preserve `SimpleRng` by moving it to an alloc-independent module (for example `rng.rs`), or carefully gate only alloc-dependent items/imports in `encoder.rs`.
- Add CI checks for both:
  - `--no-default-features`
  - `--no-default-features --features alloc`

#### Alternative: require `alloc`

- Remove the zero-feature CI job.
- Add a clear compile-time requirement / docs statement that the minimum supported profile is `no_std + alloc`.

The current mixed state is not acceptable for release.

---

## 3. P1 — high-priority API and release-quality findings

### P1.1 — “only safe wrappers are public” is not fully true

[`kernel/mod.rs`](../rlnc-simdx/src/kernel/mod.rs:63) publicly exports:

- `AxpyFn = unsafe fn(...)`
- `ScaleFn = unsafe fn(...)`
- `ScaleInplaceFn = unsafe fn(...)`
- [`KernelSet`](../rlnc-simdx/src/kernel/mod.rs:71) with public unsafe function-pointer fields

These are dispatch implementation details and are not needed by users. They create an unnecessary public/semantic-versioning surface and weaken the docs’ “safe wrappers only” message.

**Fix:** Change all four to `pub(crate)`.

---

### P1.2 — scalar implementation is publicly exposed without public-wrapper checks

[`kernel::scalar`](../rlnc-simdx/src/kernel/mod.rs:44) is public. Its vector functions use only `debug_assert_eq!`, for example [`scalar::axpy`](../rlnc-simdx/src/kernel/scalar.rs:52) and [`scalar::scale`](../rlnc-simdx/src/kernel/scalar.rs:72).

In release builds, a length mismatch can silently truncate through iterator `zip` (or behave inconsistently depending on coefficient fast path). This does not cause memory unsafety, but it is a semantic footgun and contradicts the simple public safety story.

**Recommended options:**

1. **Preferred:** make `scalar` `pub(crate)` and use a benchmark-local scalar reference.
2. Or keep it public, add release length/overlap checks, and document it as supported API.
3. Or expose only a clearly named safe reference API while keeping implementation helpers private.

---

### P1.3 — `runtime-dispatch` feature is a no-op

[`Cargo.toml`](../rlnc-simdx/Cargo.toml:23) defines:

```toml
runtime-dispatch = ["std"]
```

but source code never checks `feature = "runtime-dispatch"`. Runtime dispatch is always enabled whenever `std` is enabled. The bench crate enables this feature unnecessarily.

**Fix (simplest):** remove `runtime-dispatch` from both manifests and docs.  
**Alternative:** make it a real switch and include it in defaults.

---

### P1.4 — build script is unused and misleading

[`build.rs`](../rlnc-simdx/build.rs:1) probes the **build host** and emits `host_*` cfg flags, but the source never consumes those flags.

Its header claims it auto-selects the best kernel, while actual runtime selection occurs in [`runtime::detect`](../rlnc-simdx/src/kernel/mod.rs:98).

**Impact:** maintenance noise, cross-compilation confusion, and current Clippy failure.

**Fix:** remove `build.rs` and the empty `[build-dependencies]` section unless a real consumer is introduced. Runtime detection + normal `target_feature` cfg is sufficient.

---

### P1.5 — crate package has no packaged README metadata

`cargo metadata` reports `readme: null` for the library. `cargo package --list` succeeds, but the package inventory does not include the repository-level README (and does not provide the user-facing crates.io landing page expected by the project).

**Fix:** add to [`rlnc-simdx/Cargo.toml`](../rlnc-simdx/Cargo.toml:1):

```toml
readme = "../README.md"
```

Then run a full `cargo package -p rlnc-simdx --allow-dirty` (not only `--list`) and inspect the generated archive. Also confirm the Apache license text is included in the packaged source distribution; if not, copy a package-local `LICENSE` or adopt an explicit packaging strategy.

---

### P1.6 — README quick start is not paste-ready Rust

The example at [`README.md`](../README.md:33) uses `?` at top level without a surrounding function returning `Result`.

**Fix:** wrap it in:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // example
    Ok(())
}
```

or use `unwrap()` consistently. Prefer making it a doctest or integration test to prevent future drift.

---

### P1.7 — repository lacks a visible `.gitignore`

The workspace contains a large `target/` tree, but no `.gitignore` is present in the reviewed file list.

**Fix:** add at least:

```gitignore
/target/
**/*.rs.bk
```

This is repository hygiene rather than package correctness, but should be fixed before publishing the repository.

---

## 4. P2 — cleanup / consistency findings

### P2.1 — private child-module visibility is inconsistent

Although `x86`, `arm`, and `wasm` parents are `pub(crate)`, their child modules/functions still use `pub` in places:

- [`x86/mod.rs`](../rlnc-simdx/src/kernel/x86/mod.rs:5)
- [`arm/mod.rs`](../rlnc-simdx/src/kernel/arm/mod.rs:2)
- [`wasm/mod.rs`](../rlnc-simdx/src/kernel/wasm/mod.rs:2)
- WASM tier functions at [`simd128.rs`](../rlnc-simdx/src/kernel/wasm/simd128.rs:15)
- non-target ARM stubs at [`neon.rs`](../rlnc-simdx/src/kernel/arm/neon.rs:215) and [`sve.rs`](../rlnc-simdx/src/kernel/arm/sve.rs:37)

Parent privacy prevents external access, so this is not a security bug. Change to `pub(crate)` (or private) for clarity and lint hygiene.

---

### P2.2 — crate docs still imply SVE is an active tier

[`lib.rs`](../rlnc-simdx/src/lib.rs:35) lists “NEON / SVE” in the tier table, but [`sve.rs`](../rlnc-simdx/src/kernel/arm/sve.rs:1) explicitly says SVE is experimental and never dispatched.

**Fix:** list tier 7 as NEON only; mention SVE separately as experimental/non-production.

---

### P2.3 — CI feature jobs need tightening

- The GFNI+AVX2 job at [`ci.yml`](../.github/workflows/ci.yml:55) also enables AVX-512 flags, so it does not isolate the named configuration.
- Current Clippy command at [`ci.yml`](../.github/workflows/ci.yml:157) does not include `--all-targets --all-features`.
- Baseline `cargo test --workspace --no-default-features` does not reliably prove the library’s zero-feature topology because the bench crate explicitly enables library features.

**Fix:** use precise package-scoped checks per feature profile.

---

### P2.4 — docs contain historical test-count/state conflicts

Examples include old 59/60-test counts in [`change_review.md`](change_review.md:5) and an old `rlnc` scope label in [`test_coverage_review.md`](test_coverage_review.md:4).

Historical documents may retain dated evidence, but add a prominent “historical snapshot” marker or update current-status summaries to avoid confusion.

---

### P2.5 — public API consistency decisions remain

- README lists `GfMatrix` as a primary API, but it is not re-exported at crate root.
- [`RlncError::NotEnoughPackets`](../rlnc-simdx/src/error.rs:20) appears unused because `decode()` returns `Ok(None)` when incomplete.
- Public `field::tables` exposes raw implementation tables; decide whether that is intentional supported API.

These are not blockers, but should be settled before stabilizing 1.0 semantics.

---

## 5. Release-gate matrix

| Command | Current result |
|---------|----------------|
| `cargo metadata --no-deps --format-version 1` | **Pass** |
| `cargo test --workspace` | **Pass** |
| `cargo test -p rlnc-simdx --all-features` | **Pass** (83 + 3) |
| `cargo check -p rlnc-simdx --no-default-features --features alloc` | **Pass** |
| `cargo check -p rlnc-simdx --no-default-features` | **Fail** |
| `cargo check -p rlnc-simdx-bench --bins` | **Pass** |
| `cargo fmt --all -- --check` | **Fail** |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | **Fail** |
| `cargo package -p rlnc-simdx --allow-dirty --list` | **Pass**, package presentation incomplete |
| GitHub Actions YAML/schema | **Fail / invalid WASM job nesting** |

---

## 6. Recommended implementation order

### Step 1 — make CI and hygiene green

1. Fix WASM job indentation.
2. Run `cargo fmt --all`.
3. Remove/fix `build.rs`.
4. Run Clippy with all targets/features until clean.

### Step 2 — resolve the feature topology

1. Choose true zero-alloc core vs mandatory alloc.
2. Apply module/item cfg gates.
3. Add independent CI checks for zero features and `alloc`.

### Step 3 — close the public API leak

1. Make dispatch pointer aliases and `KernelSet` `pub(crate)`.
2. Decide whether `kernel::scalar` is private or a fully checked supported API.
3. Normalize child-module visibility.

### Step 4 — make the crate publishable/user-ready

1. Configure packaged README.
2. Verify license inclusion.
3. Fix README quick start and make it testable.
4. Add `.gitignore`.
5. Run full `cargo package` verification.

### Step 5 — polish semantics and docs

1. Remove no-op `runtime-dispatch` or implement it.
2. Correct SVE docs.
3. Clarify/re-export `GfMatrix`.
4. Resolve unused error variant and historical docs.

---

## 7. Final decision

```text
CORE CORRECTNESS:     PASS
RENAME:               PASS
DEFAULT / ALL-FEATURE TESTS: PASS
NO_STD + ALLOC:       PASS
ZERO-FEATURE CORE:    FAIL
CI / LINT:            FAIL
PUBLIC API STORY:     NEEDS TIGHTENING
CRATES.IO READY:      NO (P0 fixes required)
```

The implementation is healthy enough for continued development and benchmarking. The remaining blockers are release engineering and API-boundary issues, not a newly discovered SIMD arithmetic failure.
