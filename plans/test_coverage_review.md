# Expert Panel: Unit Test Coverage Review

**Panel:** Rust QA · Field algebra · SIMD / ISA · RLNC / coding theory · Security
**Scope:** All `#[test]` in `rlnc` (not Criterion / `bench_standalone`)
**Baseline (after P0+P1 package):** **83** unit tests + 3 doctests
**Date:** 2026-07-23

## P0+P1 package implemented

| Item | Status |
|------|--------|
| T1–T2 Overlap + scale/dot length panics | Done (`kernel/proptest.rs`) |
| T4 `row_axpy` same-row panic | Done |
| T5–T7 Encoder/Decoder errors + incomplete decode | Done |
| T8 Recoder hard assert recovery + error paths | Done |
| T9 `row_scale` vs kernel | Done |
| T10 empty axpy | Done |
| T16–T18 Display + axpy_multi panics | Done |

Still open (P2): per-tier scale smoke, WASM host tests, llvm-cov CI.

---

## 1. Executive verdict

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Field / tables correctness | **A** | Strong axiom + table round-trips |
| Scalar golden model | **A−** | Good; nibble tables exhaustive |
| Dispatch vs scalar (property) | **A−** | Best investment so far; product path |
| Per-tier SIMD smoke | **B** | Mostly axpy-only; feature-gated skip |
| Encoder / Decoder E2E | **B+** | Core paths covered; error paths thin |
| Security / API contracts (H3, panics) | **C+** | One length panic; **no overlap panic** tests |
| Matrix GE | **B** | Small cases only |
| Cross-ISA (NEON / WASM) | **C** | NEON only on aarch64; WASM none in unit tests |
| Overall product confidence | **B+** | Safe for core RLNC; gaps are security contracts + scale/inplace tiers + errors |

**Ship tests as-is?** **Yes** for correctness of happy-path encode/decode/SIMD.  
**Need changes?** **Yes — targeted additions**, not a rewrite. Priority list in §6.

---

## 2. Inventory by module

| Module | # Tests (approx) | What they cover | Gap severity |
|--------|------------------|-----------------|--------------|
| `field/tables` | 5 | gen order, exp/log, inv, mul commute | Low |
| `field/gf8` | 10 | +/×/inv/div/pow axioms | Low |
| `kernel/scalar` | 6 | c=0/1, known vec, scale≡axpy empty y, dot, nibble 256×c | Low |
| `kernel/proptest` | 8 | random len/c, unaligned, aligned, scale_inplace, axpy_multi, len panic | Med (missing overlap) |
| `kernel/mod` | 4 | axpy²=0, scale≡axpy, name, tier detect | Low |
| `kernel/x86/*` | ~7 | per-tier **axpy** vs scalar (skip if no CPU) | Med (no scale / inplace) |
| `kernel/arm/neon` | 2 | axpy + c=1 (aarch64 only) | Med off-host |
| `kernel/wasm` | 0 | — | Med for wasm CI |
| `aligned` | 5 | align, empty, from_slice, clone, deref+kernel | Low |
| `matrix` | 5 | align, I RREF, rank 0/2, row_axpy | Med |
| `encoder` | 4 | systematic, random non-zero, align, into_vecs | Med (errors) |
| `decoder` | 5 | round-trip, systematic, redundant, align, free-list | Med (errors, incomplete) |
| `recoder` | 1 | recode→decode (conditional complete) | High-ish (flaky shape) |
| `error` | 0 | Display / variants | Low |
| Doctests | 3 | lib quickstart, active_kernel, AlignedBuffer | Low |

---

## 3. What is already strong (do not regress)

### 3.1 Field algebra (Rust + algebra panel)

- EXP/LOG bijection on 1..255, inv×x=1, generator order.  
- Gf8: XOR add, mul zero/one, commute/associate/distribute, inv/div, pow.  

**Verdict:** Adequate for AES field; no urgent change.

### 3.2 Scalar as oracle

- `nibble_tables_correctness` for **all x in 0..255** vs `gf_mul` is the right golden for SSSE3/NEON/WASM.  
- c=0 / c=1 special cases tested.

### 3.3 Property tests on **public dispatch** (critical after H2)

- Random lengths including tails (1, 15, 17, 33, 65, 777, …).  
- Unaligned offsets (1, 3, 7, 15, …).  
- AlignedBuffer path.  
- `scale_inplace` vs scale.  
- `axpy_multi` vs sequential scalar.  
- `#[should_panic]` length mismatch on public axpy.

**Verdict:** This is the **correct primary net** for “safe API + active tier == scalar”. Keep expanding *this* style, not only per-file smoke tests.

### 3.4 RLNC E2E

- Random encode → decode recovers symbols.  
- Systematic identity.  
- Redundant packet → rank unchanged + free-list recycle.  
- Alignment of packets / matrix / decoder rows.

---

## 4. Gaps by expert (severity)

### P0 — should add soon

| ID | Gap | Why | Suggested test |
|----|-----|-----|----------------|
| **T1** | **No `#[should_panic]` for overlap** on `axpy` / `scale` | H3 is a release safety contract; currently untested | Split one buffer into overlapping views or same range; expect panic message contains `overlap` |
| **T2** | **No panic test for `scale` length** | Only axpy length panic tested | Mirror `public_axpy_panics_on_len_mismatch` for scale |
| **T3** | **Tier tests almost only `axpy`** | `scale` + `scale_inplace` are public product paths (GE normalize) | For each tier (or only dispatch): scale vs scalar, scale_inplace vs scale |
| **T4** | **`row_axpy` same-row panic (M2)** | `assert_ne!(dst, src)` untested | `#[should_panic] matrix.row_axpy(0, c, 0)` |

### P1 — important for confidence

| ID | Gap | Why | Suggested test |
|----|-----|-----|----------------|
| **T5** | Encoder **error paths** | `InvalidParameters`, count/size mismatch, index OOR | `Encoder::new(0,n)`, wrong source len, `encode_systematic(..., k)` |
| **T6** | Decoder **error paths** | PacketSizeMismatch on receive | Wrong coeff/payload lengths |
| **T7** | Decoder **incomplete decode** | `decode()` → `Ok(None)` before full rank | receive k-1 packets → None |
| **T8** | **Recoder** weak | Single test; may not assert if rank never completes | Force enough recodes or systematic mix; always assert recovery **or** fail test if incomplete |
| **T9** | Matrix **row_scale** / non-trivial GE | Only tiny ranks | 3×3 known inverse; row_scale then check vs scalar |
| **T10** | **Empty / zero-length** kernels on public API | proptest includes 0 but E2E rarely uses n=0 (API forbids 0) | Explicit `axpy(c, &[], &mut [])` ok |

### P2 — ISA / CI / polish

| ID | Gap | Why | Suggested test |
|----|-----|-----|----------------|
| **T11** | **scale** per x86 tier | Only axpy smoke | Optional; dispatch props already cover active tier |
| **T12** | **c=1 XOR path** on x86 tiers | Branch exists; only NEON + scalar + prop_c_one | prop already hits dispatch c=1; add scale c=1 |
| **T13** | WASM unit tests | Never run on host | CI wasi/wasm check only; add wasm-bindgen/wasi tests later |
| **T14** | NEON on x86 CI | cfg-gated out | Rely on aarch64 CI job (already planned) |
| **T15** | SVE | intentionally unimplemented | No unit test needed; document skip |
| **T16** | `RlncError` Display | no snapshot | one test formatting each variant |
| **T17** | `dot` public length panic | public assert | should_panic |
| **T18** | `axpy_multi` length panics | multi assert paths | mismatched sources / coeffs |

### What **not** to do

| Anti-pattern | Reason |
|--------------|--------|
| Require 100% line coverage on all SIMD tiers on every laptop | Features absent → tests skip; line % lies |
| Public re-export of unsafe tiers only for tests | Undoes H2; use `#[cfg(test)]` access inside crate only |
| Heavy proptest dependency | Current xorshift props are zero-dep and good enough |
| Test SVE success path | Code is `unimplemented!` |

---

## 5. Coverage quality notes (not just “count of tests”)

### 5.1 Feature-gated skips are correct but silent

```rust
if !is_x86_feature_detected!("avx2") { return; }
```

- **Good:** no false failures on weak CPUs.  
- **Risk:** CI host without AVX2 never executes tier5 body.  
- **Mitigation:** CI jobs with `RUSTFLAGS=+avx2,+ssse3` (you have similar); document that **proptest dispatch** is the portable guarantee.

### 5.2 Recoder test control flow

```rust
if dec.is_complete() {
    let decoded = ...
    assert_eq!(...);
}
// if never complete → test still passes
```

**Panel: High priority to fix structure** — a green test that doesn’t assert recovery is worse than no test. Always assert `is_complete()` after enough packets or use systematic bases.

### 5.3 Security remediations under-tested

| Remediation | Tested? |
|-------------|---------|
| H1 new_uninit private | Compile-time only (good) |
| H2 tiers private | Compile-time only |
| H3 overlap assert | **No** |
| M2 row_axpy assert | **No** |
| M4 retry cap | **No** (hard to force LFSR stuck; optional force via test-only hook later) |

### 5.4 Doctests

Quickstart E2E is valuable. Keep them compiling; they double as integration smoke.

---

## 6. Recommended change list (ordered)

### Must-add (small PR)

1. `public_axpy_panics_on_overlap`  
2. `public_scale_panics_on_overlap`  
3. `public_scale_panics_on_len_mismatch`  
4. `matrix_row_axpy_panics_on_same_row`  
5. Fix `recode_then_decode` to **require** `is_complete()` and always assert symbols  

### Should-add

6. Encoder/Decoder error-path table (one test module `api_errors`)  
7. `decode_returns_none_when_incomplete`  
8. Dispatch `scale` / `scale_inplace` for c∈{0,1,0x53} + lens {0,1,16,17,64,100} (fold into proptest)  
9. `axpy_multi` panic on coeff/source mismatch  

### Nice-to-have

10. Error Display snapshots  
11. Known 4×4 GE against hand vector  
12. WASM target test job (not host unit tests)  
13. Optional `cargo llvm-cov` in CI for **safe modules only** (field, encoder, decoder, proptest) — ignore raw SIMD line %  

---

## 7. Coverage map (conceptual, not llvm-cov %)

```text
                    Tested well          Thin / missing
Field mul/add/inv      ████████████      ░
Scalar axpy/scale      ███████████       ░
Public dispatch axpy   ██████████        ░ unaligned+random OK
Public dispatch scale  ████████          ░░ less than axpy
scale_inplace          ███████           ░░
axpy_multi             ██████            ░░ panics
Overlap panics         ░                 ████ need
Encode happy           ████████          ░
Encode errors          ██                ████ need
Decode happy           █████████         ░
Decode errors/incomplete ██              ████ need
Recoder                ████              ██ flaky assert
Matrix GE              ██████            ██ larger / row_scale
Per-tier scale         ██                ████
NEON/WASM              ██ / ░            CI-dependent
```

---

## 8. Panel one-liners

| Expert | Quote |
|--------|--------|
| **Rust QA** | “Property tests on the safe API are the right backbone after H2. Add overlap panics — that’s free regression insurance for H3.” |
| **Algebra** | “Field and nibble tables are in good shape; don’t spend cycles re-proving AES there.” |
| **SIMD** | “Per-tier axpy smoke is enough if dispatch props stay strong; add scale_inplace at dispatch level, not six copy-pasted tier tests.” |
| **RLNC** | “Fix recoder so green always means recovered data. Add incomplete-decode and packet size errors.” |
| **Security** | “Contracts you assert in release without tests will rot. T1–T4 are the security-test minimum.” |

---

## 9. Formal decision

```text
STATUS:  COVERAGE ADEQUATE FOR CORE CORRECTNESS
ACTION:  ADD TARGETED TESTS (P0 list) — no full suite rewrite
BLOCKER: None for development; recoder soft-assert is the worst smell
PRIORITY: T1–T5 + recoder harden before next crates.io cut
```

---

*End of unit-test coverage review.*
