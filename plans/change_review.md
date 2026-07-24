# Team Review — Remediation Change Set (post-expert_review)

**Panel:** Intel ISA · AMD Zen · ARM NEON/SVE · Rust API/safety · HPC network coding
**Scope:** P0/P1/P2 work landed after `plans/expert_review.md`
**Evidence:** source re-read 2026-07-23 · `cargo test -p rlnc-simdx` **59/59** green (later **60/60**)
**Verdict:** **Approve with minor follow-ups** — P0 goals met; one doc/perf overclaim on `scale_inplace`

## Later status

Follow-ups from this review (SIMD `scale_inplace`, free-list decoder, docs) and the
**security remediation plan** (H1–H3, M2–M5) are implemented. Public API is
**safe-by-default**; see [`README.md`](../README.md) and [`architecture.md`](architecture.md) §4.

---

## 1. Change map (what landed)

| ID | Intent | Status | Primary files |
|----|--------|--------|---------------|
| P0a | Decoder GE no `to_vec` / no temp scale buffer | **Done** | [`decoder.rs`](rlnc-simdx/src/decoder.rs) |
| P0b | Public length checks + aliasing docs | **Done** | [`kernel/mod.rs`](rlnc-simdx/src/kernel/mod.rs), [`lib.rs`](rlnc-simdx/src/lib.rs) |
| P1-c1 | `c==1` XOR SIMD all x86 + NEON | **Done** | `kernel/x86/*`, [`neon.rs`](rlnc-simdx/src/kernel/arm/neon.rs) |
| P1-ARM | Delete NEON dual align paths | **Done** | `neon.rs` |
| P1-SVE | Gate off non-production SVE | **Done** | [`sve.rs`](rlnc-simdx/src/kernel/arm/sve.rs) |
| P1-matrix | `row_scale` → kernel path | **Partial** | [`matrix.rs`](rlnc-simdx/src/matrix.rs) → scalar `scale_inplace` only |
| P2-API | CodedPacket FFI + error field docs | **Done** | `encoder.rs`, `error.rs`, `aligned.rs` |
| P2-encode | `axpy_multi` cache-blocked encode | **Done** | `kernel/mod.rs`, `encoder.rs` |
| Props | Bit-exact random c/len/align | **Done** | [`proptest.rs`](rlnc-simdx/src/kernel/proptest.rs) |

**Explicit non-goals kept (per user):** x86 dual aligned/unaligned load paths **not** removed.

---

## 2. Panel verdicts

### 2.1 Rust core / safety — **Approve**

**What we like**

- Public `axpy` / `scale` / `dot` use `assert_eq!` on lengths in **release** — closes UB footgun.
- Aliasing contract documented at crate root and on kernels.
- Decoder borrow story is now correct: working `row` is a distinct `AlignedBuffer` from `self.rows[r]` → `kernel::axpy(coeff, self.rows[r].as_slice(), row.as_mut_slice())` is sound without `to_vec`.
- Property tests cover unaligned offsets, aligned buffers, `c∈{0,1}`, panic on mismatch, `axpy_multi` vs loop.
- `CodedPacket::{into_vecs, from_vecs, from_slices, *_slice}` is the right FFI surface without forcing callers off aligned storage.

**Nits (non-blocking)**

| # | Issue | Severity | Note |
|---|--------|----------|------|
| R1 | Partial-overlap check is only `debug_assert` on `scale` | Low | Documented caller contract; OK for perf. Optional: release check behind `cfg(debug_assertions)` only is current — fine. |
| R2 | `matrix.rs` `row_scale` indentation looks odd | Cosmetic | Compiles; run `cargo fmt` |
| R3 | `AlignedBuffer::into_vec` still copies | Low | Documented; true move would need custom allocator → Vec — out of scope |

**Rust sign-off:** Safe default path is materially better. Ship.

---

### 2.2 HPC / network coding — **Approve with one performance gap**

**What we like**

- **P0 smoking gun fixed:** GE hot loop no longer heap-allocates per pivot row.
- Encode uses blocked multi-source AXPY — correct first cut for DRAM-bound `k × n` work.
- Systematic / random encode-decode tests still green.

**Residual issues**

| # | Issue | Severity | Detail |
|---|--------|----------|--------|
| H1 | **`scale_inplace` is scalar-only** | **Medium** | Public API routes to `scalar::scale_inplace` only. Doc on `GfMatrix::row_scale` claims “SIMD via scale_inplace” — **incorrect**. Pivot normalize and matrix GE normalize do **not** use GFNI/AVX for large symbols. |
| H2 | One `AlignedBuffer::zeroed` per `receive` | Expected | Not a GE-loop alloc; still O(1) alloc per packet. Optional later: free-list of rows. |
| H3 | `decode()` still allocates full `ordered` matrix | Low | Once per completed generation; acceptable. Could swap-permute in place later. |
| H4 | `axpy_multi` BLOCK=4096 fixed | Low | Fine default; could tune per L2 size later. |

**HPC sign-off:** Decode path is no longer “SIMD-washed by allocs.” Fix H1 before claiming matrix/normalize is SIMD-accelerated.

---

### 2.3 Intel — **Approve**

**What we like**

- `c == 1` early-out to pure XOR on GFNI/AVX tiers — high value for systematic + many GE steps.
- Length checks stay **outside** `#[target_feature]` bodies (public wrapper) — correct: assert + then call specialized fn.
- x86 dual align paths retained as requested; still optional for thr. on modern µarch (prior review stands).

**Nits**

| # | Issue | Severity |
|---|--------|----------|
| I1 | `scale_inplace` not using GFNI for large `n` | Medium (same as H1) |
| I2 | GFNI AVX-512 XOR helper exists; scale `c==1` uses `copy_from_slice` (good) | — |

**Intel sign-off:** Kernel remediation direction correct. Prefer SIMD `scale_inplace` as next win.

---

### 2.4 AMD — **Approve**

- Zen4 GFNI path benefits from `c==1` XOR same as Intel.
- Blocked encode helps multi-channel DRAM; no objection to 4 KiB blocks.
- No regression expected vs prior tier ordering.

**AMD sign-off:** OK.

---

### 2.5 ARM — **Approve**

**What we like**

- Dual NEON paths removed; single `vld1q` stream + `c==1` XOR helper — matches ARM guidance.
- SVE correctly **not** in dispatch; stubs document why (tbl VL limits). Honest engineering.

**Nits**

| # | Issue | Severity |
|---|--------|----------|
| A1 | SVE left as `unimplemented!` if someone enables `+sve` and calls stubs | Low | Doc says don’t call; optional `#[deprecated]` / private module later |
| A2 | No AArch64 CI in-repo yet | Medium process | Not a code defect in this diff |

**ARM sign-off:** NEON path is production-clean; SVE correctly parked.

---

## 3. Code review of critical diffs

### 3.1 Decoder `receive` — **LGTM**

```text
// Working row ≠ self.rows  →  disjoint
kernel::axpy(coeff, self.rows[r].as_slice(), row.as_mut_slice());
kernel::scale_inplace(inv, row.as_mut_slice());
```

Correctness: pivot normalize uses `inv` of pivot, in-place on full augmented row (coeffs + payload) — **required** for GE. Good.

Comment “no heap allocation for forward elimination or pivot normalisation” is accurate **relative to the previous `to_vec` / second AlignedBuffer**. Still allocates the working row once — wording is OK if read carefully.

### 3.2 Public kernels — **LGTM**

- `assert_eq` on public API + `debug_assert` length inside SIMD is the right split.
- `axpy_multi` correctness proven by property test vs sequential scalar axpy.

### 3.3 `scale_inplace` — **Request follow-up**

```rust
pub fn scale_inplace(c: u8, y: &mut [u8]) {
    scalar::scale_inplace(c, y);  // always scalar
}
```

Panel consensus:

1. Fix `GfMatrix::row_scale` docs to say “via scalar or future SIMD in-place scale” **or**
2. Implement SIMD in-place scale (load→mul→store same pointer, sequential VL chunks — **safe** for non-overlapping vector iterations).

### 3.4 SVE module — **LGTM as gate**

Better than shipping wrong math. Keep out of `detect()`.

### 3.5 Property tests — **LGTM**

Good coverage without external `proptest` dep (zero-deps crate policy preserved). Consider later: more trials, and direct calls to each x86 tier under `is_x86_feature_detected!` (today tests **dispatch** vs scalar, which is the product path).

---

## 4. Regression / risk matrix

| Risk | Likelihood | Impact | Mitigation now |
|------|------------|--------|----------------|
| Decode correctness regression | Low | High | Round-trip + systematic tests green |
| Release panic on bad lengths | Low (API misuse) | Medium | Intentional; documented |
| Perf claim overstate on normalize | Medium | Medium | H1 — docs/code mismatch |
| Someone calls `axpy_sve` | Low | High (panic) | Module docs; not in dispatch |
| `axpy_multi` slower for tiny `n` | Low | Low | BLOCK ≥ n → one pass; OK |

---

## 5. Checklist vs original expert review

| Original finding | This change set |
|------------------|-----------------|
| P0 decoder allocs | **Fixed** |
| P0 public length checks | **Fixed** |
| P1 c==1 all ISA | **Fixed** (x86+NEON; WASM not updated — minor) |
| P1 NEON dead dual path | **Fixed** |
| P1 SVE not production | **Fixed** (gated) |
| P1 GfMatrix row_scale SIMD | **Incomplete** (calls scalar inplace) |
| P2 fused encode | **Fixed** (blocked multi) |
| P2 API FFI / docs | **Fixed** |
| Property tests | **Fixed** |
| Drop x86 align branch | **Out of scope** (per user) |

---

## 6. Formal decision

```text
STATUS:  APPROVED WITH FOLLOW-UPS
TESTS:   59 unit + 3 doctests green
SHIP:    Yes for correctness / P0 performance narrative on GE-axpy
BLOCKER: None for merge
FOLLOW:  (1) SIMD scale_inplace or fix docs
         (2) cargo fmt on matrix.rs
         (3) optional: WASM c==1; aarch64 CI; free-list decoder rows
```

### One-liners

| Expert | Quote |
|--------|--------|
| **Rust** | “Borrow fix was free and correct; release length asserts are non-negotiable. Approved.” |
| **HPC** | “Decode axpy path is finally honest. Don’t document scale_inplace as SIMD until it is.” |
| **Intel** | “c=1 XOR on GFNI is the right quick win. Next: in-place GFNI scale.” |
| **AMD** | “No objections; encode blocking is sensible.” |
| **ARM** | “NEON cleaned up; SVE honesty > wrong SVE. Approved.” |

---

## 7. Suggested next PR (optional, ordered)

1. **SIMD `scale_inplace`** via dispatch (or reuse scale with identical ptr after documenting full-alias OK for sequential VL stores).
2. Fix `row_scale` doc string to match reality until (1).
3. `cargo fmt` + silence remaining `missing_docs` on WASM stubs.
4. Add decode Criterion bench vs pre-P0 baseline (alloc-heavy) for regression story.
5. Cross-ISA CI (aarch64, wasm32).

---

*End of team change review.*
