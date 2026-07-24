# Expert-Panel Validation of Security Review

**Document under review:** [`plans/security_review.md`](plans/security_review.md)
**Validators:** Rust language / unsafe · Intel µarch · AMD · ARM · HPC / threat modeling · crates.io supply-chain
**Method:** Re-open each finding against current source; accept, regrade, reject, or refine.
**Date:** 2026-07-23

## Implementation follow-up

The remediation plan (H1, H2, H3, M2, M3/M4/M5) has been **implemented** in tree.
See [`CHANGELOG.md`](../CHANGELOG.md) **Unreleased** and [`README.md`](../README.md)
“Safe by default”. Remaining open item from the original audit: **M1 size caps**.

---

## Executive validation verdict

| | |
|--|--|
| **Security report overall quality** | **High** — findings are real, code-backed, and prioritization is mostly right |
| **False positives** | **None material** among HIGH; one **severity nuance** on H3; L3 is process not security |
| **Missed items** | A few **additions** (below), not contradicting the report |
| **Ship recommendation** | **Confirm** security report: safe API usable; fix H1 + H2 before “hardened library” claims |

```text
STATUS:  SECURITY REPORT CONFIRMED (with regrades noted)
BLOCKER for crates.io security claims: H1, H2
```

---

## Finding-by-finding adjudication

### HIGH findings

#### H1 — `AlignedBuffer::new_uninit` public + uninit via `as_slice` / `Deref`

| Panel | Vote | Rationale |
|-------|------|-----------|
| **Rust unsafe** | **Confirm HIGH** | Live code: `pub fn new_uninit` + `as_slice` uses `from_raw_parts` without init. Reading uninit is UB in Rust. `from_slice` calls `new_uninit` then writes — OK only if callers never read before write. Public `new_uninit` + safe `Deref` is a real footgun. |
| **Intel / AMD** | Confirm (lang, not ISA) | N/A to SIMD; uninit heap can leak prior process data → info disclosure. |
| **HPC threat** | Confirm | Attacker who can induce app to call `new_uninit` and serialize buffer → heap disclosure. |

**Evidence (current):**

```rust
// aligned.rs
pub fn new_uninit(len: usize) -> Self { ... alloc(layout) ... }
pub fn as_slice(&self) -> &[u8] {
    unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
}
// from_slice uses new_uninit then copy_from_slice — safe if fully written
```

**Severity regrade:** Keep **HIGH**.  
**Mitigation still correct:** `pub(crate)` / `unsafe fn` / remove; never expose uninit as `&[u8]`.

**Optional refinement:** Even `from_slice` path is fine *after* full write; only **uninit exposure to safe code** is the defect. Report wording is accurate.

---

#### H2 — Public `unsafe` SIMD kernels + length only `debug_assert`

| Panel | Vote | Rationale |
|-------|------|-----------|
| **Rust** | **Confirm HIGH** (library-surface risk) | `pub mod x86` / `arm` + `pub unsafe fn axpy_*` — any crate user can call without going through `kernel::axpy`. Safety is callee-enforced only by docs + debug_assert. |
| **Intel** | Confirm | Wrong feature or OOB → #UD or mem corruption on load/store. Runtime `is_x86_feature_detected!` protects **dispatch**, not raw exports. |
| **ARM** | Confirm | Same for NEON `pub unsafe`. SVE stubs panic — different failure mode. |
| **crates.io** | Confirm | Exporting raw SIMD is fine only under feature gate or `pub(crate)`. |

**Evidence:** `kernel/mod.rs` has `pub mod x86` / `pub mod arm`; all tiers `pub unsafe fn` with `debug_assert_eq!(x.len(), y.len())`.

**Severity regrade:** Keep **HIGH** for a general-purpose crate.  
**Nuance (not a reject):** If the product goal is “advanced users may call tiers,” H2 is **by design** but still must have airtight `# Safety`. Report’s mitigations (`pub(crate)` or `unsafe-kernels` feature) are the right fix.

**Panel note:** Public **safe** `kernel::axpy` already asserts lengths — report correctly separates safe vs unsafe surface.

---

#### H3 — Partial overlap UB in release

| Panel | Vote | Rationale |
|-------|------|-----------|
| **Rust** | **Confirm as real; regrade → MEDIUM–HIGH** | Documented for `axpy`; `scale` only `debug_assert!(!overlaps_partial)`. Overlap is **caller contract**, same class as `ptr::copy` vs `copy_nonoverlapping`. For a **safe** function, Rust community often treats silent UB on overlap as a **soundness bug** if the API is safe. |
| **Intel/AMD** | Confirm impact | SIMD load/store on overlapping regions → torn RMW / wrong results; not always a clean crash. |
| **HPC** | Soften | Hot path cost of release-mode overlap checks can matter; feature-flagged `paranoid` is reasonable. |

**Severity regrade:** Report said HIGH. Panel:  
- If API is **safe** and overlap is possible from pure safe Rust → treat as **soundness → HIGH** is justified.  
- If treated as **documented precondition** like `slice::copy_from_slice` (panics on overlap length but not all alias cases) → **MEDIUM** with explicit docs is defensible.  

**Consensus:** Finding is **valid**. Prefer report stay HIGH for safe APIs, or split:  
- **H3a** safe API soundness (HIGH)  
- **H3b** performance-sensitive optional check (design choice)

**Mitigation still correct:** release check under feature **or** accept documented contract + tests.

---

### MEDIUM findings

#### M1 — Size overflow / unbounded alloc DoS

| Panel | Vote |
|-------|------|
| **All** | **Confirm MEDIUM** |

**Evidence:** `GfMatrix::zeros` → `rows * stride` unchecked; `Decoder::new` allocates `k` rows of `k+n`; `padded_cap` can wrap.

**Not RCE** — report correctly says DoS/panic. Caps + `checked_mul` still the right fix.

**Regrade:** Keep MEDIUM. For untrusted multi-tenant inputs, escalate operational severity to HIGH at **app** layer.

---

#### M2 — `row_axpy` / matrix indices

| Panel | Vote |
|-------|------|
| **Rust** | **Confirm** |

**Evidence:** `debug_assert_ne!(dst, src)` only; `dst == src` in release → `split_at_mut` logic assumes disjoint offsets — same-row is UB / wrong.

Out-of-range row → panic (slice) — availability, not silent corruption.

**Mitigation confirmed:** `assert_ne!` in release + bounds checks for public matrix API.

---

#### M3 — `SimpleRng` not CSPRNG

| Panel | Vote |
|-------|------|
| **HPC / crypto-adjacent** | **Confirm MEDIUM** (integrity, not mem safety) |

Correct: fine for tests; insufficient if adversary chooses linear dependence via predictable coeffs.

**Mitigation confirmed:** document + optional CSPRNG feature.

---

#### M4 — Infinite loop on all-zero coeffs

| Panel | Vote |
|-------|------|
| **Rust** | **Confirm LOW–MEDIUM** |

With this LFSR, all-zero forever is impractical; still defensive programming. **Accept as LOW** in practice; report MEDIUM is slightly aggressive but mitigations are cheap → **keep MEDIUM as “should fix.”**

---

#### M5 — Not constant-time

| Panel | Vote |
|-------|------|
| **All** | **Confirm** as documentation / misuse risk |

Not a memory-safety bug. Correct for RLNC coding lib. **Confirm MEDIUM** as product/security-doc item.

---

### LOW findings

| ID | Panel vote | Notes |
|----|------------|-------|
| L1 OOM panic | **Confirm LOW** | Same as `Vec`; OK |
| L2 unimplemented stubs | **Confirm LOW** | Prefer non-public; panic ≠ RCE |
| L3 file size | **Regrade → informational** | Maintainability, not security vulnerability |
| L4 from_slices no validate | **Confirm LOW / by design** | Integrity at receive |
| L5 free-list | **Confirm positive** | Zero-on-take prevents stale payload leak — good finding |

---

## 3. “What is done well” — panel endorsement

All seven positives in the security report are **confirmed**:

1. Zero runtime deps — **yes**  
2. Safe public kernel length checks — **yes**  
3. Encode/decode without forcing `unsafe` — **yes**  
4. Runtime feature detect before dispatch — **yes**  
5. No secrets / env / Command — **yes**  
6. Size validation on packets/sources — **yes**  
7. Property tests vs scalar — **yes**  

---

## 4. Threat model table — validated

| Threat | Report status | Panel |
|--------|---------------|-------|
| Wrong-size packet | Handled | **Confirm** |
| Right-size bad math | Integrity only | **Confirm** |
| Huge k/n | DoS (M1) | **Confirm** |
| Misused unsafe kernels | Corruption (H2) | **Confirm** |
| `new_uninit` read | UB (H1) | **Confirm** |
| Network/TLS | OOS | **Confirm** |
| Side channel | M5 | **Confirm** |

---

## 5. Gaps the security report under-emphasized (addenda)

Panel adds these without invalidating the report:

| Addendum | Severity | Detail |
|----------|----------|--------|
| **G1** `scale_inplace` public but SIMD tiers exist while docs in matrix claim SIMD | Medium (perf/docs integrity) | Related to change_review H1; security impact low unless timing |
| **G2** `axpy` has no overlap `debug_assert` (only docs); `scale` has debug check | Low–Med | Asymmetry; unify policy |
| **G3** Integer overflow in `k + n` for `row_len` | Subsumed by M1 | Mention explicitly |
| **G4** Recoder/Encoder all-zero retry — same M4 pattern | Low | Same mitigation |
| **G5** Public `as_ptr` / `as_mut_ptr` on `AlignedBuffer` | Low | Safe to expose; enables unsafe footguns elsewhere — document |
| **G6** No `#![deny(unsafe_op_in_unsafe_fn)]` | Low process | Hygiene for future |

No critical vulnerability was **missed** that would reverse the overall rating.

---

## 6. Priority order — confirmed / adjusted

Security report remediations:

```text
P0  new_uninit non-public / no Deref uninit     ← CONFIRMED #1
P0  hide or feature-gate pub unsafe SIMD          ← CONFIRMED #2
P1  checked sizes + caps                          ← CONFIRMED #3
P1  assert_ne row_axpy                            ← CONFIRMED #4
P2  SimpleRng + non-CT docs                       ← CONFIRMED #5
P2  retry cap all-zero coeffs                     ← CONFIRMED #6
P3  split kernel/mod.rs                           ← process, OK
```

**Panel-adjusted order (if one sprint):**

1. H1 uninit (must)  
2. H2 kernel visibility (must for crates.io)  
3. M2 `assert_ne` row_axpy (cheap)  
4. M1 size caps (DoS)  
5. H3 policy decision (check vs document)  
6. M3/M5 docs  
7. M4 retry cap  

---

## 7. Formal panel decision on the security report itself

| Question | Answer |
|----------|--------|
| Are HIGH findings real? | **Yes** (H1, H2 solid; H3 valid, severity philosophy debatable) |
| Are severities inflated? | **Slightly** on H3-as-absolute-HIGH if contract-only model accepted; **not** on H1/H2 |
| Are MEDIUM findings real? | **Yes** |
| False positives? | **No** material FPs |
| Overall suitability rating? | **Confirmed** |
| Can we trust this report for remediation planning? | **Yes — use as backlog** |

### One-liners

| Expert | On the security report |
|--------|------------------------|
| **Rust** | “H1 is textbook uninit exposure. H2 is export-surface soundness. Fix those first.” |
| **Intel** | “H2 matches how we treat raw intrinsics: not for general callers.” |
| **AMD** | “M1 DoS is real for network-facing apps; cap at the API.” |
| **ARM** | “Agree; stubs and free-list notes are fair.” |
| **HPC / threat** | “Threat model is correctly limited; don’t pretend this is AEAD.” |
| **Supply chain** | “Zero-deps clean bill is accurate and valuable.” |

---

## 8. Residual risk (joint)

After implementing H1+H2+M1+M2:

- App-layer DoS still needs quotas for huge generations.  
- RLNC integrity still needs auth outside this crate.  
- Side channels remain if misused with secrets.  

**Confirmed.**

---

*End of security-report validation. Next step when desired: implement P0 remediations (H1/H2) in Code mode.*
