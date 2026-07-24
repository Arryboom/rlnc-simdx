# Security Review — `rlnc-simdx` crate

**Mode:** Security Reviewer (static + architectural)
**Date:** 2026-07-23
**Scope (historical names):** the library now at `rlnc-simdx/`, its then-present
`build.rs`, and the workspace surface
**Out of scope:** network transport, auth, multi-tenant hosting of this crate

## Remediation status (updated)

| ID | Finding | Status after implementation plan |
|----|---------|----------------------------------|
| H1 | Public `new_uninit` / uninit read | **Fixed** — `pub(crate)` only |
| H2 | Public `unsafe` SIMD tiers | **Fixed** — modules + fns `pub(crate)`; safe public wrappers only |
| H3 | Overlap UB on safe axpy/scale | **Fixed** — release `assert!` non-overlap on public `axpy`/`scale` |
| M2 | `row_axpy` same-row | **Fixed** — `assert_ne!(dst, src)` |
| M3/M5 | Weak RNG / non-CT | **Documented** — README + `lib.rs` + `SimpleRng` docs |
| M4 | All-zero RNG loop | **Fixed** — max 100 retries then force non-zero |
| M1 | Unbounded k/n alloc | **Open** — app quotas; optional library caps planned |

Validation of this review: [`security_review_validation.md`](security_review_validation.md).

---

## Executive summary

| Area | Rating | Notes |
|------|--------|-------|
| Secrets / credentials | **Clean** | No keys, tokens, env-bound config |
| Dependency supply chain | **Strong** | Zero runtime deps |
| Memory safety (safe API) | **Good** | Length asserts on public kernels |
| Memory safety (`unsafe` / SIMD) | **Moderate risk** | Public `unsafe` kernels + alias contract |
| Availability / DoS | **Moderate** | Unbounded alloc by design; rare panics |
| Cryptographic strength | **N/A / weak RNG** | RLNC ≠ crypto; `SimpleRng` not CSPRNG |
| Constant-time | **Not provided** | Document if used near secrets |

**Overall:** Suitable for a performance coding library **if callers stay on the safe API** and treat sizes as trusted or capped. Not a crypto library. Address **High** items before crates.io “security posture” claims.

---

## 1. Secrets, env, supply chain

| Check | Result |
|-------|--------|
| Hardcoded secrets / private keys | **None** |
| `std::env` / dotenv / env-coupled config | **None** in library |
| Runtime dependencies | **None** (ideal for auditability) |
| `build.rs` (historical) | Host CPUID only → `rustc-cfg`; later removed as unused |

**Mitigation:** Keep zero-deps policy. Do not add telemetry or cloud SDKs without review.

---

## 2. Findings (severity-ordered)

### HIGH

#### H1 — `AlignedBuffer::new_uninit` is public and exposes uninit memory

[`aligned.rs`](rlnc-simdx/src/aligned.rs): public `new_uninit` → `as_slice()` / `Deref` returns `&[u8]` over **uninitialized** bytes.

**Impact:** Safe code can read uninit memory → **undefined behavior** / info leak of prior heap contents.

**Mitigation:**

- Make `new_uninit` `pub(crate)` or `unsafe fn` with clear contract, **or**
- Remove it; keep only `zeroed` / `from_slice`.
- Never expose uninit via `Deref` without `MaybeUninit` API.

#### H2 — Public `unsafe` SIMD entry points with only `debug_assert` lengths

All `kernel::x86::*` / `kernel::arm::*` kernels are `pub unsafe fn` and use `debug_assert_eq!(x.len(), y.len())`.

**Impact:** External `unsafe` callers who skip length equality or CPU feature checks → OOB SIMD stores / illegal instruction → crash or memory corruption.

**Mitigation:**

- Prefer `pub(crate)` for tier kernels; export only through `kernel::axpy` / `scale` / `scale_inplace`.
- Or feature-gate: `#[cfg(feature = "unsafe-kernels")]`.
- Document `# Safety` as: (1) CPU features, (2) equal lengths, (3) no partial overlap.

#### H3 — Partial buffer overlap is UB in release

Public `scale` only `debug_assert!(!overlaps_partial(...))`. `axpy` documents disjoint requirement but does not check.

**Impact:** Overlapping `x`/`y` (except full-alias cases handled by `scale_inplace`) → data races / torn SIMD RMW → silent corruption.

**Mitigation:**

- Release-mode check behind a `cfg` / `paranoid` feature, **or**
- Keep as documented contract and add fuzz tests for non-overlap assumption in safe paths only.

---

### MEDIUM

#### M1 — Integer overflow on large dimensions (alloc DoS / panic)

Examples:

- `GfMatrix::zeros(rows, cols)` → `AlignedBuffer::zeroed(rows * stride)` without checked mul.
- `Decoder::new(k, n)` → `row_len = k + n`, `k` rows.
- `AlignedBuffer::padded_cap`: `(len + ALIGN - 1) & !(ALIGN - 1)` can wrap for huge `len`.

**Impact:** Attacker-controlled `k`/`n` → panic (layout overflow) or multi-GB OOM → **DoS**. Not RCE by itself.

**Mitigation:**

```rust
// checked arithmetic + hard caps
const MAX_SYMBOL: usize = 16 << 20;
const MAX_GEN: usize = 4096;
rows.checked_mul(stride).and_then(|s| ...) 
```

Return `RlncError::InvalidParameters` instead of panic/OOM where possible.

#### M2 — `GfMatrix::row` / `row_mut` / `row_axpy` trust indices

Out-of-range `r` → slice panic. `row_axpy(dst, c, src)` with `dst == src` only `debug_assert_ne!` → release can create **overlapping** `split_at_mut` views → UB.

**Mitigation:** `assert_ne!(dst, src)` in release; bounds-check or `get`/`Result` API for public matrix ops.

#### M3 — `SimpleRng` is not cryptographically secure

Xorshift-style LFSR, seed-reproducible. Fine for tests/bench; **not** for adversarial RLNC where coeff unpredictability matters.

**Mitigation:** Document “not CSPRNG”; optional feature for `rand_core` / OS RNG inject.

#### M4 — Unbounded / long loops on pathological RNG

`while coeffs.iter().all(|&c| c == 0) { rng.fill(...) }` — theoretically infinite if RNG stuck (not with this LFSR in practice).

**Mitigation:** Max retries then force a non-zero coefficient.

#### M5 — No constant-time guarantees

Table lookups and data-dependent branches on field elements. Side-channel leakage if coeffs/payloads are secret.

**Mitigation:** Explicit crate-level note: **not constant-time; do not process secrets**.

---

### LOW

#### L1 — Alloc failure panics

`NonNull::new(ptr).expect("allocation failed")` — OOM aborts thread (like `Vec`). Acceptable for Rust libs; document for embedded.

#### L2 — `unimplemented!` stubs (SVE / cross-arch NEON / WASM)

Panic if called on wrong target. Low risk (not on default dispatch path). Prefer private modules.

#### L3 — File size / modular boundary

| File | Approx. lines | Flag |
|------|---------------|------|
| [`kernel/mod.rs`](rlnc-simdx/src/kernel/mod.rs) | ~540 | Near/over 500 — split dispatch vs API |
| [`kernel/x86/gfni_avx512.rs`](rlnc-simdx/src/kernel/x86/gfni_avx512.rs) | ~290 | OK |
| [`decoder.rs`](rlnc-simdx/src/decoder.rs) | ~330 | OK |

No monolith >1k lines. Kernel modules are correctly layered; **unsafe surface is too wide publicly** (see H2).

#### L4 — `CodedPacket::from_slices` accepts arbitrary sizes

No validation until `Decoder::receive`. By design; not a memory-safety bug.

#### L5 — Free-list growth

Free-list roughly bounded by generation size; zeros rows on take → no stale data leak between packets. **Good.**

---

## 3. What is done well

1. **Zero runtime dependencies** — minimal attack surface.
2. **Public `kernel::axpy` / `scale` / `dot` length checks in release** — correct pattern.
3. **Safe path does not require caller `unsafe`** for encode/decode.
4. **Runtime CPU feature detection** before selecting SIMD function pointers (avoids #UD for normal API).
5. **No secrets, no env coupling, no process spawning.**
6. **Decoder/encoder validate packet and source sizes** with typed errors.
7. **Property tests** vs scalar reduce silent arithmetic bugs that could become integrity issues.

---

## 4. Threat model snapshot

| Threat | Status |
|--------|--------|
| Malicious coded packet (wrong sizes) | Handled → `RlncError` |
| Malicious packet (right size, bad math) | Integrity only; no mem corruption on safe API |
| Untrusted `k`/`n` sizes | **DoS via alloc** (M1) |
| Caller uses `unsafe` kernels wrongly | **Memory corruption** (H2) |
| Caller uses `new_uninit` + read | **UB** (H1) |
| Network attacker on wire | Out of scope (no TLS/auth here) |
| Side-channel on field ops | Not mitigated (M5) |

---

## 5. Recommended remediations (priority)

```text
P0  Make new_uninit non-public or unsafe; never Deref uninit
P0  Hide or feature-gate pub unsafe SIMD kernels
P1  checked_mul + max k/n for Encoder/Decoder/GfMatrix/AlignedBuffer
P1  assert_ne!(dst, src) on row_axpy in release
P2  Document SimpleRng + non-CT clearly in lib.rs / README
P2  Retry cap on all-zero coeff generation
P3  Split kernel/mod.rs if it keeps growing
```

---

## 6. Residual risk statement

After P0/P1 fixes, residual risk is mainly:

- Application-level DoS from large allocations (mitigate with quotas at app layer).
- Integrity of decoded data under adversarial coding (inherent to RLNC; use authentication outside this crate).
- Side channels if misused with secrets.

---

## 7. Verdict

| Decision | |
|----------|--|
| **Ship internal / research** | Yes, with safe-API discipline |
| **crates.io “production”** | After H1 + H2 + M1/M2 |
| **Security-critical / CT** | **No** without redesign |

*End of security review.*
