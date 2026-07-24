# Expert Panel: Old vs New `bench_standalone` Performance

**Machine (both runs):** AMD Ryzen 7 **5800X3D**, dual-channel **DDR4**, `avx2+ssse3 (tier5)`, no GFNI / no AVX-512
**Compare:** old binary (per-tier + dispatch) vs latest (scalar + **safe dispatch only**)
**Question:** Is the apparent regression real, or measurement / methodology artifact?
**Published numbers:** [README benchmarks](../README.md#benchmarks-reference-machine) (two later full runs on the same machine)

### Follow-up runs (post-analysis)

Two consecutive full `bench_standalone` runs on the same 5800X3D confirmed:

- **AXPY dispatch** very stable (~62 GB/s @ 16 KiB, ~60 @ 64 KiB, ~55 @ 1 MiB).
- **SCALE @ 64 KiB / 256 KiB** still run-dependent (~32–70 GB/s) — treat as noisy; 16 KiB ~70 and 1 MiB ~55 stable.
- **Encode k=16 @ 64 KiB** ~23–25 µs/pkt (~2.7 GB/s payload thr.).

---

## 1. Executive verdict

| Question | Answer |
|----------|--------|
| Is the test **inaccurate**? | **Partially** — same harness, but **not fully like-for-like**, and one outlier looks noisy |
| Is there a **real** slowdown? | **Yes, small on the safe wrapper; larger on encode at 64 KiB** |
| Is the **AVX2 kernel** itself broken? | **Unlikely** — mid-size AXPY matches old to ~0.1% |
| Root cause of real deltas? | **Safety checks + encode path change (`axpy_multi`)**, not wrong SIMD math |

**Bottom line:** Hot-path AXPY/SCALE for ≥16 KiB is **essentially the same**. Tiny sizes and **encode@64 KiB** show real, explainable overhead from security remediations. The **SCALE@256 KiB ~2×** jump is **not trusted** without a re-run (high risk of noise).

---

## 2. Apples-to-apples table (dispatch only)

### AXPY — dispatch

| Size | Old | New | Δ time | Δ thr. |
|------|-----|-----|--------|--------|
| 64 B | 9.5 ns · 6.72 GB/s | 10.4 ns · 6.17 GB/s | **+9%** | −8% |
| 1 KiB | 23.0 ns · 44.5 | 24.4 ns · 41.9 | **+6%** | −6% |
| 16 KiB | 263.1 ns · 62.3 | 263.5 ns · 62.2 | **0%** | 0% |
| 64 KiB | 1.60 µs · 41.0 | 1.43 µs · 45.7 | **−11%** | +12% |
| 256 KiB | 4.51 µs · 58.1 | 4.64 µs · 56.5 | +3% | −3% |
| 1 MiB | 19.18 µs · 54.7 | 19.22 µs · 54.6 | **0%** | 0% |

### SCALE — dispatch

| Size | Old | New | Δ time | Note |
|------|-----|-----|--------|------|
| 64 B | 9.1 ns | 10.6 ns | **+16%** | Fixed overhead visible |
| 1 KiB | 21.9 | 22.5 | +3% | |
| 16 KiB | 226.9 | 224.8 | ~0% | |
| 64 KiB | 2.02 µs | 1.55 µs | **−23%** | Likely noise / frequency; treat carefully |
| 256 KiB | 4.05 µs · 64.8 GB/s | **8.29 µs · 31.6 GB/s** | **~+105%** | **Outlier — re-measure** |
| 1 MiB | 18.32 | 19.49 | +6% | |

### ENCODE (k=16)

| Symbol | Old | New | Δ |
|--------|-----|-----|---|
| 1 KiB | 478 ns | 505 ns | **+6%** |
| 4 KiB | 1.25 µs | 1.27 µs | +2% |
| 16 KiB | 4.76 µs | 5.25 µs | **+10%** |
| 64 KiB | 19.88 µs | 25.33 µs | **+27%** |

### Scalar (control)

Scalar AXPY/SCALE match within ~1–3% on all sizes → **same machine class, same measurement method for the bulk of the work**. Not a broken timer.

---

## 3. Methodology caveats (why “degrade” can look worse than it is)

### 3.1 New binary no longer prints per-tier direct calls

Old:

```text
scalar | dispatch | ssse3 | avx2+ssse3
```

New (after H2 crate-private tiers):

```text
scalar | dispatch only
```

- **Fair compare:** old **dispatch** vs new **dispatch** (table above).  
- **Unfair:** old **direct `avx2+ssse3`** vs new **dispatch** — would mix path differences.  
  On the old run, direct avx2 ≈ dispatch (e.g. 262.4 vs 263.1 ns @16 KiB) → dispatch cost was already tiny.

### 3.2 Single-sample median harness

Standalone bench: ~1.5 s budget, median of samples.  
- Fine for order-of-magnitude SIMD wins (20–30×).  
- **±5–15%** on absolute thr. is normal (boost clocks, background, power).  
- **±100%** (SCALE 256 KiB) needs a **second run** before declaring regression.

### 3.3 Throughput definition

Both: SI GB/s = `size / time` on **payload size** for axpy/scale. Same definition → thr. deltas mirror time deltas.

---

## 4. Root-cause analysis (expert panel)

### 4.1 Rust / safety — **real fixed cost on every safe call**

New public path does **every** call:

```text
assert_eq!(len)           // always
assert!(!ranges_overlap)  // pointer arithmetic + compare  (H3)
OnceLock load + indirect call  (unchanged idea)
→ SIMD body
```

| Size | Kernel work | Check overhead | Expected effect |
|------|-------------|----------------|-----------------|
| 64 B | tiny | **visible** | 5–20% slower — **matches data** |
| 1 KiB | small | small | few % — **matches** |
| ≥16 KiB | dominates | ~noise | flat — **matches AXPY** |

**Panel:** Not a measurement error for tiny buffers. Expected tax of **release-mode soundness**.

### 4.2 Intel / AMD µarch — **kernel body looks healthy**

- AXPY 16 KiB / 1 MiB **identical** → nibble tables + `vpshufb` path not wrecked.  
- `c==1` XOR path **not exercised** in this bench (`COEFF = 0x53`) → no credit/debit from that change here.  
- Dual aligned/unaligned branch still present on x86 (user asked not to remove) — same as old on aligned buffers → both take aligned path.

**Panel:** No evidence of a systematic AVX2 implementation regression on this CPU.

### 4.3 SCALE@256 KiB ~2× — **suspect noise, not kernel death**

Why skepticism:

1. SCALE@1 MiB only **+6%**, not +100%.  
2. AXPY@256 KiB only **+3%**.  
3. Old SCALE@64 KiB was **slower** than new (2.02 vs 1.55 µs) — irregular pattern screams **variance**, not a clean algorithmic hit.

**Action:** Re-run `bench_standalone` 3×; if 256 KiB scale stays ~8 µs, dig into scale path + branch; if it returns ~4 µs, **discard outlier**.

### 4.4 HPC / encode — **real regression plausible at 64 KiB**

Old encode: likely `for i in 0..k { axpy(c_i, source[i], payload) }` — **k full passes**.

New encode: `axpy_multi` with **BLOCK = 4096**:

```text
for off in 0..n step 4096:
  for each source i:
    axpy(c_i, source[i][off..end], y[off..end])
```

For **k=16, n=64 KiB**:

- Old: **16** safe `axpy` calls (if already on dispatch) or 16 kernel entries  
- New: `65536/4096 = 16` blocks × up to **16** sources = up to **256** safe `axpy` calls  

Each call pays **length + overlap asserts** + slice metadata.  

| Effect | Direction |
|--------|-----------|
| Better cache locality for multi-source | Can help DRAM-bound large n |
| **More wrapper invocations** | Hurts when asserts dominate; hurts moderate n |
| Your data: 64 KiB encode **+27%** | Fits “more assert traffic” more than “better locality won” |

At 1–4 KiB encode, fewer blocks → smaller delta (**+2–6%**) — also fits.

**Panel:** Encode regression is **likely real** and tied to **`axpy_multi` × safety checks**, not wrong GF mul.

### 4.5 ARM / other ISA

Not in these numbers (x86 tier5 host). No cross-ISA claim.

---

## 5. Is “degraded” the right story for users?

| Workload | Story |
|----------|--------|
| **AXPY ≥16 KiB** | **No meaningful degrade** — thr. within noise of old |
| **AXPY 64 B–1 KiB** | **Mild real tax** (~5–10%) from release checks |
| **SCALE large** | Mostly flat; **ignore 256 KiB until re-run** |
| **Encode k=16 @ 64 KiB** | **Real ~25% worse** — product of blocked multi + per-chunk asserts |
| **SIMD vs scalar** | Still **~20–30×** on dispatch — product still “SIMD wins big” |

You did **not** lose the multi-ten× SIMD win. You paid a small **safety tax** and an **encode structure tax**.

---

## 6. Recommendations (panel consensus)

### Confirm measurement

1. Re-run latest binary **3 times** full (not `--quick`).  
2. Focus: SCALE 256 KiB, encode 64 KiB.  
3. Optional: pin process / high performance power plan to cut clock noise.

### If encode regression is confirmed (likely)

| Fix | Idea |
|-----|------|
| **A (preferred)** | `axpy_multi` calls **internal** tier fn after **one** length/overlap check on full buffers, not `axpy()` per chunk (avoids N× asserts) |
| **B** | Larger BLOCK (e.g. 64 KiB) or single-pass when `n ≤ L2` |
| **C** | Keep public `axpy` checks; document multi-source internal path as trusted |

### If tiny-size AXPY tax must go to zero

| Option | Tradeoff |
|--------|----------|
| Feature `paranoid-checks` off by default | Faster; weaker footguns |
| `#[inline(always)]` + ensure overlap check optimizes | Small win only |
| Accept 5–10% on 64 B | Correct for library safety |

### Bench product

- Optionally restore **internal** per-tier bench behind `#[cfg(test)]` or a `bench-kernels` feature that re-exports tiers for measurement only — so you can still compare ssse3 vs avx2 without public unsafe API.

---

## 7. One-liners

| Expert | Quote |
|--------|--------|
| **HPC** | “16 KiB–1 MiB AXPY is the same. Encode@64K pays for too many safe-wrapper entries in `axpy_multi`.” |
| **Rust** | “Release length+overlap asserts are a real fixed cost — visible only when the buffer is tiny. Not a false test.” |
| **Intel** | “Kernel throughput at cache-friendly sizes is intact; don’t redesign pshufb over a 9% hit at 64 bytes.” |
| **AMD** | “±10% single-run thr. is noise band; the 2× scale@256K needs a re-run before blaming Zen/AVX2.” |
| **Measurement** | “Same scalar baseline → machine and thr. formula are consistent. Methodology change is only ‘no direct tier rows’.” |

---

## 8. Final answer to the user

| | |
|--|--|
| **Inaccurate test?** | Not fundamentally — but **one outlier (SCALE 256 KiB)** is likely inaccurate; treat as re-run required. |
| **True degrade?** | **Slight** on small buffers and **noticeable on encode@64 KiB**; **not** a collapse of SIMD thr. |
| **Why?** | (1) H3/H2 safe wrapper checks every call; (2) `axpy_multi` multiplies those checks; (3) normal bench variance. |

**Ship narrative:** Security remediations cost **single-digit %** on bulk AXPY and more on multi-source encode structure — fixable without reopening public unsafe SIMD.
