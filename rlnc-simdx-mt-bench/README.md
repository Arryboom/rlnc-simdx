# rlnc-simdx-mt-bench

Autotuned multithreaded encode/decode throughput comparison for `rlnc-simdx`.

The benchmark compares two GF(2^8) kernel backends:

- **Scalar:** `rlnc_simdx::kernel::scalar`
- **SIMD:** the safe runtime-dispatched `rlnc_simdx::kernel` API

Both backends use the same benchmark-local RLNC encoder/decoder, deterministic
64-byte-aligned source data, full-rank Vandermonde coefficient matrices,
allocation strategy, and Rayon generation-level parallelism. The encoder uses
multi-source AXPY for complete symbols: scalar is the cache-blocked reference,
while SIMD uses the safe adaptive/fused runtime kernel. The decoder mirrors the
library's separated coefficient/payload row layout. Only the kernel backend
changes between each paired measurement.

Independent generations are parallelized because elimination steps inside one
generation have data dependencies. Scalar and SIMD encode/decode measurements
autotune their Rayon worker counts independently.

## Workload matrix

- Generation sizes: `k = 8, 16, 32`
- Symbol sizes: `64 B, 1 KiB, 4 KiB, 16 KiB, 64 KiB`

Effective throughput is reported in GiB/s using `k * symbol_size` recovered or
encoded source bytes per complete generation.

## Run

```bash
cargo run --release -p rlnc-simdx-mt-bench
```

Short CI-style run with at most two workers:

```bash
cargo run --release -p rlnc-simdx-mt-bench -- --quick --max-threads 2
```

The program starts with a portable ASCII metadata panel containing the benchmark
version, host OS/architecture, logical CPU count, active runtime SIMD kernel,
autotuning range, mode, workload matrix, throughput definition, and timing scope.

Encode and Decode use separate ASCII tables. Rows are flushed as soon as each
workload completes, so long full runs show continuous progress. Each row contains
scalar GiB/s and selected worker count, SIMD GiB/s and selected worker count,
and SIMD/scalar speedup. Every table ends with peak scalar throughput, peak SIMD
throughput, and best-speedup summaries; the report ends with total elapsed time.
Autotuning trial details are intentionally not printed.

## Autotuning

For each operation, backend, and workload, the benchmark:

1. caps workers to `available_parallelism()` and optional `--max-threads`;
2. tests powers of two plus the maximum worker count;
3. refines around the best coarse result;
4. uses warmup/calibration and median samples;
5. performs the final measurement at the selected worker count.

Thread-pool construction and fixture generation are outside timed samples.
