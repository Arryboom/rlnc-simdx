//! Benchmark: isolated GF(2⁸) multiply throughput via `scale`.
//!
//! Criterion reports **MB/s / GB/s** because `Throughput::Bytes` is set.
//! Use default Criterion CLI (do **not** pass `--output-format bencher`) to see
//! throughput in the terminal; HTML report always includes thr. charts.
//!
//! Buffers are 64-byte aligned (`AlignedBuffer`).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rlnc_simdx::AlignedBuffer;

fn bench_gf_mul(c: &mut Criterion) {
    let sizes = [64usize, 1024, 65536, 1 << 20];
    let coeff = 0x53u8;

    let mut group = c.benchmark_group("gf_mul");

    for &size in &sizes {
        let x = AlignedBuffer::from_slice(&(0u8..).take(size).collect::<Vec<_>>());
        let mut y = AlignedBuffer::zeroed(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                rlnc_simdx::kernel::scalar::scale(
                    coeff,
                    black_box(x.as_slice()),
                    black_box(y.as_mut_slice()),
                );
            });
        });

        group.bench_with_input(
            BenchmarkId::new(format!("dispatch/{}", rlnc_simdx::active_kernel()), size),
            &size,
            |b, _| {
                b.iter(|| {
                    rlnc_simdx::kernel::scale(
                        coeff,
                        black_box(x.as_slice()),
                        black_box(y.as_mut_slice()),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gf_mul);
criterion_main!(benches);
