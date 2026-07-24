//! Benchmark: axpy (y[i] ^= c * x[i]) throughput at multiple buffer sizes.
//!
//! Compares scalar vs the safe public dispatch path (`kernel::axpy`).
//! Raw SIMD tiers are crate-private (H2); isolation benches use dispatch only.
//!
//! Criterion reports **MB/s / GB/s** when `Throughput::Bytes` is set.
//! Buffers are 64-byte aligned so the aligned SIMD fast-path is exercised.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rlnc_simdx::AlignedBuffer;

fn bench_axpy(c: &mut Criterion) {
    let sizes = [64usize, 1024, 16 * 1024, 64 * 1024, 1 << 20];
    let coeff = 0x53u8;

    let mut group = c.benchmark_group("axpy");
    group.sample_size(50);

    for &size in &sizes {
        let x = AlignedBuffer::from_slice(&(0u8..).take(size).collect::<Vec<_>>());
        let mut y = AlignedBuffer::from_slice(&vec![0xAAu8; size]);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                rlnc_simdx::kernel::scalar::axpy(
                    coeff,
                    black_box(x.as_slice()),
                    black_box(y.as_mut_slice()),
                )
            });
        });

        group.bench_with_input(
            BenchmarkId::new(format!("dispatch/{}", rlnc_simdx::active_kernel()), size),
            &size,
            |b, _| {
                b.iter(|| {
                    rlnc_simdx::kernel::axpy(
                        coeff,
                        black_box(x.as_slice()),
                        black_box(y.as_mut_slice()),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_axpy);
criterion_main!(benches);
