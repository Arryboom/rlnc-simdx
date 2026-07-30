//! Benchmark: axpy (y[i] ^= c * x[i]) throughput at multiple buffer sizes.
//!
//! Compares scalar, safe public dispatch, and CPU-validated direct-tier handles.
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
        let x = AlignedBuffer::from_slice(&(0..size).map(|index| index as u8).collect::<Vec<_>>());
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

        for tier in rlnc_simdx::kernel::bench::available_axpy_tiers() {
            group.bench_with_input(
                BenchmarkId::new(format!("direct/{}", tier.name()), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        tier.axpy(coeff, black_box(x.as_slice()), black_box(y.as_mut_slice()));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_axpy_tail_alignment(c: &mut Criterion) {
    const TAIL_SIZES: [usize; 9] = [15, 16, 17, 31, 32, 33, 63, 64, 65];
    let mut group = c.benchmark_group("axpy_tail_alignment");
    group.sample_size(30);

    for size in TAIL_SIZES {
        let source_data: Vec<u8> = (0..size).map(|index| index as u8).collect();
        let source = AlignedBuffer::from_slice(&source_data);
        let mut destination = AlignedBuffer::from_slice(&vec![0xAA; size]);
        let mut source_unaligned = vec![0u8; size + 1];
        source_unaligned[1..].copy_from_slice(&source_data);
        let mut destination_unaligned = vec![0xAA; size + 1];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("aligned", size), &size, |b, _| {
            b.iter(|| {
                rlnc_simdx::kernel::axpy(
                    0x53,
                    black_box(source.as_slice()),
                    black_box(destination.as_mut_slice()),
                );
            });
        });
        group.bench_with_input(BenchmarkId::new("unaligned", size), &size, |b, _| {
            b.iter(|| {
                rlnc_simdx::kernel::axpy(
                    0x53,
                    black_box(&source_unaligned[1..]),
                    black_box(&mut destination_unaligned[1..]),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_axpy, bench_axpy_tail_alignment);
criterion_main!(benches);
