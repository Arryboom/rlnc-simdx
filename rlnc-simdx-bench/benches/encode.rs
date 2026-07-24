//! Benchmark: end-to-end encoding throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rlnc_simdx::{Encoder, SimpleRng};

fn bench_encode(c: &mut Criterion) {
    let k = 32usize;
    let symbol_sizes = [128usize, 1024, 4096, 65536];

    let mut group = c.benchmark_group("encode");
    group.sample_size(20);

    for &n in &symbol_sizes {
        let source: Vec<Vec<u8>> = (0..k).map(|i| vec![i as u8; n]).collect();
        let refs: Vec<&[u8]> = source.iter().map(|v| v.as_slice()).collect();
        let enc = Encoder::new(k, n).unwrap();
        let mut rng = SimpleRng::new(0xC0FFEE);

        // Throughput = k * symbol_size bytes of source data encoded per call
        group.throughput(Throughput::Bytes((k * n) as u64));

        group.bench_with_input(
            BenchmarkId::new(
                format!("k={k}/sym={n}/kernel={}", rlnc_simdx::active_kernel()),
                n,
            ),
            &n,
            |b, _| {
                b.iter(|| {
                    for _ in 0..k {
                        let _ = enc.encode_random(black_box(&refs), black_box(&mut rng));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
