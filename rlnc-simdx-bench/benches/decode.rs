//! Benchmark: end-to-end decoding throughput (Gaussian elimination).
//!
//! Measures receive+decode after P0/P1 remediation (no GE to_vec, free-list
//! rows, SIMD scale_inplace). Criterion reports MB/s / GB/s via Throughput::Bytes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rlnc_simdx::{Decoder, Encoder, SimpleRng};

fn bench_decode(c: &mut Criterion) {
    let symbol_sizes = [128usize, 1024, 4096, 16384];
    let generation_sizes = [8usize, 16, 32];

    let mut group = c.benchmark_group("decode");
    group.sample_size(20);

    for &k in &generation_sizes {
        for &n in &symbol_sizes {
            let source: Vec<Vec<u8>> = (0..k).map(|i| vec![i as u8; n]).collect();
            let refs: Vec<&[u8]> = source.iter().map(|v| v.as_slice()).collect();
            let enc = Encoder::new(k, n).unwrap();
            let mut rng = SimpleRng::new(0xBEEF);

            // Pre-generate enough innovative packets
            let packets: Vec<_> = (0..k + 4)
                .map(|_| enc.encode_random(&refs, &mut rng).unwrap())
                .collect();

            // Throughput = k * symbol_size bytes recovered per full decode session
            group.throughput(Throughput::Bytes((k * n) as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    format!("k={k}/sym={n}/kernel={}", rlnc_simdx::active_kernel()),
                    n,
                ),
                &n,
                |b, _| {
                    b.iter(|| {
                        let mut dec = Decoder::new(black_box(k), black_box(n)).unwrap();
                        for pkt in packets.iter() {
                            let _ = dec.receive(pkt.clone());
                            if dec.is_complete() {
                                break;
                            }
                        }
                        if dec.is_complete() {
                            let _ = black_box(dec.decode());
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
