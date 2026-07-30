//! Benchmark: end-to-end decoding throughput (Gaussian elimination).
//!
//! Measures end-to-end and staged public Decoder operations. Criterion reports
//! MB/s / GB/s via Throughput::Bytes.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rlnc_simdx::{CodedPacket, Decoder, Encoder, SimpleRng};

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
                    b.iter_batched(
                        || packets.clone(),
                        |packets| {
                            let mut dec = Decoder::new(black_box(k), black_box(n)).unwrap();
                            for packet in packets {
                                let _ = dec.receive(packet);
                                if dec.is_complete() {
                                    break;
                                }
                            }
                            if dec.is_complete() {
                                let _ = black_box(dec.decode());
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_decode_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_stages");
    group.sample_size(20);
    let k = 16usize;

    for &n in &[1024usize, 4096, 16384, 65536] {
        let source: Vec<Vec<u8>> = (0..k)
            .map(|row| vec![(row as u8).wrapping_add(1); n])
            .collect();
        let refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let encoder = Encoder::new(k, n).unwrap();
        let systematic: Vec<_> = (0..k)
            .map(|index| encoder.encode_systematic(&refs, index).unwrap())
            .collect();
        let mut dense_dependent_coefficients = vec![0u8; k];
        for (index, coefficient) in dense_dependent_coefficients[..k / 2].iter_mut().enumerate() {
            *coefficient = (index as u8).wrapping_mul(17).wrapping_add(1);
        }
        let dense_dependent =
            CodedPacket::from_slices(&dense_dependent_coefficients, &vec![0xA5; n]);
        let mut dense_innovative_coefficients = dense_dependent_coefficients.clone();
        dense_innovative_coefficients[k / 2] = 1;
        let dense_innovative =
            CodedPacket::from_slices(&dense_innovative_coefficients, &vec![0x5A; n]);

        group.throughput(Throughput::Bytes(n as u64));
        group.bench_with_input(BenchmarkId::new("receive_innovative", n), &n, |b, _| {
            b.iter_batched(
                || (Decoder::new(k, n).unwrap(), systematic[0].clone()),
                |(mut decoder, packet)| black_box(decoder.receive(packet).unwrap()),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(
            BenchmarkId::new("receive_innovative_after_8_pivots", n),
            &n,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut decoder = Decoder::new(k, n).unwrap();
                        for packet in &systematic[..k / 2] {
                            decoder.receive(packet.clone()).unwrap();
                        }
                        (decoder, dense_innovative.clone())
                    },
                    |(mut decoder, packet)| black_box(decoder.receive(packet).unwrap()),
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("receive_dependent", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let mut decoder = Decoder::new(k, n).unwrap();
                    for packet in &systematic[..k / 2] {
                        decoder.receive(packet.clone()).unwrap();
                    }
                    (decoder, dense_dependent.clone())
                },
                |(mut decoder, packet)| black_box(decoder.receive(packet).unwrap()),
                BatchSize::SmallInput,
            );
        });

        group.throughput(Throughput::Bytes((k * n) as u64));
        group.bench_with_input(BenchmarkId::new("decode_only", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let mut decoder = Decoder::new(k, n).unwrap();
                    for packet in &systematic {
                        decoder.receive(packet.clone()).unwrap();
                    }
                    decoder
                },
                |mut decoder| black_box(decoder.decode().unwrap()),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_decode, bench_decode_stages);
criterion_main!(benches);
