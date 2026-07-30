//! Focused coverage for fused and end-to-end hot paths.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rlnc_simdx::{kernel, AlignedBuffer, Decoder, Encoder, Recoder, SimpleRng};

fn bench_fused_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_primitives");
    group.sample_size(30);

    for &size in &[15usize, 16, 17, 31, 32, 33, 63, 64, 65, 4096, 65536] {
        let sources: Vec<AlignedBuffer> = (0..16)
            .map(|source| {
                AlignedBuffer::from_slice(
                    &(0..size)
                        .map(|index| (index as u8).wrapping_add(source as u8))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let source_refs: Vec<&[u8]> = sources.iter().map(AlignedBuffer::as_slice).collect();
        let coefficients: Vec<u8> = (0..16)
            .map(|index| (index as u8).wrapping_mul(17).wrapping_add(1))
            .collect();
        let mut destination = AlignedBuffer::zeroed(size);
        group.throughput(Throughput::Bytes((size * sources.len()) as u64));
        group.bench_with_input(BenchmarkId::new("axpy_multi", size), &size, |b, _| {
            b.iter(|| {
                kernel::axpy_multi(
                    black_box(&coefficients),
                    black_box(&source_refs),
                    black_box(destination.as_mut_slice()),
                );
            });
        });

        let mut inplace = AlignedBuffer::from_slice(sources[0].as_slice());
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("scale_inplace", size), &size, |b, _| {
            b.iter(|| kernel::scale_inplace(black_box(0x53), black_box(inplace.as_mut_slice())));
        });
        group.bench_with_input(BenchmarkId::new("dot", size), &size, |b, _| {
            b.iter(|| black_box(kernel::dot(black_box(&sources[0]), black_box(&sources[1]))));
        });
    }
    group.finish();
}

fn bench_public_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("public_pipeline");
    group.sample_size(20);
    let k = 16usize;
    for &symbol_size in &[1024usize, 4096, 16384] {
        let source: Vec<Vec<u8>> = (0..k)
            .map(|row| vec![(row as u8).wrapping_add(1); symbol_size])
            .collect();
        let source_refs: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let encoder = Encoder::new(k, symbol_size).unwrap();
        let mut rng = SimpleRng::new(0xC0FFEE);
        let packets: Vec<_> = (0..k + 4)
            .map(|_| encoder.encode_random(&source_refs, &mut rng).unwrap())
            .collect();

        group.throughput(Throughput::Bytes((k * symbol_size) as u64));
        group.bench_with_input(
            BenchmarkId::new("decoder_receive_decode", symbol_size),
            &symbol_size,
            |b, _| {
                b.iter_batched(
                    || packets.clone(),
                    |packets| {
                        let mut decoder = Decoder::new(k, symbol_size).unwrap();
                        for packet in packets {
                            let _ = decoder.receive(packet).unwrap();
                            if decoder.is_complete() {
                                break;
                            }
                        }
                        black_box(decoder.decode().unwrap());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("recoder", symbol_size),
            &symbol_size,
            |b, _| {
                b.iter(|| black_box(Recoder::recode(black_box(&packets), black_box(&mut rng))));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fused_primitives, bench_public_pipeline);
criterion_main!(benches);
