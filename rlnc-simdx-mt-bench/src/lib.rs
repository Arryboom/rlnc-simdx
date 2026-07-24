//! Autotuned multithreaded RLNC benchmark support.
//!
//! Scalar and SIMD measurements use the same benchmark-local encoder, decoder,
//! fixtures, allocation strategy, and Rayon parallelism. Only the GF(2^8)
//! kernel backend changes.

use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use rlnc_simdx::{kernel, AlignedBuffer, Gf8};
use std::{
    hint::black_box,
    marker::PhantomData,
    time::{Duration, Instant},
};

/// Generation sizes covered by the default benchmark matrix.
pub const GENERATION_SIZES: [usize; 3] = [8, 16, 32];
/// Symbol sizes covered by the default benchmark matrix.
pub const SYMBOL_SIZES: [usize; 5] = [64, 1024, 4096, 16 * 1024, 64 * 1024];

const MAX_CALIBRATED_GENERATIONS: usize = 1 << 20;

/// Runtime controls for the benchmark.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Run short measurements suitable for CI smoke testing.
    pub quick: bool,
    /// Maximum number of Rayon workers considered by the autotuner.
    pub max_threads: usize,
}

impl Config {
    /// Build a configuration capped to the machine's available parallelism.
    #[must_use]
    pub fn for_machine(quick: bool, requested_max_threads: Option<usize>) -> Self {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let requested = requested_max_threads.unwrap_or(available).max(1);
        Self {
            quick,
            max_threads: requested.min(available),
        }
    }
}

/// Best measured throughput for one backend.
#[derive(Clone, Copy, Debug)]
pub struct BackendResult {
    /// Effective source-data throughput in GiB/s.
    pub gib_per_second: f64,
    /// Rayon worker count selected by the autotuner.
    pub threads: usize,
}

/// Scalar and SIMD results for one workload.
#[derive(Clone, Copy, Debug)]
pub struct CaseResult {
    /// Number of source symbols in the generation.
    pub generation_size: usize,
    /// Bytes per source symbol.
    pub symbol_size: usize,
    /// Best scalar result.
    pub scalar: BackendResult,
    /// Best runtime-dispatched SIMD result.
    pub simd: BackendResult,
}

impl CaseResult {
    /// SIMD throughput divided by scalar throughput.
    #[must_use]
    pub fn speedup(self) -> f64 {
        self.simd.gib_per_second / self.scalar.gib_per_second
    }
}

#[derive(Clone, Copy)]
struct Timing {
    tune_duration: Duration,
    tune_samples: usize,
    final_duration: Duration,
    final_samples: usize,
}

impl Timing {
    fn new(quick: bool) -> Self {
        if quick {
            Self {
                tune_duration: Duration::from_millis(2),
                tune_samples: 2,
                final_duration: Duration::from_millis(5),
                final_samples: 3,
            }
        } else {
            Self {
                tune_duration: Duration::from_millis(12),
                tune_samples: 3,
                final_duration: Duration::from_millis(50),
                final_samples: 5,
            }
        }
    }
}

/// Runs the benchmark matrix with isolated Rayon thread pools.
pub struct Runner {
    max_threads: usize,
    timing: Timing,
}

impl Runner {
    /// Create a benchmark runner.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            max_threads: config.max_threads.max(1),
            timing: Timing::new(config.quick),
        }
    }

    /// Highest worker count considered by the autotuner.
    #[must_use]
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// Benchmark encoding for one `(generation_size, symbol_size)` workload.
    pub fn benchmark_encode(
        &self,
        generation_size: usize,
        symbol_size: usize,
    ) -> Result<CaseResult, String> {
        let fixture = Fixture::new(generation_size, symbol_size)?;
        fixture.verify_encode::<ScalarKernel>()?;
        fixture.verify_encode::<SimdKernel>()?;
        let effective_bytes = fixture.effective_bytes();

        let scalar = self.autotune(
            effective_bytes,
            || EncodeState::<ScalarKernel>::new(&fixture),
            |state| state.encode(&fixture),
        )?;
        let simd = self.autotune(
            effective_bytes,
            || EncodeState::<SimdKernel>::new(&fixture),
            |state| state.encode(&fixture),
        )?;

        Ok(CaseResult {
            generation_size,
            symbol_size,
            scalar,
            simd,
        })
    }

    /// Benchmark decoding for one `(generation_size, symbol_size)` workload.
    pub fn benchmark_decode(
        &self,
        generation_size: usize,
        symbol_size: usize,
    ) -> Result<CaseResult, String> {
        let fixture = Fixture::new(generation_size, symbol_size)?;
        fixture.verify_decode::<ScalarKernel>()?;
        fixture.verify_decode::<SimdKernel>()?;
        let effective_bytes = fixture.effective_bytes();

        let scalar = self.autotune(
            effective_bytes,
            || DecodeState::<ScalarKernel>::new(&fixture),
            |state| {
                assert!(state.decode(&fixture), "full-rank fixture became singular");
                digest_decoded(&state.rows, fixture.generation_size)
            },
        )?;
        let simd = self.autotune(
            effective_bytes,
            || DecodeState::<SimdKernel>::new(&fixture),
            |state| {
                assert!(state.decode(&fixture), "full-rank fixture became singular");
                digest_decoded(&state.rows, fixture.generation_size)
            },
        )?;

        Ok(CaseResult {
            generation_size,
            symbol_size,
            scalar,
            simd,
        })
    }

    fn autotune<S, M, F>(
        &self,
        effective_bytes: usize,
        make_state: M,
        run: F,
    ) -> Result<BackendResult, String>
    where
        S: Send,
        M: Fn() -> S,
        F: Fn(&mut S) -> u64 + Sync,
    {
        let coarse = coarse_candidates(self.max_threads);
        let mut observations = Vec::with_capacity(coarse.len() + 4);

        for threads in coarse {
            let score = self.measure_candidate(
                threads,
                effective_bytes,
                self.timing.tune_duration,
                self.timing.tune_samples,
                &make_state,
                &run,
            )?;
            observations.push((threads, score));
        }

        let coarse_best = best_thread_count(&observations);
        for threads in refinement_candidates(coarse_best, self.max_threads) {
            if observations
                .iter()
                .any(|&(measured_threads, _)| measured_threads == threads)
            {
                continue;
            }
            let score = self.measure_candidate(
                threads,
                effective_bytes,
                self.timing.tune_duration,
                self.timing.tune_samples,
                &make_state,
                &run,
            )?;
            observations.push((threads, score));
        }

        let best_threads = best_thread_count(&observations);
        let gib_per_second = self.measure_candidate(
            best_threads,
            effective_bytes,
            self.timing.final_duration,
            self.timing.final_samples,
            &make_state,
            &run,
        )?;

        Ok(BackendResult {
            gib_per_second,
            threads: best_threads,
        })
    }

    fn measure_candidate<S, M, F>(
        &self,
        threads: usize,
        effective_bytes: usize,
        minimum_sample_time: Duration,
        samples: usize,
        make_state: &M,
        run: &F,
    ) -> Result<f64, String>
    where
        S: Send,
        M: Fn() -> S,
        F: Fn(&mut S) -> u64 + Sync,
    {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("rlnc-mt-{threads}-{index}"))
            .build()
            .map_err(|error| format!("unable to create {threads}-thread Rayon pool: {error}"))?;
        let mut states: Vec<S> = (0..threads).map(|_| make_state()).collect();

        let generations = calibrate_generations(&pool, &mut states, minimum_sample_time, run);
        let mut elapsed = Vec::with_capacity(samples);
        for _ in 0..samples {
            let started = Instant::now();
            let digest = run_parallel(&pool, &mut states, generations, run);
            elapsed.push(started.elapsed());
            black_box(digest);
        }
        elapsed.sort_unstable();
        let median = elapsed[elapsed.len() / 2].as_secs_f64();
        let bytes = effective_bytes as f64 * generations as f64;
        Ok(bytes / median / (1024.0 * 1024.0 * 1024.0))
    }
}

fn calibrate_generations<S, F>(
    pool: &ThreadPool,
    states: &mut [S],
    minimum_sample_time: Duration,
    run: &F,
) -> usize
where
    S: Send,
    F: Fn(&mut S) -> u64 + Sync,
{
    let mut generations = states.len().max(1);
    loop {
        let started = Instant::now();
        let digest = run_parallel(pool, states, generations, run);
        let elapsed = started.elapsed();
        black_box(digest);

        if elapsed >= minimum_sample_time || generations >= MAX_CALIBRATED_GENERATIONS {
            return generations;
        }

        let elapsed_nanos = elapsed.as_nanos().max(1);
        let multiplier = (minimum_sample_time.as_nanos() / elapsed_nanos)
            .clamp(2, 8)
            .try_into()
            .unwrap_or(8usize);
        generations = generations
            .saturating_mul(multiplier)
            .min(MAX_CALIBRATED_GENERATIONS);
    }
}

fn run_parallel<S, F>(pool: &ThreadPool, states: &mut [S], generations: usize, run: &F) -> u64
where
    S: Send,
    F: Fn(&mut S) -> u64 + Sync,
{
    let worker_count = states.len();
    pool.install(|| {
        states
            .par_iter_mut()
            .enumerate()
            .map(|(worker, state)| {
                let assigned =
                    generations / worker_count + usize::from(worker < generations % worker_count);
                (0..assigned).fold(0u64, |digest, generation| {
                    digest.wrapping_add(run(state) ^ generation as u64)
                })
            })
            .reduce(|| 0, u64::wrapping_add)
    })
}

fn coarse_candidates(max_threads: usize) -> Vec<usize> {
    let mut candidates = Vec::new();
    let mut threads = 1usize;
    while threads < max_threads {
        candidates.push(threads);
        threads = threads.saturating_mul(2);
    }
    candidates.push(max_threads);
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn refinement_candidates(best: usize, max_threads: usize) -> Vec<usize> {
    let start = best.saturating_sub(2).max(1);
    let end = best.saturating_add(2).min(max_threads);
    (start..=end).collect()
}

fn best_thread_count(observations: &[(usize, f64)]) -> usize {
    observations
        .iter()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map_or(1, |&(threads, _)| threads)
}

trait KernelBackend: Send + Sync + 'static {
    fn axpy(coefficient: u8, source: &[u8], destination: &mut [u8]);
    fn scale_inplace(coefficient: u8, destination: &mut [u8]);
}

struct ScalarKernel;

impl KernelBackend for ScalarKernel {
    #[inline]
    fn axpy(coefficient: u8, source: &[u8], destination: &mut [u8]) {
        kernel::scalar::axpy(coefficient, source, destination);
    }

    #[inline]
    fn scale_inplace(coefficient: u8, destination: &mut [u8]) {
        kernel::scalar::scale_inplace(coefficient, destination);
    }
}

struct SimdKernel;

impl KernelBackend for SimdKernel {
    #[inline]
    fn axpy(coefficient: u8, source: &[u8], destination: &mut [u8]) {
        kernel::axpy(coefficient, source, destination);
    }

    #[inline]
    fn scale_inplace(coefficient: u8, destination: &mut [u8]) {
        kernel::scale_inplace(coefficient, destination);
    }
}

struct Fixture {
    generation_size: usize,
    symbol_size: usize,
    sources: Vec<AlignedBuffer>,
    coefficients: Vec<Vec<u8>>,
    coded_rows: Vec<AlignedBuffer>,
}

impl Fixture {
    fn new(generation_size: usize, symbol_size: usize) -> Result<Self, String> {
        if generation_size == 0 || generation_size > 255 || symbol_size == 0 {
            return Err(format!(
                "invalid workload: generation_size={generation_size}, symbol_size={symbol_size}"
            ));
        }

        let sources = make_sources(generation_size, symbol_size);
        let coefficients = make_vandermonde(generation_size);
        let mut coded_rows = Vec::with_capacity(generation_size);
        for coefficient_row in &coefficients {
            let mut row = AlignedBuffer::zeroed(generation_size + symbol_size);
            row.as_mut_slice()[..generation_size].copy_from_slice(coefficient_row);
            for (&coefficient, source) in coefficient_row.iter().zip(&sources) {
                ScalarKernel::axpy(
                    coefficient,
                    source.as_slice(),
                    &mut row.as_mut_slice()[generation_size..],
                );
            }
            coded_rows.push(row);
        }

        Ok(Self {
            generation_size,
            symbol_size,
            sources,
            coefficients,
            coded_rows,
        })
    }

    fn effective_bytes(&self) -> usize {
        self.generation_size * self.symbol_size
    }

    fn verify_encode<B: KernelBackend>(&self) -> Result<(), String> {
        let mut state = EncodeState::<B>::new(self);
        state.encode(self);
        for (actual, expected) in state.payloads.iter().zip(&self.coded_rows) {
            if actual.as_slice() != &expected.as_slice()[self.generation_size..] {
                return Err("scalar/SIMD encoder output differs from the fixture".to_owned());
            }
        }
        Ok(())
    }

    fn verify_decode<B: KernelBackend>(&self) -> Result<(), String> {
        let mut state = DecodeState::<B>::new(self);
        if !state.decode(self) {
            return Err("full-rank fixture failed to decode".to_owned());
        }
        for (row, source) in state.rows.iter().zip(&self.sources) {
            if &row.as_slice()[self.generation_size..] != source.as_slice() {
                return Err("decoded symbols differ from source symbols".to_owned());
            }
        }
        Ok(())
    }
}

fn make_sources(generation_size: usize, symbol_size: usize) -> Vec<AlignedBuffer> {
    (0..generation_size)
        .map(|source_index| {
            let mut source = AlignedBuffer::zeroed(symbol_size);
            let mut state = 0x9E37_79B9_7F4A_7C15u64
                ^ (source_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
            for byte in source.as_mut_slice() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = state as u8;
            }
            source
        })
        .collect()
}

fn make_vandermonde(generation_size: usize) -> Vec<Vec<u8>> {
    (0..generation_size)
        .map(|row| {
            let point = Gf8::new((row + 1) as u8);
            (0..generation_size)
                .map(|column| point.pow(column as u32).value())
                .collect()
        })
        .collect()
}

struct EncodeState<B> {
    payloads: Vec<AlignedBuffer>,
    backend: PhantomData<B>,
}

impl<B: KernelBackend> EncodeState<B> {
    fn new(fixture: &Fixture) -> Self {
        Self {
            payloads: (0..fixture.generation_size)
                .map(|_| AlignedBuffer::zeroed(fixture.symbol_size))
                .collect(),
            backend: PhantomData,
        }
    }

    fn encode(&mut self, fixture: &Fixture) -> u64 {
        const BLOCK: usize = 4096;

        for (payload, coefficient_row) in self.payloads.iter_mut().zip(&fixture.coefficients) {
            payload.as_mut_slice().fill(0);
            let mut offset = 0usize;
            while offset < fixture.symbol_size {
                let end = (offset + BLOCK).min(fixture.symbol_size);
                for (&coefficient, source) in coefficient_row.iter().zip(&fixture.sources) {
                    B::axpy(
                        coefficient,
                        &source.as_slice()[offset..end],
                        &mut payload.as_mut_slice()[offset..end],
                    );
                }
                offset = end;
            }
        }
        digest_payloads(&self.payloads)
    }
}

struct DecodeState<B> {
    rows: Vec<AlignedBuffer>,
    backend: PhantomData<B>,
}

impl<B: KernelBackend> DecodeState<B> {
    fn new(fixture: &Fixture) -> Self {
        Self {
            rows: (0..fixture.generation_size)
                .map(|_| AlignedBuffer::zeroed(fixture.generation_size + fixture.symbol_size))
                .collect(),
            backend: PhantomData,
        }
    }

    fn decode(&mut self, fixture: &Fixture) -> bool {
        for (row, coded) in self.rows.iter_mut().zip(&fixture.coded_rows) {
            row.as_mut_slice().copy_from_slice(coded.as_slice());
        }

        let k = fixture.generation_size;
        let mut rank = 0usize;
        for column in 0..k {
            let Some(pivot) = (rank..k).find(|&row| self.rows[row].as_slice()[column] != 0) else {
                continue;
            };
            self.rows.swap(rank, pivot);

            let pivot_value = self.rows[rank].as_slice()[column];
            if pivot_value != 1 {
                B::scale_inplace(
                    Gf8::new(pivot_value).inv().value(),
                    self.rows[rank].as_mut_slice(),
                );
            }

            for target in (rank + 1)..k {
                let coefficient = self.rows[target].as_slice()[column];
                if coefficient != 0 {
                    axpy_rows::<B>(&mut self.rows, rank, target, coefficient);
                }
            }
            rank += 1;
            if rank == k {
                break;
            }
        }

        if rank != k {
            return false;
        }

        for pivot in (0..k).rev() {
            for target in 0..pivot {
                let coefficient = self.rows[target].as_slice()[pivot];
                if coefficient != 0 {
                    axpy_rows::<B>(&mut self.rows, pivot, target, coefficient);
                }
            }
        }

        true
    }
}

fn axpy_rows<B: KernelBackend>(
    rows: &mut [AlignedBuffer],
    source_index: usize,
    destination_index: usize,
    coefficient: u8,
) {
    debug_assert_ne!(source_index, destination_index);
    if source_index < destination_index {
        let (before_destination, destination_and_after) = rows.split_at_mut(destination_index);
        B::axpy(
            coefficient,
            before_destination[source_index].as_slice(),
            destination_and_after[0].as_mut_slice(),
        );
    } else {
        let (before_source, source_and_after) = rows.split_at_mut(source_index);
        B::axpy(
            coefficient,
            source_and_after[0].as_slice(),
            before_source[destination_index].as_mut_slice(),
        );
    }
}

fn digest_payloads(payloads: &[AlignedBuffer]) -> u64 {
    payloads
        .iter()
        .enumerate()
        .fold(0u64, |digest, (index, payload)| {
            let bytes = payload.as_slice();
            let sample = u64::from(bytes[0])
                | (u64::from(bytes[bytes.len() / 2]) << 8)
                | (u64::from(bytes[bytes.len() - 1]) << 16);
            digest.rotate_left(7) ^ sample ^ index as u64
        })
}

fn digest_decoded(rows: &[AlignedBuffer], generation_size: usize) -> u64 {
    rows.iter().enumerate().fold(0u64, |digest, (index, row)| {
        let payload = &row.as_slice()[generation_size..];
        let sample = u64::from(payload[0])
            | (u64::from(payload[payload.len() / 2]) << 8)
            | (u64::from(payload[payload.len() - 1]) << 16);
        digest.rotate_left(7) ^ sample ^ index as u64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn scalar_and_simd_encode_match_fixture() {
        let fixture = Fixture::new(8, 1024).unwrap();
        fixture.verify_encode::<ScalarKernel>().unwrap();
        fixture.verify_encode::<SimdKernel>().unwrap();
    }

    #[test]
    fn scalar_and_simd_decode_round_trip() {
        let fixture = Fixture::new(16, 4096).unwrap();
        fixture.verify_decode::<ScalarKernel>().unwrap();
        fixture.verify_decode::<SimdKernel>().unwrap();
    }

    #[test]
    fn candidate_search_is_bounded_and_refined() {
        assert_eq!(coarse_candidates(1), vec![1]);
        assert_eq!(coarse_candidates(12), vec![1, 2, 4, 8, 12]);
        assert_eq!(refinement_candidates(1, 12), vec![1, 2, 3]);
        assert_eq!(refinement_candidates(8, 12), vec![6, 7, 8, 9, 10]);
        assert_eq!(refinement_candidates(12, 12), vec![10, 11, 12]);
    }

    #[test]
    fn best_observation_selects_highest_throughput() {
        let observations = [(1, 2.0), (2, 3.5), (4, 3.0)];
        assert_eq!(best_thread_count(&observations), 2);
    }

    #[test]
    fn default_matrix_has_no_duplicate_cases() {
        let cases: BTreeSet<_> = GENERATION_SIZES
            .iter()
            .flat_map(|&k| SYMBOL_SIZES.iter().map(move |&symbol| (k, symbol)))
            .collect();
        assert_eq!(cases.len(), GENERATION_SIZES.len() * SYMBOL_SIZES.len());
    }
}
