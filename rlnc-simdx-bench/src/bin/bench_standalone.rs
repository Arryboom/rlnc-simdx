//! Standalone RLNC kernel throughput benchmark.
//!
//! Prints **MB/s and GB/s**. Compile once, copy the binary to any matching
//! OS/arch machine — no Rust toolchain required on the target.
//!
//! ```text
//! cargo build --release -p rlnc-simdx-bench --bin bench_standalone
//! ```
//!
//! Flags: `--quick`  shorter timing · `--csv`  machine-readable output
//!
//! Note: per-tier SIMD kernels are crate-private; this binary measures
//! **scalar** vs **safe dispatch** (`kernel::axpy` / `kernel::scale`).

use std::hint::black_box;
use std::time::{Duration, Instant};

use rlnc_simdx::AlignedBuffer;

const COEFF: u8 = 0x53;

const DEFAULT_SIZES: &[usize] = &[64, 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024];

fn measure_ns_per_iter(target: Duration, mut f: impl FnMut()) -> f64 {
    for _ in 0..8 {
        f();
    }

    let mut batch = 1usize;
    let t0 = Instant::now();
    f();
    let one = t0.elapsed();
    if one < Duration::from_micros(50) {
        let per = one.as_secs_f64().max(1e-12);
        batch = ((0.002 / per) as usize).clamp(1, 1_000_000);
    }

    let mut samples: Vec<f64> = Vec::with_capacity(64);
    let deadline = Instant::now() + target;
    while Instant::now() < deadline {
        let t0 = Instant::now();
        for _ in 0..batch {
            f();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        samples.push(elapsed / batch as f64 * 1e9);
        if samples.len() >= 200 {
            break;
        }
    }

    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn fmt_size(n: usize) -> String {
    if n >= 1024 * 1024 {
        format!("{} MiB", n / (1024 * 1024))
    } else if n >= 1024 {
        format!("{} KiB", n / 1024)
    } else {
        format!("{n} B")
    }
}

fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1e6)
    } else if ns >= 1_000.0 {
        format!("{:.2} µs", ns / 1e3)
    } else {
        format!("{:.1} ns", ns)
    }
}

fn fmt_throughput(bytes: u64, ns: f64) -> String {
    if ns <= 0.0 {
        return "—".to_string();
    }
    let bytes_per_s = (bytes as f64) / (ns * 1e-9);
    let gb_s = bytes_per_s / 1e9;
    let mb_s = bytes_per_s / 1e6;
    if gb_s >= 1.0 {
        format!("{gb_s:6.2} GB/s")
    } else {
        format!("{mb_s:6.1} MB/s")
    }
}

fn thr_gb_s(bytes: u64, ns: f64) -> f64 {
    if ns <= 0.0 {
        0.0
    } else {
        (bytes as f64) / (ns * 1e-9) / 1e9
    }
}

type KernelFn = fn(u8, &[u8], &mut [u8]);

struct Kernel {
    name: &'static str,
    axpy: KernelFn,
    scale: KernelFn,
}

fn available_kernels() -> Vec<Kernel> {
    vec![
        Kernel {
            name: "scalar",
            axpy: |c, x, y| rlnc_simdx::kernel::scalar::axpy(c, x, y),
            scale: |c, x, y| rlnc_simdx::kernel::scalar::scale(c, x, y),
        },
        Kernel {
            name: "dispatch",
            axpy: |c, x, y| rlnc_simdx::kernel::axpy(c, x, y),
            scale: |c, x, y| rlnc_simdx::kernel::scale(c, x, y),
        },
    ]
}

fn print_table_header(op: &str) {
    println!();
    println!("=== {op}  (aligned 64-byte buffers) ===");
    println!(
        "{:<10}  {:<16}  {:>12}  {:>12}  {:>10}",
        "Size", "Kernel", "Time/iter", "Throughput", "vs scalar"
    );
    println!("{}", "-".repeat(70));
}

fn print_row(size: usize, name: &str, ns: f64, bytes: u64, scalar_ns: Option<f64>) {
    let speedup = match scalar_ns {
        Some(s) if s > 0.0 && name != "scalar" => format!("{:.1}×", s / ns),
        _ if name == "scalar" => "1.0×".to_string(),
        _ => "—".to_string(),
    };
    println!(
        "{:<10}  {:<16}  {:>12}  {:>12}  {:>10}",
        fmt_size(size),
        name,
        fmt_ns(ns),
        fmt_throughput(bytes, ns),
        speedup
    );
}

fn print_csv_header() {
    println!("op,size_bytes,kernel,ns_per_iter,throughput_gb_s,speedup_vs_scalar");
}

fn print_csv_row(op: &str, size: usize, name: &str, ns: f64, bytes: u64, scalar_ns: Option<f64>) {
    let thr = thr_gb_s(bytes, ns);
    let speedup = match scalar_ns {
        Some(s) if s > 0.0 => s / ns,
        _ => 1.0,
    };
    println!("{op},{size},{name},{ns:.3},{thr:.4},{speedup:.2}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let csv = args.iter().any(|a| a == "--csv");

    let target = if quick {
        Duration::from_millis(250)
    } else {
        Duration::from_millis(1500)
    };

    let kernels = available_kernels();
    let active = rlnc_simdx::active_kernel();

    if !csv {
        println!("RLNC SIMD standalone throughput benchmark");
        println!("─────────────────────────────────────────");
        println!("Active dispatch kernel : {active}");
        println!(
            "Timing budget / cell   : {} ms{}",
            target.as_millis(),
            if quick { " (--quick)" } else { "" }
        );
        println!("Buffers                : AlignedBuffer (64-byte aligned)");
        println!("Throughput units       : SI  (1 GB/s = 1e9 bytes/s)");
        println!("Kernels measured       : scalar vs safe dispatch only");
        println!("  (per-tier SIMD is crate-private — see kernel::axpy)");
        println!();
    } else {
        print_csv_header();
    }

    if !csv {
        print_table_header("AXPY  y[i] ^= c * x[i]");
    }

    for &size in DEFAULT_SIZES {
        let x = AlignedBuffer::from_slice(&(0u8..).take(size).collect::<Vec<_>>());
        let y_seed = AlignedBuffer::from_slice(
            &(0..size)
                .map(|i| (i as u8).wrapping_mul(3))
                .collect::<Vec<_>>(),
        );

        let mut scalar_ns = None;

        for ker in &kernels {
            let mut y_local = AlignedBuffer::from_slice(y_seed.as_slice());
            let ns = measure_ns_per_iter(target, || {
                (ker.axpy)(
                    COEFF,
                    black_box(x.as_slice()),
                    black_box(y_local.as_mut_slice()),
                );
            });
            if ker.name == "scalar" {
                scalar_ns = Some(ns);
            }
            if csv {
                print_csv_row("axpy", size, ker.name, ns, size as u64, scalar_ns);
            } else {
                print_row(size, ker.name, ns, size as u64, scalar_ns);
            }
        }
        if !csv {
            println!();
        }
    }

    if !csv {
        print_table_header("SCALE  y[i] = c * x[i]");
    }

    for &size in DEFAULT_SIZES {
        let x = AlignedBuffer::from_slice(&(0u8..).take(size).collect::<Vec<_>>());
        let mut y = AlignedBuffer::zeroed(size);

        let mut scalar_ns = None;

        for ker in &kernels {
            let ns = measure_ns_per_iter(target, || {
                (ker.scale)(COEFF, black_box(x.as_slice()), black_box(y.as_mut_slice()));
            });
            if ker.name == "scalar" {
                scalar_ns = Some(ns);
            }
            if csv {
                print_csv_row("scale", size, ker.name, ns, size as u64, scalar_ns);
            } else {
                print_row(size, ker.name, ns, size as u64, scalar_ns);
            }
        }
        if !csv {
            println!();
        }
    }

    if !csv {
        println!("=== ENCODE  k=16 symbols, random RLNC ===");
        println!(
            "{:<10}  {:>12}  {:>12}  {:>14}",
            "Symbol", "Time/pkt", "Payload thr", "Notes"
        );
        println!("{}", "-".repeat(55));
    }

    for &n in &[1024usize, 4096, 16 * 1024, 64 * 1024] {
        let k = 16usize;
        let source_owned: Vec<AlignedBuffer> = (0..k)
            .map(|i| {
                AlignedBuffer::from_slice(
                    &(0..n)
                        .map(|j| (i as u8).wrapping_mul(17).wrapping_add(j as u8))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let refs: Vec<&[u8]> = source_owned.iter().map(|b| b.as_slice()).collect();

        let enc = rlnc_simdx::Encoder::new(k, n).expect("encoder");
        let mut rng = rlnc_simdx::SimpleRng::new(0xC0FFEE);

        for _ in 0..4 {
            let _ = enc.encode_random(&refs, &mut rng);
        }

        let ns = measure_ns_per_iter(target, || {
            let pkt = enc
                .encode_random(black_box(&refs), black_box(&mut rng))
                .unwrap();
            black_box(pkt);
        });

        if csv {
            print_csv_row("encode", n, "random_k16", ns, n as u64, None);
        } else {
            println!(
                "{:<10}  {:>12}  {:>12}  {:>14}",
                fmt_size(n),
                fmt_ns(ns),
                fmt_throughput(n as u64, ns),
                format!("k={k}, kernel={active}")
            );
        }
    }

    if !csv {
        println!();
        println!("Done.");
        println!("  cargo build --release -p rlnc-simdx-bench --bin bench_standalone");
        println!("  flags: --quick | --csv");
    }
}
