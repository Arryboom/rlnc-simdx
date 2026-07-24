use rlnc_simdx_mt_bench::{CaseResult, Config, Runner, GENERATION_SIZES, SYMBOL_SIZES};
use std::{
    env,
    io::{self, Write},
    process::{self, ExitCode},
    time::Instant,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rlnc-simdx-mt-bench: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let started = Instant::now();
    let (quick, max_threads) = parse_args()?;
    let config = Config::for_machine(quick, max_threads);
    let runner = Runner::new(config);

    print_report_header(&config, runner.max_threads())?;

    let mut encode_results = Vec::with_capacity(GENERATION_SIZES.len() * SYMBOL_SIZES.len());
    print_table_header("Encode")?;
    for &generation_size in &GENERATION_SIZES {
        for &symbol_size in &SYMBOL_SIZES {
            let result = runner.benchmark_encode(generation_size, symbol_size)?;
            print_result_row(&result)?;
            encode_results.push(result);
        }
    }
    print_table_summary(&encode_results)?;

    let mut decode_results = Vec::with_capacity(GENERATION_SIZES.len() * SYMBOL_SIZES.len());
    print_table_header("Decode")?;
    for &generation_size in &GENERATION_SIZES {
        for &symbol_size in &SYMBOL_SIZES {
            let result = runner.benchmark_decode(generation_size, symbol_size)?;
            print_result_row(&result)?;
            decode_results.push(result);
        }
    }
    print_table_summary(&decode_results)?;

    println!();
    println!("Completed in {:.1}s", started.elapsed().as_secs_f64());
    flush_stdout()
}

fn parse_args() -> Result<(bool, Option<usize>), String> {
    let mut quick = false;
    let mut max_threads = None;
    let mut args = env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--quick" => quick = true,
            "--max-threads" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-threads requires a positive integer".to_owned())?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --max-threads value: {value}"))?;
                if parsed == 0 {
                    return Err("--max-threads must be at least 1".to_owned());
                }
                max_threads = Some(parsed);
            }
            "-h" | "--help" => {
                println!("Usage: rlnc-simdx-mt-bench [--quick] [--max-threads N]");
                process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    Ok((quick, max_threads))
}

fn print_report_header(config: &Config, max_threads: usize) -> Result<(), String> {
    println!("+================================================================================+");
    println!(
        "| {:^78} |",
        format!(
            "rlnc-simdx-mt-bench | Multithreaded RLNC | v{}",
            env!("CARGO_PKG_VERSION")
        )
    );
    println!("+--------------------------------------------------------------------------------+");
    print_metadata("Host OS", env::consts::OS);
    print_metadata("Host arch", env::consts::ARCH);
    print_metadata(
        "Logical CPUs",
        &std::thread::available_parallelism()
            .map_or(1, usize::from)
            .to_string(),
    );
    print_metadata("Active kernel", rlnc_simdx::active_kernel());
    print_metadata(
        "Rayon workers",
        &format!("1..{max_threads} (autotuned per backend and operation)"),
    );
    print_metadata("Mode", if config.quick { "quick" } else { "full" });
    print_metadata("Backends", "scalar vs safe runtime SIMD dispatch");
    print_metadata(
        "Workloads",
        "k = 8, 16, 32; symbol = 64 B, 1, 4, 16, 64 KiB",
    );
    print_metadata(
        "Throughput",
        "effective source bytes / elapsed time, binary GiB/s",
    );
    print_metadata(
        "Timing scope",
        "reusable fixtures, worker state, and thread pools excluded",
    );
    println!("+================================================================================+");
    flush_stdout()
}

fn print_metadata(label: &str, value: &str) {
    println!("{}", format_metadata_line(label, value));
}

fn format_metadata_line(label: &str, value: &str) -> String {
    format!("| {label:<14}: {value:<62} |")
}

fn print_table_header(title: &str) -> Result<(), String> {
    println!();
    println!("{title}");
    println!("+------+----------+--------------+----------+--------------+----------+----------+");
    println!(
        "| {:>4} | {:<8} | {:>12} | {:>8} | {:>12} | {:>8} | {:>8} |",
        "K", "Symbol", "Scalar GiB/s", "Threads", "SIMD GiB/s", "Threads", "Speedup"
    );
    println!("+------+----------+--------------+----------+--------------+----------+----------+");
    flush_stdout()
}

fn print_result_row(result: &CaseResult) -> Result<(), String> {
    println!("{}", format_result_row(result));
    flush_stdout()
}

fn format_result_row(result: &CaseResult) -> String {
    format!(
        "| {:>4} | {:<8} | {:>12.3} | {:>8} | {:>12.3} | {:>8} | {:>7.2}x |",
        result.generation_size,
        format_size(result.symbol_size),
        result.scalar.gib_per_second,
        result.scalar.threads,
        result.simd.gib_per_second,
        result.simd.threads,
        result.speedup(),
    )
}

fn print_table_summary(results: &[CaseResult]) -> Result<(), String> {
    let peak_scalar = results
        .iter()
        .max_by(|left, right| {
            left.scalar
                .gib_per_second
                .total_cmp(&right.scalar.gib_per_second)
        })
        .ok_or_else(|| "cannot summarize an empty result table".to_owned())?;
    let peak_simd = results
        .iter()
        .max_by(|left, right| {
            left.simd
                .gib_per_second
                .total_cmp(&right.simd.gib_per_second)
        })
        .ok_or_else(|| "cannot summarize an empty result table".to_owned())?;
    let best_speedup = results
        .iter()
        .max_by(|left, right| left.speedup().total_cmp(&right.speedup()))
        .ok_or_else(|| "cannot summarize an empty result table".to_owned())?;

    println!("+------+----------+--------------+----------+--------------+----------+----------+");
    println!("Summary");
    println!(
        "  Peak scalar : {:>8.3} GiB/s  (k={}, symbol={}, {} threads)",
        peak_scalar.scalar.gib_per_second,
        peak_scalar.generation_size,
        format_size(peak_scalar.symbol_size),
        peak_scalar.scalar.threads
    );
    println!(
        "  Peak SIMD   : {:>8.3} GiB/s  (k={}, symbol={}, {} threads)",
        peak_simd.simd.gib_per_second,
        peak_simd.generation_size,
        format_size(peak_simd.symbol_size),
        peak_simd.simd.threads
    );
    println!(
        "  Best speedup: {:>8.2}x       (k={}, symbol={})",
        best_speedup.speedup(),
        best_speedup.generation_size,
        format_size(best_speedup.symbol_size)
    );
    flush_stdout()
}

fn flush_stdout() -> Result<(), String> {
    io::stdout()
        .flush()
        .map_err(|error| format!("unable to flush benchmark output: {error}"))
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else {
        format!("{} KiB", bytes / 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlnc_simdx_mt_bench::BackendResult;

    #[test]
    fn size_format_is_compact() {
        assert_eq!(format_size(64), "64 B");
        assert_eq!(format_size(1024), "1 KiB");
        assert_eq!(format_size(65_536), "64 KiB");
    }

    #[test]
    fn metadata_and_result_rows_are_portable_ascii() {
        let metadata = format_metadata_line("Active kernel", "gfni+avx2 (tier2)");
        assert!(metadata.is_ascii());
        assert_eq!(metadata.len(), 82);
        assert!(metadata.contains("Active kernel : gfni+avx2 (tier2)"));

        let result = CaseResult {
            generation_size: 8,
            symbol_size: 4096,
            scalar: BackendResult {
                gib_per_second: 0.375,
                threads: 4,
            },
            simd: BackendResult {
                gib_per_second: 24.109,
                threads: 3,
            },
        };
        let row = format_result_row(&result);
        assert!(row.is_ascii());
        assert_eq!(row.len(), 82);
        assert!(row.contains("|    8 | 4 KiB"));
        assert!(row.contains("|        3 |"));
        assert!(row.ends_with("64.29x |"));
    }
}
