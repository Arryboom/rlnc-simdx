use rlnc_simdx_mt_benchmark::{CaseResult, Config, Runner, GENERATION_SIZES, SYMBOL_SIZES};
use std::{
    env,
    process::{self, ExitCode},
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mt_benchmark: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let (quick, max_threads) = parse_args()?;
    let config = Config::for_machine(quick, max_threads);
    let runner = Runner::new(config);

    let mut encode_results = Vec::with_capacity(GENERATION_SIZES.len() * SYMBOL_SIZES.len());
    let mut decode_results = Vec::with_capacity(GENERATION_SIZES.len() * SYMBOL_SIZES.len());

    for &generation_size in &GENERATION_SIZES {
        for &symbol_size in &SYMBOL_SIZES {
            encode_results.push(runner.benchmark_encode(generation_size, symbol_size)?);
        }
    }
    print_table("Encode", &encode_results);

    for &generation_size in &GENERATION_SIZES {
        for &symbol_size in &SYMBOL_SIZES {
            decode_results.push(runner.benchmark_decode(generation_size, symbol_size)?);
        }
    }
    print_table("Decode", &decode_results);

    Ok(())
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
                println!("Usage: mt_benchmark [--quick] [--max-threads N]");
                process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    Ok((quick, max_threads))
}

fn print_table(title: &str, results: &[CaseResult]) {
    println!("\n{title}");
    println!(
        "{:<4} {:>8} {:>14} {:>9} {:>14} {:>7} {:>9}",
        "K", "Symbol", "Scalar GiB/s", "Scalar T", "SIMD GiB/s", "SIMD T", "Speedup"
    );
    println!("{}", "-".repeat(75));

    for result in results {
        println!(
            "{:<4} {:>8} {:>14.3} {:>9} {:>14.3} {:>7} {:>8.2}x",
            result.generation_size,
            format_size(result.symbol_size),
            result.scalar.gib_per_second,
            result.scalar.threads,
            result.simd.gib_per_second,
            result.simd.threads,
            result.speedup(),
        );
    }
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

    #[test]
    fn size_format_is_compact() {
        assert_eq!(format_size(64), "64 B");
        assert_eq!(format_size(1024), "1 KiB");
        assert_eq!(format_size(65_536), "64 KiB");
    }
}
