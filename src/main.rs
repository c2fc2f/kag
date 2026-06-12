//! A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented
//! Generation (KAG), featuring generation pipelines and evaluation benchmarks

mod config;
mod generation;
mod retrieval;
mod subcommand;

use std::{io::Write, process::ExitCode};

use clap::{Parser, Subcommand};
use clap_verbosity_flag::Verbosity;
use log::{LevelFilter, debug};

use crate::config::{Config, load_config};

/// Unwraps a Result::Ok, or logs the error and returns ExitCode::FAILURE.
#[macro_export]
macro_rules! match_err {
  ($expr:expr, $msg:literal $(, $arg:expr)* $(,)?) => {
    match $expr {
      Ok(val) => val,
      Err(e) => {
        ::log::error!(concat!($msg, ": {:#}") $(, $arg)*, e);
        return ::std::process::ExitCode::FAILURE;
      }
    }
  };
}

/// Unwraps an Option::Some, or logs the error and returns ExitCode::FAILURE.
#[macro_export]
macro_rules! match_some {
  ($expr:expr, $msg:literal $(, $arg:expr)* $(,)?) => {
    match $expr {
      Some(val) => val,
      None => {
        ::log::error!($msg $(, $arg)*);
        return ::std::process::ExitCode::FAILURE;
      }
    }
  };
}

/// A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented
/// Generation (KAG), featuring generation pipelines and evaluation benchmarks
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
  /// The specific operation to perform with the toolkit
  #[command(subcommand)]
  command: Command,

  /// Path to the config file
  #[arg(short, long, global = true, default_value = "config.toml")]
  config: std::path::PathBuf,

  /// Control the output verbosity (-v, -q)
  #[command(flatten)]
  verbosity: Verbosity,
}

/// List of available subcommands in the toolkit
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Command {
  /// Generate text using a specified model, with optional Knowledge-Augmented
  /// Generation (KAG).
  ///
  /// This subcommand runs either a standard text generation workflow or an
  /// augmented generation workflow (KAG/RAG) when a retriever component is
  /// provided.
  Generation(subcommand::generation::Args),
  /// Run performance and quality benchmarks across evaluation datasets.
  ///
  /// This subcommand evaluates specified datasets against configured models
  /// and techniques. It supports parallel execution, resuming previously
  /// interrupted runs, and saving the evaluation metrics to a target output
  /// directory.
  Benchmark(subcommand::benchmark::Args),
}

fn main() -> ExitCode {
  let args = Args::parse();

  let log_level = args.verbosity.log_level_filter();
  let mut builder = env_logger::Builder::new();
  builder.filter_level(log_level);
  builder.format(move |buf, record| {
    if log_level > LevelFilter::Error {
      writeln!(
        buf,
        "[{timestamp} {level} {target}] {message}",
        timestamp = buf.timestamp(),
        level = record.level(),
        target = record.target(),
        message = record.args()
      )
    } else {
      writeln!(
        buf,
        "{level}: {message}",
        level = record.level().to_string().to_lowercase(),
        message = record.args()
      )
    }
  });
  builder.init();

  let config: Config = match_err!(
    load_config(args.config),
    "Unable to load the configuration file"
  );

  debug!("Final configuration: {config:?}");

  match args.command {
    Command::Generation(args) => subcommand::generation::run(args, config),
    Command::Benchmark(args) => subcommand::benchmark::run(args, config),
  }
}
