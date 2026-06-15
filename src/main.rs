//! A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented
//! Generation (KAG), featuring generation pipelines and evaluation benchmarks

pub mod cli;
mod config;
mod generation;
mod retrieval;
mod subcommand;

use std::{io::Write, process::ExitCode};

use clap::{CommandFactory, Parser};
use clap_complete::generate;
use log::{LevelFilter, debug};

use crate::{
  cli::Command,
  config::{Config, load_config},
};

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

fn main() -> ExitCode {
  let args = cli::Args::parse();

  if let Command::Completion(a) = args.command {
    let mut cmd = cli::Args::command();
    let name = cmd.get_name().to_string();
    generate(a.shell, &mut cmd, name, &mut std::io::stdout());
    return ExitCode::SUCCESS;
  }

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
    Command::Completion(_) => unreachable!(),
  }
}
