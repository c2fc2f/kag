//! `stats` subcommand: arguments for the benchmark scoring runner
//!
//! Defines [`Args`], the parameters that point the scorer at a directory of
//! benchmark result files and at the datasets file that holds the ground
//! truth. The scorer walks the result tree produced by the
//! [`benchmark`](super::benchmark) subcommand, compares each answer against
//! the expected option, and reports accuracy and precision metrics per setup

pub mod find;

use std::path::PathBuf;

/// Command-line arguments for the statistics computation
#[derive(clap::Args, Debug)]
#[command(subcommand_negates_reqs = true)]
pub struct Args {
  /// The specific sub-operation to perform
  #[command(subcommand)]
  pub command: Option<Command>,

  /// Path to the JSON datasets file used as the ground truth
  ///
  /// This is the same datasets file that was passed to the `benchmark`
  /// subcommand. Every question that defines an `output` is scorable; entries
  /// with no `output` (free-form) are reported as skipped.
  #[arg(short, long, required = true)]
  pub datasets: Option<PathBuf>,

  /// Root directory containing the benchmark result files
  ///
  /// This must match the `--output` directory used by the `benchmark`
  /// subcommand. The expected layout is
  /// `<results>/<dataset>/<question>/<prefix><setup>.json`.
  #[arg(short, long, default_value = ".", global = true)]
  pub results: PathBuf,

  /// Naming prefix that was prepended to the result filenames
  ///
  /// When the benchmark was run with `--prefix`, the same value must be
  /// supplied here so the setup name can be recovered from each filename.
  #[arg(long, default_value = "", global = true)]
  pub prefix: String,

  /// Output format for the computed report
  #[arg(long, value_enum, default_value_t = Format::Text, global = true)]
  pub format: Format,
}

/// The rendering format of the computed statistics report
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
  /// A human-readable, aligned table written to standard output
  Text,
  /// A machine-readable JSON document written to standard output
  Json,
}

/// List of available subcommands
#[derive(clap::Subcommand, Debug)]
#[non_exhaustive]
pub enum Command {
  /// Generate text using a specified model, with optional Knowledge-Augmented
  /// Generation (KAG)
  ///
  /// This subcommand runs either a standard text generation workflow or an
  /// augmented generation workflow (KAG/RAG) when a retriever component is
  /// provided
  Find(find::Args),
}
