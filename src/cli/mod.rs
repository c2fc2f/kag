//! Command-line interface definition for the KAG toolkit

pub mod benchmark;
pub mod completion;
pub mod component;
pub mod generation;
pub mod stats;

use clap::{Parser, Subcommand};
use clap_verbosity_flag::Verbosity;

/// A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented
/// Generation (KAG), featuring generation pipelines and evaluation benchmarks
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
  /// The specific operation to perform with the toolkit
  #[command(subcommand)]
  pub command: Command,

  /// Control the output verbosity (-v, -q)
  #[command(flatten)]
  pub verbosity: Verbosity,
}

/// List of available subcommands in the toolkit
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Command {
  /// Generate text using a specified model, with optional Knowledge-Augmented
  /// Generation (KAG)
  ///
  /// This subcommand runs either a standard text generation workflow or an
  /// augmented generation workflow (KAG/RAG) when a retriever component is
  /// provided
  Generation(generation::Args),
  /// Run performance and quality benchmarks across evaluation datasets
  ///
  /// This subcommand evaluates specified datasets against configured models
  /// and techniques. It supports parallel execution, resuming previously
  /// interrupted runs, and saving the evaluation metrics to a target output
  /// directory
  Benchmark(benchmark::Args),
  /// Compute accuracy and precision statistics from a benchmark result tree
  ///
  /// This subcommand grades the result files produced by the `benchmark`
  /// subcommand against the ground truth in the datasets file. For every
  /// question that defines an `output`, it extracts the option the model
  /// committed to, compares it against the expected answer, and reports
  /// per-setup accuracy, precision, and coverage as a table or as JSON
  Stats(stats::Args),
  /// Print shell completions and exit
  #[command(hide = true)]
  Completion(completion::Args),
}
