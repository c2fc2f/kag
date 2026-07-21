//! `benchmark` subcommand: arguments for the evaluation benchmark runner
//!
//! Defines [`Args`], the parameters that select the evaluation datasets and
//! the techniques/models to compare, control the degree of parallelism,
//! decide where result files are written, and whether an interrupted run is
//! resumed rather than overwritten

use std::{num::NonZero, path::PathBuf};

use crate::cli::component::ComponentName;

/// Command-line arguments for the benchmark execution
#[derive(clap::Args, Debug)]
pub struct Args {
  /// Path to the JSON file containing the collection of evaluation datasets
  #[arg(short, long)]
  pub datasets: PathBuf,

  /// Path to the configuration file defining the different techniques and
  /// models to benchmark
  #[arg(short, long)]
  pub benchmark: PathBuf,

  /// Number of parallel tasks to use for the benchmark execution
  #[arg(short, long, default_value = "1")]
  pub parallel: NonZero<usize>,

  /// Flag to resume a previously interrupted benchmark run, preserving
  /// existing result files instead of overwriting them
  #[arg(long, action)]
  pub r#continue: bool,

  /// Directory path where the generated benchmark result files will be saved
  #[arg(short, long, default_value = ".")]
  pub output: PathBuf,

  /// Optional naming prefix to append to the generated JSON result filenames
  #[arg(long, default_value = "")]
  pub prefix: String,

  /// Optional list of techniques that should not be performed (separated by
  /// commas)
  #[arg(long, value_delimiter = ',')]
  pub skip: Vec<ComponentName>,

  /// Path to the config file
  #[arg(short, long, default_value = "config.toml")]
  pub config: std::path::PathBuf,
}
