//! `stats find` subcommand: arguments for the regex search over benchmark
//! results
//!
//! Defines [`Args`], the single regular expression the model's response must
//! validate for a result file to be kept by the
//! [`find`](crate::subcommand::stats::subcommand::find) subcommand.

use regex::Regex;

/// Command-line arguments for the regex search over benchmark results
#[derive(clap::Args, Debug)]
pub struct Args {
  /// The regular expression the model's response must validate to be kept
  ///
  /// The pattern is matched against the trimmed model response extracted
  /// from each result file. Result files that recorded a benchmark
  /// execution error are never kept, since they hold no response to match
  /// against.
  pub regex: Regex,
}
