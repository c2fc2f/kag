//! `completions` subcommand: generate a shell completion script
//!
//! Defines [`Args`], which selects the target shell whose completion script
//! is written to standard output. The available shells come from
//! [`clap_complete::Shell`]

use clap_complete::Shell;

/// Command-line arguments for the shell completion generation
#[derive(clap::Args, Debug)]
pub struct Args {
  /// Type of shell completion to generate
  #[arg(value_enum)]
  pub shell: Shell,
}

