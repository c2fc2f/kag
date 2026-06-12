//! Command implementation for the retrieval and processing workflow. Command
//! implementation for text generation, supporting optional
//! Knowledge-Augmented Generation (KAG)
//!
//! This module handles the execution of text generation tasks based on user
//! input. It supports standard text generation using a specified provider and
//! model. Additionally, if a `retriever` is provided, the generation process
//! is augmented with external context (KAG/RAG workflow) before querying the
//! model

use std::{fs, path::PathBuf, process::ExitCode};

use clap_stdin::MaybeStdin;
use log::debug;
use minijinja::Environment;

use crate::{config::Config, generation::config::Generation, match_err};

/// Command-line arguments for the generation run
#[derive(clap::Args, Debug)]
pub struct Args {
  /// The underlying text generation and model configuration
  ///
  /// This includes provider settings, hyperparameters, and prompt templates
  #[command(flatten)]
  generation: Generation,

  /// The system prompt template to structure the context and question
  ///
  /// Any instance of `{{INPUT}}` will be replaced by the user's input
  /// prompt.
  /// If Knowledge-Augmented Generation (KAG/RAG) is enabled (by providing a
  /// retriever), any instance of `{{RETRIEVAL}}` will be replaced by the
  /// retrieved context.
  #[arg(short, long, value_name = "FILE")]
  system_prompt: Option<PathBuf>,

  /// The user's input prompt
  ///
  /// You can provide the prompt directly as a standard argument. To pipe or
  /// read the prompt from standard input, you must explicitly pass `-` as the
  /// argument.
  ///
  /// This value will be injected into the `{{INPUT}}` placeholder within
  /// the system prompt.
  prompt: MaybeStdin<String>,
}

/// Executes the generation command with the provided arguments and
/// configuration.
///
/// # Arguments
///
/// * `args` - The parsed command-line arguments containing the user's prompt
///   and component targets.
/// * `config` - The application configuration containing the available
///   component definitions.
///
/// # Returns
///
/// Returns `ExitCode::SUCCESS` on successful generation, `ExitCode::FAILURE`
/// otherwise.
pub fn run(args: Args, config: Config) -> ExitCode {
  let rt = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
    .expect("Failed building the Runtime");

  let _ = rustls::crypto::ring::default_provider().install_default();

  let system_prompt = match_err!(
    args.system_prompt.map(fs::read_to_string).transpose(),
    "Failed to read system prompt file"
  );

  let response = match_err!(
    rt.block_on(args.generation.generate(
      &config,
      system_prompt.as_deref(),
      args.prompt.into_inner(),
      Environment::new(),
    )),
    "Failed to generate response from the model"
  );

  println!("{}", response.result);
  debug!(
    "Generation phase completed successfully. Stats: {:?}",
    response.stats
  );

  ExitCode::SUCCESS
}
