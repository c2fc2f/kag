//! Command implementation for the retrieval and processing workflow. Command
//! implementation for text generation, supporting optional
//! Knowledge-Augmented Generation (KAG)
//!
//! This module handles the execution of text generation tasks based on user
//! input. It supports standard text generation using a specified provider and
//! model. Additionally, if a `retriever` is provided, the generation process
//! is augmented with external context (KAG/RAG workflow) before querying the
//! model

use std::{fs, process::ExitCode};

use log::debug;
use minijinja::Environment;

use crate::{cli::generation::Args, config::Config, match_err};

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
