//! Command implementation for the retrieval and processing workflow.
//! Command implementation for text generation, supporting optional
//! Knowledge-Augmented Generation (KAG)
//!
//! This module handles the execution of text generation tasks based on user
//! input. It supports standard text generation using a specified provider and
//! model. Additionally, if a `retriever` is provided, the generation process
//! is augmented with external context (KAG/RAG workflow) before querying the
//! model

use std::process::ExitCode;

use clap_stdin::MaybeStdin;
use log::error;

use crate::config::{ComponentName, Config};

/// Command-line arguments for the generation run
#[derive(clap::Args, Debug)]
pub struct Args {
    /// The optional name of the retriever component to use from the
    /// configuration. If provided it enables Knowledge-Augmented Generation
    #[arg(short, long)]
    retriever: Option<ComponentName>,

    /// The optional name of the database component to use for retrieval.
    #[arg(short, long, requires = "retriever")]
    database: Option<ComponentName>,

    /// The name of the text generation provider component to use
    #[arg(short, long)]
    provider: ComponentName,

    /// The specific model identifier to be used by the provider
    #[arg(short, long)]
    model: String,

    /// The user's input prompt
    ///
    /// You can provide the prompt directly as a standard argument. To pipe or
    /// read the prompt from standard input, you must explicitly pass `-` as
    /// the argument
    ///
    /// If Knowledge-Augmented Generation (KAG/RAG) is enabled (by providing
    /// a retriever), any instance of `{{RETRIEVAL}}` in the prompt will be
    /// replaced by the retrieved context
    prompt: MaybeStdin<String>,
}

/// Executes the generation command with the provided arguments and
/// configuration.
///
/// This function orchestrates the workflow by resolving the requested
/// components (`provider`, and optionally `retriever` and `database`) from
/// the application configuration. If any of the requested components are
/// missing from the configuration, an error is logged and the execution is
/// aborted.
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
/// Returns `ExitCode::SUCCESS` on successful generation, or
/// `ExitCode::FAILURE` if a required component cannot be resolved.
pub fn run(args: Args, config: Config) -> ExitCode {
    let retriever = match &args.retriever {
        Some(name) => match config.retrievers.get(name) {
            Some(r) => Some(r),
            None => {
                error!(
                    "The requested retriever '{}' is missing from the configuration.",
                    name
                );
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let database = match &args.database {
        Some(name) => match config.databases.get(name) {
            Some(r) => Some(r),
            None => {
                error!(
                    "The requested database '{}' is missing from the configuration.",
                    name
                );
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let Some(provider) = config.providers.get(&args.provider) else {
        error!(
            "The requested provider '{}' is missing from the configuration.",
            args.provider
        );
        return ExitCode::FAILURE;
    };

    unimplemented!();
}
