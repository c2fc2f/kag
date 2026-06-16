//! `generation` subcommand: arguments for standard and knowledge-augmented
//! text generation
//!
//! Defines the CLI [`Args`] alongside the reusable [`Generation`] and
//! [`AugmentedGeneration`] configuration groups (also
//! `Serialize`/`Deserialize`, so they double as config-file entries).
//! Retrieval is opt-in: when an [`AugmentedGeneration`] is supplied, the
//! retrieved context is injected into the system prompt through the
//! `{{RETRIEVAL}}` placeholder, while the user prompt fills `{{INPUT}}`

use std::path::PathBuf;

use clap_stdin::MaybeStdin;
use serde::{Deserialize, Serialize};

use super::component::ComponentName;

/// Command-line arguments for the generation run
#[derive(clap::Args, Debug)]
pub struct Args {
  /// The underlying text generation and model configuration
  ///
  /// This includes provider settings, hyperparameters, and prompt templates
  #[command(flatten)]
  pub generation: Generation,

  /// The system prompt template to structure the context and question
  ///
  /// Any instance of `{{INPUT}}` will be replaced by the user's input
  /// prompt.
  /// If Knowledge-Augmented Generation (KAG/RAG) is enabled (by providing a
  /// retriever), any instance of `{{RETRIEVAL}}` will be replaced by the
  /// retrieved context.
  #[arg(short, long, value_name = "FILE")]
  pub system_prompt: Option<PathBuf>,

  /// The user's input prompt
  ///
  /// You can provide the prompt directly as a standard argument. To pipe or
  /// read the prompt from standard input, you must explicitly pass `-` as the
  /// argument.
  ///
  /// This value will be injected into the `{{INPUT}}` placeholder within
  /// the system prompt.
  pub prompt: MaybeStdin<String>,

  /// Path to the config file
  #[arg(short, long, default_value = "config.toml")]
  pub config: std::path::PathBuf,
}

/// Full configuration for a single generation.
///
/// Defines the generation pipeline and all model parameters required to
/// reproduce the run. Knowledge-Augmented Generation is opt-in via
/// [`augmented_generation`](Generation::augmented_generation)
#[derive(Debug, Clone, Deserialize, Serialize, clap::Args)]
pub struct Generation {
  /// Optional Knowledge-Augmented Generation (KAG/RAG) settings.
  ///
  /// When [`None`], the entry runs in standard generation mode without
  /// retrieval. When provided, retrieved context is injected into the
  /// system prompt via `{{RETRIEVAL}}`
  #[command(flatten)]
  pub augmented_generation: Option<AugmentedGeneration>,
  /// The name of the text generation provider component to use
  #[arg(short, long)]
  pub provider: ComponentName,
  /// The specific model identifier to be used by the provider
  #[arg(short, long)]
  pub model: String,
  /// Sampling temperature controlling output creativity
  ///
  /// Higher values produce more varied and creative responses; lower values
  /// make the output more deterministic
  #[arg(short, long)]
  pub temperature: f64,
  /// Maximum number of tokens for the completion
  ///
  /// Higher values allow longer, more detailed outputs. When [`None`], the
  /// provider's default is used.
  ///
  /// On Ollama, sets `num_ctx` (context window size)
  #[arg(short = 'n', long)]
  pub tokens: Option<u64>,
}

/// Knowledge-Augmented Generation (KAG/RAG) settings for a generation.
///
/// Pairs a retriever component with a database to inject retrieved context
/// into the generation pipeline
#[derive(Debug, Clone, Deserialize, Serialize, clap::Args)]
pub struct AugmentedGeneration {
  /// The name of the retriever component to use from the configuration
  #[arg(short, long)]
  pub retriever: ComponentName,
  /// The name of the database component to use for retrieval
  #[arg(short, long)]
  pub database: ComponentName,
}
