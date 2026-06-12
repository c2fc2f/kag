//! Provider Abstraction and Prompt Orchestration Module
//!
//! This module serves as a unified orchestration layer for multi-provider
//! Large Language  Model (LLM) workflows. It abstracts the underlying
//! initialization boilerplate for completion and embedding clients across
//! different API ecosystems (such as OpenAI and Ollama) and provides
//! structured template utilities for Context-Augmented Generation (RAG/KAG)

pub mod config;

use rig_core::{
  agent::Agent,
  client::{CompletionClient, EmbeddingsClient, Nothing},
  completion::Prompt,
  embeddings::EmbeddingModel,
  http_client,
  providers::{ollama, openai},
};
use serde_json::json;

use crate::config::Provider;

/// A unified wrapper for various LLM completion providers
pub enum AnyCompletionModel {
  /// Wraps the Ollama provider.
  Ollama(Agent<ollama::CompletionModel>),

  /// Wraps the OpenAI provider.
  OpenAI(Agent<openai::responses_api::GenericResponsesCompletionModel>),
}

impl AnyCompletionModel {
  /// Dispatches a standard prompt request to the active underlying model.
  ///
  /// # Arguments
  ///
  /// * `text` - The prompt string to be processed by the LLM.
  ///
  /// # Returns
  ///
  /// Returns a `Result` containing the generated text response on success,
  /// or an error if the underlying provider request fails.
  pub async fn prompt(
    &self,
    text: &str,
  ) -> Result<String, rig_core::completion::PromptError> {
    match self {
      Self::Ollama(model) => model.prompt(text).await,
      Self::OpenAI(model) => model.prompt(text).await,
    }
  }
}

impl Provider {
  /// Initializes and configures a completion model agent based on the current
  /// provider variant
  ///
  /// This method acts as a factory that abstracts away the boilerplate of
  /// instantiating underlying LLM clients and maps them into a unified
  /// [`AnyCompletionModel`] wrapper
  ///
  /// # Parameters
  ///
  /// * `model` - The name/ID of the target LLM model
  /// * `temperature` - Controls randomness in generation. Higher values mean
  ///   more creative but less predictable responses.
  /// * `tokens` - Optional upper constraint for the generation context:
  ///   * For `Ollama`, maps to the `num_ctx` additional parameter.
  ///   * For `OpenAI`, maps to the `max_tokens` configuration.
  ///
  /// # Errors
  ///
  /// Returns an [`http_client::Error`] if the underlying builder fails to
  /// initialize the client
  pub fn completion(
    &self,
    model: &str,
    temperature: f64,
    tokens: Option<u64>,
  ) -> Result<AnyCompletionModel, http_client::Error> {
    match self {
      Self::Ollama { url } => ollama::Client::builder()
        .api_key(Nothing)
        .base_url(url)
        .build()
        .map(|client| {
          let mut c = client.agent(model).temperature(temperature);
          if let Some(tokens) = tokens {
            c = c.additional_params(json!({
              "num_ctx": tokens
            }));
          }
          AnyCompletionModel::Ollama(c.build())
        }),
      Self::OpenAI { url, key } => openai::Client::builder()
        .api_key(key)
        .base_url(url)
        .build()
        .map(|client| {
          let mut c = client.agent(model).temperature(temperature);
          if let Some(tokens) = tokens {
            c = c.max_tokens(tokens);
          }
          AnyCompletionModel::OpenAI(c.build())
        }),
    }
  }
}

/// A unified wrapper for various LLM embedding providers
pub enum AnyEmbedderModel {
  /// Wraps the Ollama provider.
  Ollama(ollama::EmbeddingModel),

  /// Wraps the OpenAI provider.
  OpenAI(openai::GenericEmbeddingModel),
}

impl AnyEmbedderModel {
  /// Generates a vector embedding for the provided text using the active
  /// underlying model.
  ///
  /// # Arguments
  ///
  /// * `text` - The input string to be embedded.
  ///
  /// # Returns
  ///
  /// Returns a `Result` containing the
  /// [`Embedding`](rig_core::embeddings::Embedding) on success, or an
  /// [`EmbeddingError`](rig_core::embeddings::EmbeddingError) if the
  /// underlying provider request fails
  pub async fn embed_text(
    &self,
    text: &str,
  ) -> Result<
    rig_core::embeddings::Embedding,
    rig_core::embeddings::EmbeddingError,
  > {
    match self {
      Self::Ollama(model) => model.embed_text(text).await,
      Self::OpenAI(model) => model.embed_text(text).await,
    }
  }
}

impl Provider {
  /// Initializes and configures a text embedding model based on the current
  /// provider variant
  ///
  /// This method acts as a factory that abstracts away the boilerplate of
  /// instantiating underlying embedding clients and wraps them into a unified
  /// [`AnyEmbedderModel`] enum
  ///
  /// # Parameters
  ///
  /// * `model` - The name/ID of the target embedding model
  ///
  /// # Errors
  ///
  /// Returns an [`http_client::Error`] if the underlying builder fails to
  /// initialize the client
  pub fn embedder(
    &self,
    model: &str,
  ) -> Result<AnyEmbedderModel, http_client::Error> {
    match self {
      Self::Ollama { url } => ollama::Client::builder()
        .api_key(Nothing)
        .base_url(url)
        .build()
        .map(|client| AnyEmbedderModel::Ollama(client.embedding_model(model))),
      Self::OpenAI { url, key } => openai::Client::builder()
        .api_key(key)
        .base_url(url)
        .build()
        .map(|client| AnyEmbedderModel::OpenAI(client.embedding_model(model))),
    }
  }
}
