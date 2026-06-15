//! Benchmark configuration definitions
//!
//! This module provides the data structures used to describe benchmark runs,
//! where each [`ComponentName`] is associated with a [`BenchmarkEntry`] that
//! fully specifies the generation pipeline and model settings to evaluate

use std::collections::BTreeSet;

use hashbrown::HashMap;
use serde::Deserialize;

use crate::cli::{component::ComponentName, generation::Generation};

/// A collection of benchmark entries, keyed by component name.
///
/// Wraps a [`HashMap`] mapping each [`ComponentName`] to its corresponding
/// [`BenchmarkEntry`]. Deserialized directly from a map-shaped configuration
/// file
#[derive(Debug, Deserialize)]
pub struct Benchmark(HashMap<ComponentName, BenchmarkEntry>);

impl AsRef<HashMap<ComponentName, BenchmarkEntry>> for Benchmark {
  fn as_ref(&self) -> &HashMap<ComponentName, BenchmarkEntry> {
    &self.0
  }
}

/// Full configuration for a single benchmark run.
///
/// Defines the generation pipeline and all model parameters required to
/// reproduce the run. Knowledge-Augmented Generation is opt-in via
/// [`augmented_generation`](BenchmarkEntry::augmented_generation)
#[derive(Debug, Deserialize)]
pub struct BenchmarkEntry {
  /// The underlying text generation and model configuration
  ///
  /// This includes provider settings, hyperparameters, and prompt templates
  #[serde(flatten)]
  pub config: Generation,
  /// System prompt template used to structure context and the user question
  ///
  /// - `{{INPUT}}` is replaced by the user's input prompt
  /// - `{{RETRIEVAL}}` is replaced by the retrieved context
  /// - `{{CHOICE}}` is replaced by the available answer options
  pub system_prompt: String,
  /// The subset of datasets to evaluate against
  ///
  /// When [`None`], all available datasets are included in the benchmark run
  /// When provided, only the named [`ComponentName`] entries are evaluated
  pub datasets: Option<BTreeSet<ComponentName>>,
}
