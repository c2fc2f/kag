//! This module provides structures and logic for aggregating, calculating,
//! and serializing benchmark metrics.
//!
//! It enables tracking grading outcomes (`Correct`, `Incorrect`, `Error`),
//! generation performance, and optional retrieval statistics across benchmark
//! result files. The module provides high-level reporting structure
//! (`Report`, `SetupReport`) and implementation for calculating derived
//! statistics such as accuracy, precision, and coverage, ensuring they are
//! readily available for export through JSON serialization.

use std::{collections::BTreeMap, time::Duration};

use serde::{Serialize, ser::SerializeStruct};

use crate::cli::component::ComponentName;

/// Mutable counters accumulating grading outcomes for a group of files
#[derive(Debug, Default, Clone)]
pub struct Metrics {
  /// Number of files whose prediction matched the expected option.
  pub correct: usize,
  /// Number of files whose prediction did not match the expected option.
  pub incorrect: usize,
  /// Number of files that recorded a benchmark execution error.
  pub errors: usize,
  /// Cumulative generation duration over the timed (successful) files.
  pub generation_time: Duration,
  /// Number of files that contributed a generation duration.
  pub generation_samples: usize,
  /// Performance and volume metrics for retrieval operations, present if a
  /// retriever was utilized during the benchmark run.
  pub retrieval: Option<RetrievalMetrics>,
}

/// Aggregated performance and volume metrics for retrieval operations.
#[derive(Debug, Default, Clone)]
pub struct RetrievalMetrics {
  /// Cumulative retrieval duration over the timed (successful) files.
  pub time: Duration,
  /// Total count of graph vertices retrieved across all samples.
  pub vertices: usize,
  /// Total count of graph relationships retrieved across all samples.
  pub relationships: usize,
  /// Total count of properties retrieved across all samples.
  pub properties: usize,
  /// Number of files that successfully contributed to these retrieval
  /// metrics.
  pub samples: usize,
}

impl Metrics {
  /// Records a single grading [`Outcome`] and its associated timing/retrieval
  /// data.
  pub fn record(
    &mut self,
    outcome: Outcome,
    timing: Option<&Duration>,
    retrieval: Option<&(Duration, usize, usize, usize)>,
  ) {
    match outcome {
      Outcome::Correct => self.correct += 1,
      Outcome::Incorrect => self.incorrect += 1,
      Outcome::Error => self.errors += 1,
    }

    if let Some(time) = timing {
      self.generation_time += *time;
      self.generation_samples += 1;
    }

    if let Some((time, vertices, relationships, properties)) = retrieval {
      let r = self.retrieval.get_or_insert(RetrievalMetrics::default());
      r.time += *time;
      r.vertices += vertices;
      r.relationships += relationships;
      r.properties += properties;
      r.samples += 1;
    }
  }

  /// Total number of scored files in this group.
  pub fn total(&self) -> usize {
    self.correct + self.incorrect + self.errors
  }

  /// Number of files for which the model committed to a parseable option.
  pub fn parsed(&self) -> usize {
    self.correct + self.incorrect
  }

  /// Mean generation duration per timed file, or [`None`] without samples.
  pub fn avg_generation(&self) -> Option<Duration> {
    (self.generation_samples > 0)
      .then(|| self.generation_time.div_f64(self.generation_samples as f64))
  }

  /// Fraction of all scored files that were answered correctly.
  ///
  /// Returns [`None`] when no file was scored.
  pub fn accuracy(&self) -> Option<f64> {
    let total = self.total();
    (total > 0).then(|| self.correct as f64 / total as f64)
  }

  /// Fraction of committed answers that were correct.
  ///
  /// Returns [`None`] when the model never committed to a parseable option.
  pub fn precision(&self) -> Option<f64> {
    let parsed = self.parsed();
    (parsed > 0).then(|| self.correct as f64 / parsed as f64)
  }

  /// Fraction of scored files for which a parseable option was committed.
  ///
  /// Returns [`None`] when no file was scored.
  pub fn coverage(&self) -> Option<f64> {
    let total = self.total();
    (total > 0).then(|| self.parsed() as f64 / total as f64)
  }
}

impl RetrievalMetrics {
  /// Mean retrieval duration per sampled file.
  pub fn avg_time(&self) -> Option<Duration> {
    (self.samples > 0).then(|| self.time.div_f64(self.samples as f64))
  }

  /// Mean number of vertices retrieved per sampled file.
  pub fn avg_vertices(&self) -> Option<f64> {
    (self.samples > 0).then(|| self.vertices as f64 / self.samples as f64)
  }

  /// Mean number of relationships retrieved per sampled file.
  pub fn avg_relationships(&self) -> Option<f64> {
    (self.samples > 0).then(|| self.relationships as f64 / self.samples as f64)
  }

  /// Mean number of properties retrieved per sampled file.
  pub fn avg_properties(&self) -> Option<f64> {
    (self.samples > 0).then(|| self.properties as f64 / self.samples as f64)
  }
}

impl Serialize for Metrics {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::SerializeStruct;

    let mut state = serializer.serialize_struct("Metrics", 15)?;

    state.serialize_field("total", &self.total())?;
    state.serialize_field("correct", &self.correct)?;
    state.serialize_field("incorrect", &self.incorrect)?;
    state.serialize_field("errors", &self.errors)?;
    state.serialize_field("parsed", &self.parsed())?;
    state.serialize_field("accuracy", &self.accuracy())?;
    state.serialize_field("precision", &self.precision())?;
    state.serialize_field("coverage", &self.coverage())?;

    state.serialize_field(
      "avg_generation_secs",
      &self.avg_generation().map(|d| d.as_secs_f64()),
    )?;
    state.serialize_field(
      "total_generation_secs",
      &self.generation_time.as_secs_f64(),
    )?;

    state.serialize_field("retrieval", &self.retrieval)?;

    state.end()
  }
}

impl Serialize for RetrievalMetrics {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let mut state = serializer.serialize_struct("RetrievalMetrics", 9)?;
    state.serialize_field("time_secs", &self.time.as_secs_f64())?;
    state.serialize_field("vertices", &self.vertices)?;
    state.serialize_field("relationships", &self.relationships)?;
    state.serialize_field("properties", &self.properties)?;
    state.serialize_field("samples", &self.samples)?;

    state.serialize_field(
      "avg_time_secs",
      &self.avg_time().map(|d| d.as_secs_f64()),
    )?;
    state.serialize_field("avg_vertices", &self.avg_vertices())?;
    state.serialize_field("avg_relationships", &self.avg_relationships())?;
    state.serialize_field("avg_properties", &self.avg_properties())?;

    state.end()
  }
}

/// The complete computed report ready for rendering
#[derive(Debug, Serialize)]
pub struct Report {
  /// Number of distinct datasets observed in the result tree.
  pub datasets: usize,
  /// Number of scorable ground-truth questions (those that define an output).
  pub scorable_questions: usize,
  /// Number of result files that were ignored for lack of ground truth.
  pub skipped_files: usize,
  /// The per-setup metrics, ordered by descending accuracy then by name.
  pub setups: Vec<SetupReport>,
}

/// Aggregated metrics for one benchmark setup
#[derive(Debug, Serialize)]
pub struct SetupReport {
  /// The setup identifier recovered from the result filenames.
  pub setup: ComponentName,
  /// The combined metrics across every dataset for this setup.
  pub overall: Metrics,
  /// The per-dataset breakdown, always populated alongside the total.
  pub per_dataset: BTreeMap<ComponentName, Metrics>,
}

/// The grading outcome of a single scored result file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
  /// The extracted prediction matched the expected option.
  Correct,
  /// A prediction was extracted but it did not match the expected option.
  Incorrect,
  /// The result file recorded a benchmark execution error.
  Error,
}
