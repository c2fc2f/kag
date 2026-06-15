//! Dataset definitions for component evaluation
//!
//! This module provides the data structures used to describe evaluation
//! datasets, where each [`ComponentName`] is associated with a
//! [`DatasetEntry`] containing an input and its expected [`Output`]

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::cli::component::ComponentName;

/// A collection of evaluation entries, grouped by nested component names.
///
/// Wraps a nested [`BTreeMap`] mapping an outer [`ComponentName`] to an inner
/// [`BTreeMap`], which in turn maps an inner [`ComponentName`] to its
/// corresponding [`DatasetEntry`]. Deserialized directly from a nested
/// map-shaped configuration or data file.
#[derive(Deserialize, Debug)]
pub struct Datasets(
  pub(crate) BTreeMap<ComponentName, BTreeMap<ComponentName, DatasetEntry>>,
);

impl AsRef<BTreeMap<ComponentName, BTreeMap<ComponentName, DatasetEntry>>>
  for Datasets
{
  fn as_ref(
    &self,
  ) -> &BTreeMap<ComponentName, BTreeMap<ComponentName, DatasetEntry>> {
    &self.0
  }
}

/// A single evaluation entry for a component.
///
/// Pairs a raw input string with an optional expected [`Output`]. When
/// `output` is [`None`], the component output is considered free-form and
/// unconstrained.
#[derive(Deserialize, Debug)]
pub struct DatasetEntry {
  /// The raw input string fed to the component under evaluation.
  pub input: String,
  /// The expected output format, or [`None`] if the output is free-form.
  pub output: Option<Output>,
}

/// The expected output format for a dataset entry.
///
/// Additional variants may be added in the future; match exhaustively with a
/// wildcard arm.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", content = "options")]
#[non_exhaustive]
pub enum Output {
  /// Multiple-choice output: a map of option identifiers to their label or
  /// description.
  ///
  /// Keys are option identifiers (e.g. `"A"`, `"B"`) and values are the
  /// corresponding option texts.
  #[serde(rename = "MCQ")]
  Mcq(BTreeMap<String, String>),
}
