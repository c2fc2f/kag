//! Application configuration, AI providers, databases, and retrieval
//! components
//!
//! This module provides the data structures and custom deserialization logic
//! required to parse the application's configuration file. It enforces strict
//! validation on component names and automatically handles fallback URLs and
//! credentials for supported services

use std::{
  collections::BTreeSet, fmt::Debug, fs, ops::Deref, path::Path, str::FromStr,
};

use anyhow::Context;
use hashbrown::HashMap;
use minijinja::Environment;
use serde::{Deserialize, Deserializer, de::DeserializeOwned};

use crate::cli::component::ComponentName;

/// Structure for the configuration of the program
///
/// This serves as the root deserialization target for the configuration file.
/// Missing sections will default to empty maps automatically
#[derive(Debug, Deserialize)]
pub struct Config {
  /// A map of configured AI providers used by the application
  #[serde(default)]
  pub providers: HashMap<ComponentName, Provider>,

  /// A map of configured databases used by the application
  #[serde(default)]
  pub databases: HashMap<ComponentName, Database>,

  /// A map of data retrievers used for RAG operations
  #[serde(default)]
  pub retrievers: HashMap<ComponentName, Retriever>,
}

/// Represents a supported AI service provider.
///
/// This enum is internally tagged by the `type` field in the configuration
/// file
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum Provider {
  /// Configuration for a local or remote Ollama instance
  Ollama {
    /// The base HTTP URL of the Ollama server
    #[serde(default = "ollama_default_url")]
    url: String,
  },
  /// Configuration for the OpenAI API
  OpenAI {
    /// The base HTTP URL of the OpenAI API
    #[serde(default = "openai_default_url")]
    url: String,
    /// The secret API key required to authenticate requests to OpenAI
    key: String,
  },
}

/// Helper function that returns the default local endpoint for Ollama
fn ollama_default_url() -> String {
  "http://localhost:11434".to_string()
}

/// Helper function that returns the default v1 API endpoint for OpenAI
fn openai_default_url() -> String {
  "https://api.openai.com/v1".to_string()
}

/// Represents a supported database connection
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum Database {
  /// Configuration for connecting to a Neo4j graph database
  Neo4j {
    /// The URI connection string for the Neo4j instance.
    #[serde(default = "neo4j_default_uri")]
    uri: String,

    /// The username for database authentication.
    #[serde(default = "neo4j_default_user")]
    user: String,

    /// The password for database authentication.
    password: String,

    /// The specific target database name.
    #[serde(default = "neo4j_default_database")]
    database: String,
  },
}

/// Helper function that returns the default local endpoint for Neo4j
fn neo4j_default_uri() -> String {
  "127.0.0.1:7687".to_string()
}

/// Helper function that returns the default user for Neo4j ("neo4j")
fn neo4j_default_user() -> String {
  "neo4j".to_string()
}

/// Helper function that returns the default database for Neo4j ("neo4j")
fn neo4j_default_database() -> String {
  "neo4j".to_string()
}

/// Represents a data retrieval component
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum Retriever {
  /// Configuration for an embedding model used to vectorize data for
  /// retrieval
  Embedder {
    /// The identifier of the configured AI `Provider` to use
    provider: ComponentName,

    /// The specific model name to use for generating embeddings
    model: String,

    /// The maximum number of top similar elements to retrieve (Top-K)
    #[serde(default = "default_top_k")]
    top_k: u32,

    /// Database-specific configuration extensions for the retriever, bykeyed
    /// by the targeted database name
    #[serde(default)]
    extra: HashMap<ComponentName, RetrieverExtra>,
  },
}

/// Specialized configuration extensions for specific database backends
/// or advanced retrieval mechanisms
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum RetrieverExtra {
  /// Configuration specific to a Neo4j graph database retrieval backend
  Neo4j {
    /// The specific vector index in Neo4j to query against
    index: String,

    /// The depth of the graph neighborhood to retrieve around the matched
    /// nodes.
    /// A value of 1 means direct neighbors, 2 means neighbors of neighbors,
    /// etc.
    #[serde(default = "default_neighborhood")]
    neighborhood: u32,

    /// The strategy used to translate the retrieved Neo4j graph data into
    /// text
    ///
    /// This determines how the nodes and relationships are formatted before
    /// being passed to the next stage of the RAG pipeline.
    /// By default, it uses the `FormalTriplet` format
    #[serde(default)]
    translation: Neo4jTranslationStrategy,
  },
}

/// Defines the formatting strategy used to convert retrieved Neo4j graph data
/// into a text representation.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Neo4jTranslationStrategy {
  /// Formats the graph data as strict Subject-Predicate-Object triplets.
  /// Example:
  ///   `(Albert_Einstein)-[:educatedAt]->(University_of_Zurich)`
  ///   `(Albert_Einstein)-[age]->(76)`
  FormalTriplet {
    /// Maps a combination of node labels to a specific list of property keys
    /// that should be extracted as special property triplets.
    ///
    /// If a node's label combination is NOT present in this map, ALL of its
    /// properties will be extracted by default.
    ///
    /// Example: {"Person"} -> ["age", "birthplace"]
    /// Yields:
    ///   `(Albert_Einstein)-[age]->(76)`
    ///   `(Albert_Einstein)-[birthplace]->(Ulm)`
    #[serde(default)]
    property_filters: HashMap<LabelSet, Vec<String>>,

    /// Maps a relationship type to a specific list of property keys
    /// that should be extracted as property triplets.
    ///
    /// If a relationship type is NOT present in this map, ALL of its
    /// properties will be extracted by default.
    ///
    /// Example: "educatedAt" -> ["during"]
    /// Yields:
    ///   `(Albert_Einstein)-[:educatedAt { during: 1970 }]->(University_of_Zurich)`
    #[serde(default)]
    relationship_property_filters: HashMap<String, Vec<String>>,
  },

  /// Formats the graph data into natural language OpenIE style triplets.
  /// Example:
  ///   `Albert Einstein received his PhD from the University of Zurich`
  TextualTriplet {
    /// Maps a combination of node labels to a validated format template.
    /// The format uses curly braces to inject node properties.
    /// Example: {"Person", "Actor"} -> "the actor {name}"
    node_formats: HashMap<LabelSet, FormatTemplate>,

    /// Maps a combination of node labels to a list of validated format
    /// templates. This is used to extract intrinsic node properties as
    /// standalone statements.
    /// The special property `{FROM}` is available to inject the base text
    /// of the node (which is evaluated using `node_formats`).
    /// Example: {"Person"} -> [
    ///     "{FROM} is {age} years old.",
    ///     "{FROM} was born in {birthplace}."
    /// ]
    property_formats: HashMap<LabelSet, Vec<FormatTemplate>>,

    /// Maps a relationship type to its validated format template.
    /// The format uses curly braces to inject relationship properties.
    /// Two special properties, `{FROM}` and `{TO}`, are also available to
    /// inject the formatted text of the origin and destination nodes
    /// respectively.
    /// Example: "ACTED_IN" -> "{FROM} acted in {role} during {year} in {TO}"
    relation_formats: HashMap<String, FormatTemplate>,
  },
}

impl Default for Neo4jTranslationStrategy {
  fn default() -> Self {
    Self::FormalTriplet {
      property_filters: Default::default(),
      relationship_property_filters: Default::default(),
    }
  }
}

/// A wrapper type around `BTreeSet<String>` designed to represent and parse a
/// collection of node labels.
#[derive(PartialEq, Eq, Hash)]
pub struct LabelSet(pub BTreeSet<String>);

impl Deref for LabelSet {
  type Target = BTreeSet<String>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl Debug for LabelSet {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self.0)
  }
}

impl FromStr for LabelSet {
  type Err = std::convert::Infallible;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(LabelSet(
      s.split(':')
        .map(|label| label.trim().to_string())
        .filter(|label| !label.is_empty())
        .collect(),
    ))
  }
}

impl<'de> Deserialize<'de> for LabelSet {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let raw_format = String::deserialize(deserializer)?;
    LabelSet::from_str(&raw_format).map_err(serde::de::Error::custom)
  }
}

/// A wrapper around parsed tokens to allow custom Serde deserialization
/// for both nodes and relationships.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FormatTemplate(pub Vec<FormatToken>);

impl<'de> Deserialize<'de> for FormatTemplate {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let raw_format = String::deserialize(deserializer)?;
    parse_template(&raw_format).map_err(serde::de::Error::custom)
  }
}

/// Parses a template string into a vector of format tokens.
/// Validates that `{` and `}` are properly matched.
fn parse_template(input: &str) -> Result<FormatTemplate, String> {
  let mut tokens = Vec::new();
  let mut current_buffer = String::new();
  let mut in_property = false;

  for c in input.chars() {
    match c {
      '{' => {
        if in_property {
          return Err(
            "Invalid format: nested '{' are not allowed.".to_string(),
          );
        }
        if !current_buffer.is_empty() {
          tokens.push(FormatToken::Literal(current_buffer.clone()));
          current_buffer.clear();
        }
        in_property = true;
      }
      '}' => {
        if !in_property {
          return Err(
            "Invalid format: found '}' without a matching '{'.".to_string(),
          );
        }
        if current_buffer.is_empty() {
          return Err("Invalid format: empty property name '{}'.".to_string());
        }
        tokens.push(FormatToken::Property(current_buffer.clone()));
        current_buffer.clear();
        in_property = false;
      }
      _ => {
        current_buffer.push(c);
      }
    }
  }

  if in_property {
    return Err("Invalid format: missing closing '}'.".to_string());
  }

  if !current_buffer.is_empty() {
    tokens.push(FormatToken::Literal(current_buffer));
  }

  Ok(FormatTemplate(tokens))
}

/// Represents a single piece of a parsed template string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatToken {
  /// A literal string of text to be inserted as-is.
  Literal(String),
  /// The name of a property to be dynamically extracted.
  Property(String),
}

/// Returns the default number of elements to retrieve (Top-K).
fn default_top_k() -> u32 {
  5
}

/// Returns the default graph neighborhood depth (number of hops).
fn default_neighborhood() -> u32 {
  1
}

/// Loads and deserializes a configuration file.
///
/// The file is first rendered as a [MiniJinja] template, which exposes two
/// helper functions:
///
/// - `file(path)` — inlines the content of the file at `path`
/// - `env(name)` — inlines the value of the environment variable `name`,
///   or an empty string if it is not set
///
/// The rendered output is then parsed as [TOML] into `T`.
///
/// # Errors
///
/// Returns an error if:
/// - the file at `config_path` cannot be read
/// - the template rendering fails (e.g. a `file()` call references a missing
///   file)
/// - the rendered output is not valid TOML or does not match the shape of `T`
///
/// [MiniJinja]: https://docs.rs/minijinja
/// [TOML]: https://toml.io
pub fn load_config<T>(config_path: impl AsRef<Path>) -> anyhow::Result<T>
where
  T: DeserializeOwned,
{
  let raw = fs::read_to_string(config_path)
    .context("The configuration file could not be read")?;

  let mut env = Environment::new();
  env.add_function("file", |f: String| {
    fs::read_to_string(&f).map_err(|e| {
      minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("The file {f} could not be read: {e:#}."),
      )
    })
  });
  env.add_function("env", |e: String| std::env::var(&e).unwrap_or_default());

  let rendered = env
    .render_str(&raw, minijinja::context!())
    .context("The special syntax in the configuration file failed to render")?;

  toml::from_str(&rendered)
    .context("The configuration file could not be parsed as valid TOML")
}

#[cfg(test)]
#[allow(clippy::missing_docs_in_private_items)]
mod tests {
  use std::{collections::BTreeSet, str::FromStr};

  use super::*;

  // ---- parse_template ----

  #[test]
  fn parse_template_empty_input_yields_no_tokens() {
    assert_eq!(parse_template("").unwrap(), FormatTemplate(vec![]));
  }

  #[test]
  fn parse_template_plain_literal() {
    assert_eq!(
      parse_template("hello world").unwrap(),
      FormatTemplate(vec![FormatToken::Literal("hello world".into())])
    );
  }

  #[test]
  fn parse_template_single_property() {
    assert_eq!(
      parse_template("{name}").unwrap(),
      FormatTemplate(vec![FormatToken::Property("name".into())])
    );
  }

  #[test]
  fn parse_template_literal_then_property() {
    assert_eq!(
      parse_template("the actor {name}").unwrap(),
      FormatTemplate(vec![
        FormatToken::Literal("the actor ".into()),
        FormatToken::Property("name".into()),
      ])
    );
  }

  #[test]
  fn parse_template_adjacent_properties_have_no_literal_between() {
    assert_eq!(
      parse_template("{a}{b}").unwrap(),
      FormatTemplate(vec![
        FormatToken::Property("a".into()),
        FormatToken::Property("b".into()),
      ])
    );
  }

  #[test]
  fn parse_template_interleaved_relation_format() {
    assert_eq!(
      parse_template("{FROM} acted in {role} during {year} in {TO}").unwrap(),
      FormatTemplate(vec![
        FormatToken::Property("FROM".into()),
        FormatToken::Literal(" acted in ".into()),
        FormatToken::Property("role".into()),
        FormatToken::Literal(" during ".into()),
        FormatToken::Property("year".into()),
        FormatToken::Literal(" in ".into()),
        FormatToken::Property("TO".into()),
      ])
    );
  }

  #[test]
  fn parse_template_rejects_nested_open_brace() {
    let err = parse_template("{a{b}").unwrap_err();
    assert!(err.contains("nested"), "unexpected error: {err}");
  }

  #[test]
  fn parse_template_rejects_unmatched_close_brace() {
    let err = parse_template("a}b").unwrap_err();
    assert!(
      err.contains("without a matching"),
      "unexpected error: {err}"
    );
  }

  #[test]
  fn parse_template_rejects_empty_property() {
    let err = parse_template("{}").unwrap_err();
    assert!(err.contains("empty property"), "unexpected error: {err}");
  }

  #[test]
  fn parse_template_rejects_missing_close_brace() {
    let err = parse_template("{name").unwrap_err();
    assert!(err.contains("missing closing"), "unexpected error: {err}");
  }

  #[test]
  fn parse_template_has_no_brace_escaping() {
    // Documents current behaviour: there is no escape, so a literal '{'
    // cannot be emitted and "{{" is a (nested) error.
    assert!(parse_template("{{").is_err());
  }

  // ---- ComponentName ----

  #[test]
  fn component_name_accepts_lowercase_digits_and_hyphen() {
    for s in ["abc", "a", "123", "valid-name-123", "-"] {
      let name = ComponentName::from_str(s)
        .unwrap_or_else(|e| panic!("'{s}' should be valid: {e}"));
      assert_eq!(&*name, s);
    }
  }

  #[test]
  fn component_name_rejects_empty() {
    assert!(ComponentName::from_str("").is_err());
  }

  #[test]
  fn component_name_rejects_uppercase_space_and_specials() {
    for s in [
      "UPPER",
      "with space",
      "under_score",
      "dot.name",
      "na\u{00EF}ve",
    ] {
      assert!(
        ComponentName::from_str(s).is_err(),
        "'{s}' should be rejected"
      );
    }
  }

  // ---- LabelSet ----

  fn label_set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
  }

  #[test]
  fn label_set_single_label() {
    assert_eq!(
      LabelSet::from_str("Person").unwrap().0,
      label_set(&["Person"])
    );
  }

  #[test]
  fn label_set_splits_on_colon() {
    assert_eq!(
      LabelSet::from_str("Person:Actor").unwrap().0,
      label_set(&["Actor", "Person"])
    );
  }

  #[test]
  fn label_set_trims_whitespace() {
    assert_eq!(
      LabelSet::from_str(" Person : Actor ").unwrap().0,
      label_set(&["Actor", "Person"])
    );
  }

  #[test]
  fn label_set_drops_empty_segments() {
    assert_eq!(
      LabelSet::from_str(":Person::Actor:").unwrap().0,
      label_set(&["Actor", "Person"])
    );
  }

  #[test]
  fn label_set_deduplicates() {
    let set = LabelSet::from_str("A:A:A").unwrap();
    assert_eq!(set.0.len(), 1);
    assert_eq!(set.0, label_set(&["A"]));
  }

  #[test]
  fn label_set_empty_input_is_empty_set() {
    assert!(LabelSet::from_str("").unwrap().0.is_empty());
    assert!(LabelSet::from_str("   ").unwrap().0.is_empty());
  }
}
