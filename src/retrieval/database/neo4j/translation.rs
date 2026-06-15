//! Neo4j Graph Data Translation Module
//!
//! This module provides mechanisms to process and format Neo4j graph query
//! results (streams of [`neo4rs::Row`]) into structured textual
//! representations. This forms a critical part of a Retrieval-Augmented
//! Generation (RAG) pipeline, converting raw graph subgraphs, nodes, and
//! relationships into contextual text data that Large Language Models (LLMs)
//! can comprehend.
//!
//! It supports two primary formatting workflows via
//! [`Neo4jTranslationStrategy`]:
//! 1. **FormalTriplet**: Generates a Cypher-like representation showing
//!    formal relationship connections and explicit node/relationship
//!    properties.
//! 2. **TextualTriplet**: Generates natural language or template-bound
//!    textual statements using pre-configured tokens and string templates.

use std::{collections::BTreeSet, fmt::Write, hash::Hasher, time::Instant};

use anyhow::Context;
use futures::{Stream, StreamExt};
use hashbrown::{Equivalent, HashSet};
use log::{debug, info, trace, warn};
use neo4rs::{Node, Relation, Row};

use crate::{
  config::{FormatTemplate, FormatToken, LabelSet, Neo4jTranslationStrategy},
  retrieval::database::{Output, Stats},
};

/// A borrowed view over a set of node labels, used to look up entries in the
/// label-keyed format maps without allocating an owned [`LabelSet`].
///
/// It implements [`Equivalent<LabelSet>`] and [`Hash`] so that it hashes and
/// compares identically to a [`LabelSet`] holding the same labels. This
/// relies on the invariant that the wrapped set is sorted (a [`BTreeSet`]),
/// matching [`LabelSet`]'s own ordering.
struct QuerySet<'a>(
  /// The borrowed, sorted set of labels to look up.
  &'a BTreeSet<&'a str>,
);

impl<'a> Equivalent<LabelSet> for QuerySet<'a> {
  fn equivalent(&self, key: &LabelSet) -> bool {
    if self.0.len() != key.len() {
      return false;
    }
    self.0.iter().zip(key.iter()).all(|(&a, b)| a == b.as_str())
  }
}

impl<'a> std::hash::Hash for QuerySet<'a> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.hash(state);
  }
}

/// Processes an asynchronous stream of Neo4j database rows into a formatted
/// string.
///
/// This function consumes a stream of `neo4rs::Row` results and translates
/// them into a concatenated string representation based on the provided
/// [`Neo4jTranslationStrategy`].
///
/// # Arguments
///
/// * `translation` - A reference to the strategy dictating how the nodes and
///   relationships should be formatted into text.
/// * `stream` - An asynchronous stream yielding `Result<Row, neo4rs::Error>`.
///   It must implement `Unpin` to be safely polled within the loop.
///
/// # Returns
///
/// Returns a `Result` containing the fully concatenated `String` of
/// translated rows if successful, or a `neo4rs::Error` if reading from the
/// underlying stream fails, or a `neo4rs::DeError` if there was a
/// deserialization error.
pub async fn process_translation(
  translation: &Neo4jTranslationStrategy,
  mut stream: impl Stream<Item = Result<Row, neo4rs::Error>> + Unpin,
) -> anyhow::Result<Output> {
  let start = Instant::now();
  let mut buf = String::new();
  let mut vertices = 0u32;
  let mut relationships = 0u32;
  let mut properties = 0u32;
  let mut row_count = 0usize;

  info!("Starting Neo4j stream translation process.");
  debug!(
    "Parsing retrieved rows using translation strategy: {:?}",
    translation
  );

  match translation {
    Neo4jTranslationStrategy::FormalTriplet {
      property_filters,
      relationship_property_filters,
    } => {
      debug!("Executing FormalTriplet strategy branch.");
      let mut processed_nodes = HashSet::new();

      while let Some(row_result) = stream.next().await {
        row_count += 1;
        let row = row_result.with_context(|| {
          format!("Error reading row {row_count} from stream")
        })?;

        trace!("FormalTriplet - Processing Row {}", row_count);

        let source: Node = row
          .get("source")
          .context("Missing 'source' property in the row")?;

        let predicate: Relation = row
          .get("predicate")
          .context("Missing 'predicate' property in the row")?;

        let target: Node = row
          .get("target")
          .context("Missing 'target' property in the row")?;

        let rel_type = predicate.typ();

        let keys_iter = match relationship_property_filters.get(rel_type) {
          Some(specific_keys) => {
            debug!(
              "Applying specific property filters for relationship {}.",
              rel_type
            );
            itertools::Either::Left(specific_keys.iter().map(|k| k.as_str()))
          }
          None => {
            trace!(
              "No property filters for relationship {}, fetching all keys.",
              rel_type
            );
            itertools::Either::Right(predicate.keys().into_iter())
          }
        };

        let mut props_str = String::new();
        for key in keys_iter {
          match predicate.get::<serde_json::Value>(key) {
            Ok(val) => {
              if props_str.is_empty() {
                props_str.push_str(" { ");
              } else {
                props_str.push_str(", ");
              }
              props_str.push_str(&format!("{}: {}", key, val));
              properties += 1;
            }
            Err(e) => {
              warn!(
                "Property '{}' requested but missing on relationship {}: {}",
                key, rel_type, e
              );
            }
          }
        }
        if !props_str.is_empty() {
          props_str.push_str(" }");
        }

        relationships += 1;

        buf.push_str(&format!(
          "({})-[:{}{}]->({})\n",
          source.id(),
          predicate.typ(),
          props_str,
          target.id()
        ));

        for node in [&source, &target] {
          if !processed_nodes.insert(node.id()) {
            trace!(
              "Node {} already processed, skipping properties.",
              node.id()
            );
            continue;
          }

          vertices += 1;

          let labels: BTreeSet<_> = node.labels().into_iter().collect();

          let keys_iter = match property_filters.get(&QuerySet(&labels)) {
            Some(specific_keys) => {
              debug!(
                "Applying specific property filters for node {}.",
                node.id()
              );
              itertools::Either::Left(specific_keys.iter().map(|k| k.as_str()))
            }
            None => {
              trace!(
                "No property filters for node {}, fetching all keys.",
                node.id()
              );
              itertools::Either::Right(node.keys().into_iter())
            }
          };

          for key in keys_iter {
            match node.get::<serde_json::Value>(key) {
              Ok(val) => {
                buf.push_str(&format!(
                  "({})-[{}]->({})\n",
                  node.id(),
                  key,
                  val
                ));
                properties += 1;
              }
              Err(e) => {
                warn!(
                  "Property '{}' requested but missing on node {}: {}",
                  key,
                  node.id(),
                  e
                );
              }
            }
          }
        }
      }
    }
    Neo4jTranslationStrategy::TextualTriplet {
      node_formats,
      property_formats,
      relation_formats,
    } => {
      debug!("Executing TextualTriplet strategy branch.");
      let mut processed_nodes = BTreeSet::new();
      let mut displayed_nodes = BTreeSet::new();

      while let Some(row_result) = stream.next().await {
        row_count += 1;
        let row = row_result.with_context(|| {
          format!("Error reading row {row_count} from stream")
        })?;

        trace!("TextualTriplet - Processing Row {}: {:?}", row_count, row);

        let source: Node = row
          .get("source")
          .context("Missing 'source' property in the row")?;

        let predicate: Relation = row
          .get("predicate")
          .context("Missing 'predicate' property in the row")?;

        let target: Node = row
          .get("target")
          .context("Missing 'target' property in the row")?;

        let source_labels: BTreeSet<_> = source.labels().into_iter().collect();
        let target_labels: BTreeSet<_> = target.labels().into_iter().collect();

        let mut source_text = String::new();
        let mut source_props = 0u32;
        if let Some(template) = node_formats.get(&QuerySet(&source_labels)) {
          source_props = template.render_node(&source, &mut source_text);
        } else {
          trace!(
            "\
              No format template found for source labels {:?}, falling back \
              to ID.\
            ",
            source_labels
          );
          let _ = write!(&mut source_text, "{}", source.id());
        }

        let mut target_text = String::new();
        let mut target_props = 0u32;
        if let Some(template) = node_formats.get(&QuerySet(&target_labels)) {
          target_props = template.render_node(&target, &mut target_text);
        } else {
          trace!(
            "\
              No format template found for target labels {:?}, falling back \
              to ID.\
            ",
            target_labels
          );
          let _ = write!(&mut target_text, "{}", target.id());
        }

        if let Some(rel_template) = relation_formats.get(predicate.typ()) {
          let rel_props = rel_template.render_relation(
            &predicate,
            &source_text,
            &target_text,
            &mut buf,
          );
          buf.push('\n');

          relationships += 1;

          properties += rel_props + source_props + target_props;

          if displayed_nodes.insert(source.id()) {
            vertices += 1;
          }
          if displayed_nodes.insert(target.id()) {
            vertices += 1;
          }
        } else {
          warn!(
            "No relation format found for predicate type: {}",
            predicate.typ()
          );
        }

        for (node, labels, base_text) in [
          (&source, &source_labels, &source_text),
          (&target, &target_labels, &target_text),
        ] {
          if !processed_nodes.insert(node.id()) {
            continue;
          }

          if let Some(prop_templates) = property_formats.get(&QuerySet(labels))
          {
            for template in prop_templates {
              properties += template.render_property(node, base_text, &mut buf);
              buf.push('\n');
            }
            if !prop_templates.is_empty() && displayed_nodes.insert(node.id()) {
              vertices += 1;
            }
          } else {
            trace!("No property format found for labels: {:?}", labels);
          }
        }
      }
    }
  }

  info!("Successfully processed {} relationship rows.", row_count);

  Ok(Output {
    result: buf,
    stats: Stats::Neo4j {
      vertices,
      relationships,
      properties,
      time: start.elapsed(),
    },
  })
}

impl FormatTemplate {
  /// Renders a node directly into the provided buffer
  pub fn render_node(&self, node: &Node, buf: &mut String) -> u32 {
    let mut properties = 0u32;

    for token in &self.0 {
      match token {
        FormatToken::Literal(text) => buf.push_str(text),
        FormatToken::Property(key) => {
          if let Ok(val) = node.get::<serde_json::Value>(key) {
            let _ = write!(buf, "{}", val);
            properties += 1;
          } else {
            warn!(
              "FormatToken missing property '{}' on Node {}",
              key,
              node.id()
            );
            let _ = write!(buf, "{{{}}}", key);
          }
        }
      }
    }

    properties
  }

  /// Renders a relationship directly into the provided buffer
  pub fn render_relation(
    &self,
    rel: &Relation,
    from: &str,
    to: &str,
    buf: &mut String,
  ) -> u32 {
    let mut properties = 0u32;

    for token in &self.0 {
      match token {
        FormatToken::Literal(text) => buf.push_str(text),
        FormatToken::Property(key) => match key.as_str() {
          "FROM" => buf.push_str(from),
          "TO" => buf.push_str(to),
          _ => {
            if let Ok(val) = rel.get::<serde_json::Value>(key) {
              let _ = write!(buf, "{}", val);
              properties += 1;
            } else {
              warn!(
                "FormatToken missing property '{}' on Relation {}",
                key,
                rel.id()
              );
              let _ = write!(buf, "{{{}}}", key);
            }
          }
        },
      }
    }

    properties
  }

  /// Renders a standalone property statement directly into the provided
  /// buffer
  pub fn render_property(
    &self,
    node: &Node,
    from: &str,
    buf: &mut String,
  ) -> u32 {
    let mut properties = 0u32;

    for token in &self.0 {
      match token {
        FormatToken::Literal(text) => buf.push_str(text),
        FormatToken::Property(key) => match key.as_str() {
          "FROM" => buf.push_str(from),
          _ => {
            if let Ok(val) = node.get::<serde_json::Value>(key) {
              let _ = write!(buf, "{}", val);
              properties += 1;
            } else {
              warn!(
                "\
                  FormatToken missing property '{}' for standalone statement \
                  on Node {}\
                ",
                key,
                node.id()
              );
              let _ = write!(buf, "{{{}}}", key);
            }
          }
        },
      }
    }

    properties
  }
}

#[cfg(test)]
#[allow(clippy::missing_docs_in_private_items)]
mod tests {
  use std::{collections::BTreeSet, str::FromStr};

  use hashbrown::HashMap;

  use super::*;

  #[test]
  fn queryset_matches_equivalent_label_set_in_map() {
    let mut map: HashMap<LabelSet, &'static str> = HashMap::new();
    map.insert(LabelSet::from_str("Person:Actor").unwrap(), "hit");

    // Declared in a different order and borrowed as &str rather than owned
    // String: the lookup must still resolve to the same entry.
    let labels: BTreeSet<&str> = ["Actor", "Person"].into_iter().collect();
    assert_eq!(map.get(&QuerySet(&labels)), Some(&"hit"));
  }

  #[test]
  fn queryset_does_not_match_on_size_mismatch() {
    let mut map: HashMap<LabelSet, &'static str> = HashMap::new();
    map.insert(LabelSet::from_str("Person").unwrap(), "hit");

    let labels: BTreeSet<&str> = ["Person", "Actor"].into_iter().collect();
    assert_eq!(map.get(&QuerySet(&labels)), None);
  }

  #[test]
  fn queryset_does_not_match_on_label_difference() {
    let mut map: HashMap<LabelSet, &'static str> = HashMap::new();
    map.insert(LabelSet::from_str("Person").unwrap(), "hit");

    let labels: BTreeSet<&str> = ["Movie"].into_iter().collect();
    assert_eq!(map.get(&QuerySet(&labels)), None);
  }

  #[test]
  fn queryset_empty_label_set_matches_empty() {
    let mut map: HashMap<LabelSet, &'static str> = HashMap::new();
    map.insert(LabelSet::from_str("").unwrap(), "hit");

    let labels: BTreeSet<&str> = BTreeSet::new();
    assert_eq!(map.get(&QuerySet(&labels)), Some(&"hit"));
  }
}
