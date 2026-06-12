//! Neo4j Retrieval Module
//!
//! This module provides capabilities to perform vector-based semantic
//! searches against a Neo4j graph database and enrich the results by
//! traversing the local graph neighborhood.
//!
//! It combines vector similarity search with structural graph retrieval
//! (GraphRAG), converting the resulting subgraphs into flat context strings
//! ready for consumption by LLMs.

mod translation;

use anyhow::Context;
use async_stream::stream;
use log::{debug, info};
use neo4rs::query;
use rig_core::embeddings::Embedding;

use crate::{
  config::Neo4jTranslationStrategy,
  retrieval::database::{Output, neo4j::translation::process_translation},
};

/// Executes a hybrid vector and graph neighborhood search against Neo4j,
/// returning a formatted context string.
///
/// # Parameters
///
/// * `uri` - The connection URI for the Neo4j instance.
/// * `user` - The username for authentication.
/// * `password` - The password for authentication.
/// * `database` - The name of the target database within the Neo4j instance.
/// * `index` - The name of the pre-configured Neo4j vector index to query
///   against.
/// * `top_k` - The maximum number of initial seed nodes to retrieve via
///   vector similarity.
/// * `neighborhood` - The maximum path depth (`*0..N`) to traverse from the
///   seed nodes to collect context.
/// * `translation` - The strategy template configuration used to turn the
///   final rows into text.
/// * `embedding` - The vector representation of the user query used for the
///   initial semantic search.
///
/// # Errors
///
/// Returns an error if:
/// * The connection to the Neo4j database fails.
/// * The vector index does not exist or the embedding dimensions mismatch.
/// * Stream processing or translation formatting encounters an error.
///
/// # Panics
///
/// Panics if the `ConfigBuilder` fails to instantiate despite all credentials
/// being supplied.
#[allow(clippy::too_many_arguments)]
pub async fn retrieve_with_embedding(
  uri: &str,
  user: &str,
  password: &str,
  database: &str,
  index: &str,
  top_k: u32,
  neighborhood: u32,
  translation: &Neo4jTranslationStrategy,
  embedding: &Embedding,
) -> anyhow::Result<Output> {
  info!("Connecting to Neo4j database '{}' at {}", database, uri);
  let config = neo4rs::ConfigBuilder::default()
    .uri(uri)
    .user(user)
    .password(password)
    .db(database)
    .build()
    .context(
      "Neo4j ConfigBuilder failed despite all credentials being supplied",
    )?;

  let graph = neo4rs::Graph::connect(config).await.with_context(|| {
    format!(
      "Failed to establish a connection to Neo4j database '{}' at {}",
      database, uri
    )
  })?;

  debug!(
    "Executing vector search against index '{}' (top_k: {})",
    index, top_k
  );

  let query = query(&format!(
    "
      CALL db.index.vector.queryNodes($index, $top_k, $embed) \
      YIELD node \
      MATCH p = (node)-[*0..{neighborhood}]-(neighbor) \
      UNWIND relationships(p) AS rel
      RETURN \
        startNode(rel) AS source, \
        rel AS predicate, \
        endNode(rel) AS target\
    ",
  ))
  .param("index", index)
  .param("top_k", top_k)
  .param("embed", embedding.vec.clone());

  let mut retrieval = graph.execute(query).await.with_context(|| {
    format!(
      "\
        Failed to execute GraphRAG Cypher query (Vector Index: '{}', Top-K: \
        {}, Neighborhood Depth: {})\
      ",
      index, top_k, neighborhood
    )
  })?;

  let standard_stream = stream! {
      loop {
          match retrieval.next().await {
              Ok(Some(row)) => yield Ok(row),
              Ok(None) => break,
              Err(e) => {
                  yield Err(e);
                  break;
              }
          }
      }
  };

  debug!("Processing translation stream...");

  process_translation(translation, Box::pin(standard_stream))
    .await
    .with_context(|| {
      format!(
        "\
          Failed to translate Neo4j retrieval rows using translation \
          strategy: {:?}\
        ",
        translation
      )
    })
}
