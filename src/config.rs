//! Application configuration, AI providers, databases, and retrieval
//! components
//!
//! This module provides the data structures and custom deserialization logic
//! required to parse the application's configuration file. It enforces strict
//! validation on component names and automatically handles fallback URLs and
//! credentials for supported services

use core::fmt;
use std::{collections::HashMap, ops::Deref, str::FromStr};

use serde::{Deserialize, Deserializer};

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

    /// A map of data retrievers used for GraphRAG operations
    #[serde(default)]
    pub retrievers: HashMap<ComponentName, Retriever>,
}

/// A validated, strictly formatted component identifier.
///
/// This type wraps a `String` and guarantees that the identifier complies
/// with system constraints upon deserialization. It implements [`Deref`] to
/// allow seamless usage as a standard string slice (`&str`)
///
/// # Validation Rules
/// - Cannot be empty
/// - No spaces allowed
/// - No special characters allowed (except hyphens)
/// - Strictly lowercase alphanumeric characters (`a-z`, `0-9`, `-`)
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct ComponentName(String);

impl FromStr for ComponentName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("Component name cannot be empty".to_string());
        }

        if !s
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(
                "Component name must contain only lowercase alphanumeric characters and hyphens"
                    .to_string(),
            );
        }

        Ok(Self(s.to_string()))
    }
}

impl<'de> Deserialize<'de> for ComponentName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_str(&String::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl Deref for ComponentName {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for ComponentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for ComponentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
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
        top_k: usize,

        /// The depth of the graph neighborhood to retrieve around the matched
        /// nodes.
        /// A value of 1 means direct neighbors, 2 means neighbors of
        /// neighbors, etc.
        #[serde(default = "default_neighborhood")]
        neighborhood: usize,
    },
}

/// Returns the default number of elements to retrieve (Top-K).
fn default_top_k() -> usize {
    5
}

/// Returns the default graph neighborhood depth (number of hops).
fn default_neighborhood() -> usize {
    1
}
