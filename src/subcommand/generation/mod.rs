//! Command implementation for the retrieval and processing workflow. Command
//! implementation for text generation, supporting optional
//! Knowledge-Augmented Generation (KAG)
//!
//! This module handles the execution of text generation tasks based on user
//! input. It supports standard text generation using a specified provider and
//! model. Additionally, if a `retriever` is provided, the generation process
//! is augmented with external context (KAG/RAG workflow) before querying the
//! model

use std::{
  collections::BTreeSet,
  fmt::Write,
  hash::{Hash, Hasher},
  path::{Path, PathBuf},
  process::ExitCode,
  time::Instant,
};

use async_stream::stream;
use clap_stdin::MaybeStdin;
use futures::{Stream, StreamExt};
use hashbrown::Equivalent;
use log::{debug, error, info, trace, warn};
use neo4rs::{Node, Relation, Row, query};
use rig_core::{
  agent::Agent,
  client::{CompletionClient, EmbeddingsClient, Nothing},
  completion::Prompt,
  embeddings::EmbeddingModel,
  providers::{ollama, openai},
};
use serde_json::json;

use crate::{
  config::{
    ComponentName, Config, Database, FormatTemplate, FormatToken, LabelSet,
    Neo4jTranslationStrategy, Provider, Retriever,
  },
  match_err, match_some,
};

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

  /// The temperature of the model. Increasing the temperature will make the
  /// model answer more creatively
  #[arg(short, long, default_value_t = 1.0)]
  temperature: f64,

  /// The maximum number of tokens for the completion. Increasing this limit
  /// allows the model to produce longer, more detailed outputs
  ///
  /// On Ollama, sets `num_ctx` (context window size)
  #[arg(short = 'n', long)]
  tokens: Option<u64>,

  /// The system prompt template to structure the context and question
  ///
  /// Any instance of `{{QUESTION}}` will be replaced by the user's input
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
  /// This value will be injected into the `{{QUESTION}}` placeholder within
  /// the system prompt.
  pub prompt: MaybeStdin<String>,
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
///`ExitCode::FAILURE` if a required component cannot be resolved.
pub fn run(args: Args, config: Config) -> ExitCode {
  info!("Starting text generation workflow...");

  if args.temperature > 1.5 {
    warn!(
      "\
        High temperature ({}) detected. \
        Model output may be highly erratic or nonsensical.\
      ",
      args.temperature
    );
  }

  let retriever = match &args.retriever {
    Some(name) => {
      debug!("Resolving retriever component: {}", name);
      Some(match_some!(
        config.retrievers.get(name),
        "The requested retriever '{}' is missing from the configuration.",
        name
      ))
    }
    None => None,
  };
  let database = match &args.database {
    Some(name) => {
      debug!("Resolving database component: {}", name);
      Some(match_some!(
        config.databases.get(name),
        "The requested database '{}' is missing from the configuration.",
        name,
      ))
    }
    None => None,
  };
  let provider = match_some!(
    config.providers.get(&args.provider),
    "The requested provider '{}' is missing from the configuration.",
    args.provider
  );

  debug!(
    "Initializing {} completion model '{}'",
    args.provider, args.model
  );

  let model = match_err!(
    match provider {
      Provider::Ollama { url } => ollama::Client::builder()
        .api_key(Nothing)
        .base_url(url)
        .build()
        .map(|client| {
          AnyCompletionModel::Ollama({
            let mut c = client.agent(&args.model).temperature(args.temperature);
            if let Some(tokens) = args.tokens {
              c = c.additional_params(json!({
                "num_ctx": tokens
              }));
            }
            c.build()
          })
        }),
      Provider::OpenAI { url, key } => openai::Client::builder()
        .api_key(key)
        .base_url(url)
        .build()
        .map(|client| {
          AnyCompletionModel::OpenAI({
            let mut c = client.agent(&args.model).temperature(args.temperature);
            if let Some(tokens) = args.tokens {
              c = c.max_tokens(tokens);
            }
            c.build()
          })
        }),
    },
    "Failed to initialize the model {} for provider '{}'",
    args.model,
    args.provider
  );

  let rt = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
    .expect("Failed building the Runtime");

  trace!("Raw User Prompt: {:?}", args.prompt);

  let retrieval_buffer: Option<String> = match retriever {
    None => {
      info!("No retriever configured. Proceeding without KAG.");
      None
    }
    Some(Retriever::EmbedderNeo4j {
      provider,
      model,
      top_k,
      neighborhood,
      index,
      translation,
    }) => {
      info!("KAG enabled: Initializing Neo4j retrieval workflow...");

      let retrieval_start = Instant::now();

      debug!("Initializing embedder model '{}' via '{}'", model, provider);
      let embedder = match_err!(
        match config.providers.get(provider) {
          Some(Provider::Ollama { url }) => ollama::Client::builder()
            .api_key(Nothing)
            .base_url(url)
            .build()
            .map(|client| {
              AnyEmbedderModel::Ollama(client.embedding_model(model))
            }),
          Some(Provider::OpenAI { url, key }) => {
            openai::Client::builder()
              .api_key(key)
              .base_url(url)
              .build()
              .map(|client| {
                AnyEmbedderModel::OpenAI(client.embedding_model(model))
              })
          }
          None => {
            error!(
              "\
                The requested provider '{}' is missing from the \
                configuration.\
              ",
              args.provider
            );
            return ExitCode::FAILURE;
          }
        },
        "Failed to initialize the model {} for provider '{}'",
        args.model,
        args.provider
      );

      debug!("Generating embeddings for the user prompt...");
      let embed = match_err!(
        rt.block_on(embedder.embed_texts(&args.prompt)),
        "Failed to generate embeddings for the prompt"
      );

      let database = database.expect(
        "database name must always be populated when a retriever is provided",
      );

      let mut retrieval = match database {
        Database::Neo4j {
          uri,
          user,
          password,
          database,
        } => {
          info!("Connecting to Neo4j database '{}' at {}", database, uri);
          let config = neo4rs::ConfigBuilder::default()
            .uri(uri)
            .user(user)
            .password(password)
            .db(database.as_str())
            .build()
            .expect(
              "\
                Neo4j ConfigBuilder failed despite all credentials being \
                supplied\
              ",
            );

          let _ = rustls::crypto::ring::default_provider().install_default();

          let graph = match_err!(
            rt.block_on(neo4rs::Graph::connect(config)),
            "Failed to connect to the Neo4j database",
          );

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
          .param("index", index.as_str())
          .param("top_k", *top_k as u32)
          .param("embed", embed.vec);

          match_err!(
            rt.block_on(graph.execute(query)),
            "Failed to execute the query against the Neo4j database",
          )
        }
        #[allow(unreachable_patterns)]
        _ => {
          error!(
            "
              Unsupported database type.
              The chosen retriever requires a Neo4j database, but a \
              different database was configured.\
            "
          );
          return ExitCode::FAILURE;
        }
      };

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
      let buf = match rt.block_on(async {
        process_translation(translation, Box::pin(standard_stream)).await
      }) {
        Ok(parsed_buffer) => parsed_buffer,
        Err(e) => {
          log::error!("Failed to process database stream: {}", e);
          return ExitCode::FAILURE;
        }
      };

      debug!("Generated context buffer of {} bytes.", buf.len());
      trace!("Final Retrieval Buffer Content:\n{:?}", buf);

      info!(
        "Retrieval workflow completed in {:.2?}",
        retrieval_start.elapsed()
      );

      Some(buf)
    }
  };

  let prompt = match build_prompt(
    args.prompt.into_inner(),
    args.system_prompt.as_deref(),
    retrieval_buffer.as_deref(),
  ) {
    Ok(p) => p,
    Err(e) => {
      error!("Failed to read system prompt file: {}", e);
      return ExitCode::FAILURE;
    }
  };

  trace!("Final Prompt being sent to model:\n{}", prompt);
  info!("Sending prompt to the completion model...");

  let generation_start = Instant::now();

  let response = match_err!(
    rt.block_on(model.prompt(&prompt)),
    "Failed to generate a response from the model"
  );

  info!("Generation completed in {:.2?}", generation_start.elapsed());

  println!("{response}");

  ExitCode::SUCCESS
}

/// Compiles the final prompt by injecting the user input and retrieval
/// context into a template.
///
/// This function handles the interpolation of two specific placeholders:
/// - `{{QUESTION}}`: Replaced by the `raw_prompt`.
/// - `{{RETRIEVAL}}`: Replaced by the `retrieval_buffer`.
///
/// # Behavior
///
/// **When `system_prompt_path` is provided:**
/// 1. Replaces `{{QUESTION}}` with the `raw_prompt`.
/// 2. If a `retrieval_buffer` is provided:
///    - Replaces `{{RETRIEVAL}}` with the context.
///    - If `{{RETRIEVAL}}` is missing from the template, appends the context
///      to the end of the string.
/// 3. If `retrieval_buffer` is `None` but the template contains
///    `{{RETRIEVAL}}`, replaces the placeholder with an empty string to clean
///    up the output.
///
/// **When `system_prompt_path` is NOT provided (Fallback):**
/// 1. Uses the `raw_prompt` as the base template.
/// 2. If a `retrieval_buffer` is provided:
///    - Replaces `{{RETRIEVAL}}` within the `raw_prompt` if it exists.
///    - If `raw_prompt` does not contain `{{RETRIEVAL}}`, appends the context
///      to the end.
///
/// # Arguments
///
/// * `raw_prompt` - The user's initial question or input.
/// * `system_prompt_path` - Optional path to a text file containing the
///   system prompt template.
/// * `retrieval_buffer` - Optional context retrieved from a knowledge base or
///   search tool.
///
/// # Returns
///
/// Returns `Ok(String)` containing the fully formatted prompt, or
/// `Err(String)` if the system prompt file cannot be read.
pub fn build_prompt(
  mut raw_prompt: String,
  system_prompt_path: Option<&Path>,
  retrieval_buffer: Option<&str>,
) -> std::io::Result<String> {
  if let Some(sys_prompt_path) = system_prompt_path {
    info!(
      "System prompt template provided. Reading from {:?}",
      sys_prompt_path
    );

    let mut formatted = std::fs::read_to_string(sys_prompt_path)?;

    if formatted.contains("{{QUESTION}}") {
      debug!("Replacing {{QUESTION}} placeholder with user prompt.");
      formatted = formatted.replace("{{QUESTION}}", &raw_prompt);
    } else {
      warn!(
        "System prompt template does not contain a {{QUESTION}} placeholder."
      );
    }

    if let Some(context) = retrieval_buffer {
      if formatted.contains("{{RETRIEVAL}}") {
        debug!("Replacing {{RETRIEVAL}} placeholder with context buffer.");
        formatted = formatted.replace("{{RETRIEVAL}}", context);
      } else {
        warn!(
          "\
            KAG is enabled, but '{{RETRIEVAL}}' token was not found in the \
            template. Appending context to the end.\
          "
        );
        formatted.push_str("\n\nContext:\n");
        formatted.push_str(context);
      }
    } else if formatted.contains("{{RETRIEVAL}}") {
      warn!(
        "\
          Template contains {{RETRIEVAL}} but no retriever was configured. \
          Replacing with empty string.\
        "
      );
      formatted = formatted.replace("{{RETRIEVAL}}", "");
    }

    return Ok(formatted);
  }
  info!("No system prompt template provided. Using fallback logic.");

  if let Some(context) = retrieval_buffer {
    if raw_prompt.contains("{{RETRIEVAL}}") {
      debug!("Replacing {{RETRIEVAL}} token directly in the user prompt.");
      raw_prompt = raw_prompt.replace("{{RETRIEVAL}}", context);
    } else {
      warn!(
        "\
          KAG is enabled, but '{{RETRIEVAL}}' token was not found in the \
          prompt. Appending context to the end.\
        "
      );
      raw_prompt.push_str("\n\nContext:\n");
      raw_prompt.push_str(context);
    }
  }

  Ok(raw_prompt)
}

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
  pub async fn embed_texts(
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
///
/// # Panics
///
/// * Panics if the stream yields a row that is missing the expected
///   `"source"`, `"predicate"`, or `"target"` fields, or if those fields
///   cannot be cast to `Node`, `String`, and `Node` respectively.
pub async fn process_translation(
  translation: &Neo4jTranslationStrategy,
  mut stream: impl Stream<Item = Result<Row, neo4rs::Error>> + Unpin,
) -> anyhow::Result<String> {
  struct QuerySet<'a>(&'a BTreeSet<&'a str>);

  impl<'a> Equivalent<LabelSet> for QuerySet<'a> {
    fn equivalent(&self, key: &LabelSet) -> bool {
      if self.0.len() != key.len() {
        return false;
      }
      self.0.iter().zip(key.iter()).all(|(&a, b)| a == b.as_str())
    }
  }

  impl<'a> Hash for QuerySet<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
      self.0.hash(state);
    }
  }

  let mut buf = String::new();
  let mut row_count = 0;

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
      let mut processed_nodes = BTreeSet::new();

      while let Some(row_result) = stream.next().await {
        row_count += 1;
        let row = row_result.map_err(|e| {
          error!("Error reading row {} from stream: {}", row_count, e);
          e
        })?;

        trace!("FormalTriplet - Processing Row {}", row_count);

        let source: Node = row.get("source").unwrap();
        let predicate: Relation = row.get("predicate").unwrap();
        let target: Node = row.get("target").unwrap();

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

      while let Some(row_result) = stream.next().await {
        row_count += 1;
        let row = row_result.map_err(|e| {
          error!("Error reading row {} from stream: {}", row_count, e);
          e
        })?;

        trace!("TextualTriplet - Processing Row {}: {:#?}", row_count, row);

        let source: Node = row.get("source").unwrap();
        let predicate: Relation = row.get("predicate").unwrap();
        let target: Node = row.get("target").unwrap();

        let source_labels: BTreeSet<_> = source.labels().into_iter().collect();
        let target_labels: BTreeSet<_> = target.labels().into_iter().collect();

        let mut source_text = String::new();
        if let Some(template) = node_formats.get(&QuerySet(&source_labels)) {
          template.render_node(&source, &mut source_text);
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
        if let Some(template) = node_formats.get(&QuerySet(&target_labels)) {
          template.render_node(&target, &mut target_text);
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
          rel_template.render_relation(
            &predicate,
            &source_text,
            &target_text,
            &mut buf,
          );
          buf.push('\n');
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
              template.render_property(node, base_text, &mut buf);
              buf.push('\n');
            }
          } else {
            trace!("No property format found for labels: {:?}", labels);
          }
        }
      }
    }
  }

  info!("Successfully processed {} relationship rows.", row_count);

  Ok(buf)
}

impl FormatTemplate {
  /// Renders a node directly into the provided buffer
  pub fn render_node(&self, node: &Node, buf: &mut String) {
    for token in &self.0 {
      match token {
        FormatToken::Literal(text) => buf.push_str(text),
        FormatToken::Property(key) => {
          if let Ok(val) = node.get::<serde_json::Value>(key) {
            let _ = write!(buf, "{}", val);
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
  }

  /// Renders a relationship directly into the provided buffer
  pub fn render_relation(
    &self,
    rel: &Relation,
    from: &str,
    to: &str,
    buf: &mut String,
  ) {
    for token in &self.0 {
      match token {
        FormatToken::Literal(text) => buf.push_str(text),
        FormatToken::Property(key) => match key.as_str() {
          "FROM" => buf.push_str(from),
          "TO" => buf.push_str(to),
          _ => {
            if let Ok(val) = rel.get::<serde_json::Value>(key) {
              let _ = write!(buf, "{}", val);
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
  }

  /// Renders a standalone property statement directly into the provided
  /// buffer
  pub fn render_property(&self, node: &Node, from: &str, buf: &mut String) {
    for token in &self.0 {
      match token {
        FormatToken::Literal(text) => buf.push_str(text),
        FormatToken::Property(key) => match key.as_str() {
          "FROM" => buf.push_str(from),
          _ => {
            if let Ok(val) = node.get::<serde_json::Value>(key) {
              let _ = write!(buf, "{}", val);
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
  }
}
