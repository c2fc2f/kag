# kag

A command-line toolkit written in Rust for **Knowledge-Augmented Generation (KAG)**: it runs text generation against multiple LLM providers, optionally enriching each prompt with context retrieved from a [Neo4j](https://neo4j.com/) knowledge graph, and ships an evaluation benchmark runner — together with a scoring command — to compare techniques and models across datasets.

## Overview

KAG (also referred to as GraphRAG) extends standard Retrieval-Augmented Generation by retrieving a *subgraph* — seed nodes found through vector similarity plus their graph neighborhood — and translating it into text that is injected back into the prompt. This tool wires that pipeline end to end: it embeds the user query, runs a hybrid vector + neighborhood search against Neo4j, renders the resulting subgraph into a textual context, and feeds it to a completion model.

Unlike the sibling tools that produce the graph itself ([pm2kg](https://github.com/c2fc2f/PubMed-MeSH-to-KG), [umls2kg](https://github.com/c2fc2f/UMLS-to-KG)), `kag` *consumes* a live graph at query time. It is provider-agnostic — completion and embedding models can come from either a [Ollama](https://ollama.com/) instance or any OpenAI-compatible API.

The project is a single Cargo package (no workspace). The `kag` binary is the sole deliverable, built on:

- **`rig-core`** — provider abstraction for completion and embedding models (Ollama, OpenAI)
- **`neo4rs`** — async Bolt client for Neo4j
- **`minijinja`** — templating for both configuration files and prompts

## Requirements

- Rust toolchain (edition 2024, stable)
- At least one LLM provider: a reachable Ollama instance and/or an OpenAI-compatible API endpoint
- For knowledge augmentation: a running Neo4j instance (Bolt protocol) holding a pre-built vector index over your nodes

## Installation

### From source

```bash
git clone https://github.com/c2fc2f/kag
cd kag
cargo build --release
# or
cargo install --git https://github.com/c2fc2f/kag
```

The compiled binary will be at `target/release/kag`.

### With Nix

A Nix flake is provided:

```bash
nix run github:c2fc2f/kag -- --help
# or
nix build
# or, to enter a development shell:
nix develop
```

The Nix build also installs shell completions (bash, fish, zsh) and man pages.

## Usage

```
kag [OPTIONS] <COMMAND>
```

Run `kag --help` for the full list of subcommands, or `kag <COMMAND> --help` for subcommand-specific options.

### Global options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--verbose` | `-v` | Increase output verbosity (repeatable) | *(errors)* |
| `--quiet` | `-q` | Decrease output verbosity (repeatable) | |

## Subcommands

| Subcommand | Description | Documentation |
|---|---|---|
| `generation` | Run a single text generation, with optional Knowledge-Augmented Generation (KAG/RAG) when a retriever is supplied | [README](src/subcommand/generation/README.md) |
| `benchmark` | Evaluate datasets across multiple configured techniques and models, with parallel execution and resumable runs | [README](src/subcommand/benchmark/README.md) |
| `stats` | Score a benchmark result tree against the ground truth, reporting per-setup accuracy, precision, and coverage as a table or JSON | [README](src/subcommand/stats/README.md) |

A hidden `completion <SHELL>` subcommand prints a shell completion script to standard output.

## Configuration

The configuration file (`config.toml` by default) declares the reusable **components** that subcommands reference by name: AI providers, databases, and retrievers. Component names must be non-empty and contain only lowercase ASCII letters, digits, and hyphens.

The file is first rendered as a [MiniJinja](https://docs.rs/minijinja) template, then parsed as TOML. Two helper functions are available to keep secrets out of the file:

- `file(path)` — inlines the contents of the file at `path`
- `env(name)` — inlines the value of the environment variable `name` (empty string if unset)

### Providers

Each provider is internally tagged by its `type`.

| Type | Field | Description | Default |
|---|---|---|---|
| `Ollama` | `url` | Base URL of the Ollama server | `http://localhost:11434` |
| `OpenAI` | `url` | Base URL of the OpenAI-compatible API | `https://api.openai.com/v1` |
| `OpenAI` | `key` | API key | *(required)* |

### Databases

| Type | Field | Description | Default |
|---|---|---|---|
| `Neo4j` | `uri` | Bolt connection URI | `127.0.0.1:7687` |
| `Neo4j` | `user` | Username | `neo4j` |
| `Neo4j` | `password` | Password | *(required)* |
| `Neo4j` | `database` | Target database name | `neo4j` |

### Retrievers

A retriever describes how the knowledge graph is queried. The only current type, `Embedder`, vectorizes the query with an embedding model and runs a top-k vector search, optionally expanding into the graph neighborhood.

| Field | Description | Default |
|---|---|---|
| `provider` | Name of the provider component used to embed the query | *(required)* |
| `model` | Embedding model identifier | *(required)* |
| `top_k` | Number of seed nodes to retrieve by vector similarity | `5` |
| `extra` | Per-database backend settings, keyed by database component name | `{}` |

For a Neo4j backend, the `extra.<database>` block accepts:

| Field | Description | Default |
|---|---|---|
| `index` | Name of the pre-built Neo4j vector index to query | *(required)* |
| `neighborhood` | Graph hops to traverse from each seed node (`1` = direct neighbors) | `1` |
| `translation` | Strategy used to render the retrieved subgraph as text | `FormalTriplet` |

### Translation strategies

Two strategies turn the retrieved subgraph into prompt-ready text:

- **`FormalTriplet`** *(default)* — emits a Cypher-like representation, e.g. `(Albert_Einstein)-[:educatedAt]->(University_of_Zurich)` and `(Albert_Einstein)-[age]->(76)`. Optional `property_filters` (per node-label set) and `relationship_property_filters` (per relationship type) restrict which properties are emitted; when a label set or type is absent from the filter, all of its properties are included.
- **`TextualTriplet`** — emits natural-language statements from templates. `node_formats` maps a node-label set to a phrase (e.g. `the actor {name}`), `property_formats` turns intrinsic node properties into standalone sentences (the `{FROM}` placeholder injects the node phrase), and `relation_formats` renders each relationship type (the `{FROM}` and `{TO}` placeholders inject the source and target node phrases). Curly braces interpolate node/relationship properties.

### Sample configuration

```toml
[providers.ollama]
type = "Ollama"
url = "http://localhost:11434"

[providers.openai]
type = "OpenAI"
url = "https://api.openai.com/v1"
key = "{{ env('OPEN_AI_KEY') or file('.openai') }}"

[databases.neo4j]
type = "Neo4j"
password = "{{ env('NEO4J_PASSWORD') or 'neo4j' }}"

# Formal (Cypher-like) triplets
[retrievers.neo4j-embedding-triplet-formal]
type = "Embedder"
provider = "ollama"
model = "embeddinggemma:latest"
top_k = 7
extra.neo4j = { type = "Neo4j", index = "NODE_INDEX" }

# Natural-language triplets
[retrievers.neo4j-embedding-triplet-text]
type = "Embedder"
provider = "ollama"
model = "embeddinggemma:latest"
top_k = 7

[retrievers.neo4j-embedding-triplet-text.extra.neo4j]
type = "Neo4j"
index = "NODE_INDEX"
translation = { type = "TextualTriplet", node_formats = { Person = "the person {name}", Movie = "the movie {title}" }, relation_formats = { ACTED_IN = "{FROM} acted in {role} during {year} in {TO}" } }
```

See [`examples/config/config.sample.toml`](examples/config/config.sample.toml) for a fuller example.

## Prompts

Prompts are MiniJinja templates too. The following placeholders are substituted before the prompt reaches the model:

| Placeholder | Replaced with | Available when |
|---|---|---|
| `{{INPUT}}` | The user's input prompt | always (when a system prompt is used) |
| `{{RETRIEVAL}}` | The retrieved graph context | KAG enabled |
| `{{CHOICE}}` | The answer options | `benchmark`, on multiple-choice entries |

A starting prompt is provided in [`examples/prompt/umls_prompt.md`](examples/prompt/umls_prompt.md).

## Shell completions

Generate a completion script for your shell and source it (the Nix package installs these automatically):

```bash
kag completion bash > kag.bash
kag completion fish > kag.fish
kag completion zsh  > _kag
```

## License

This project is licensed under the [MIT License](LICENSE).
