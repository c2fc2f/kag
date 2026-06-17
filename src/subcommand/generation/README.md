# Generation Subcommand

Runs a single text generation and prints the result to standard output. By default it is a plain completion call; supplying a retriever and a database turns it into a **Knowledge-Augmented Generation (KAG/RAG)** run, where context retrieved from a knowledge graph is injected into the prompt before the model is queried.

## Pipeline

1. **Resolve components.** The `--provider` (and, for KAG, `--retriever` and `--database`) names are looked up in the [configuration file](../../../README.md#configuration). A missing component aborts the run with a clear error.
2. **Retrieve (KAG only).** When a retriever is supplied, the user prompt is embedded with the retriever's embedding model, a top-k vector search is run against the configured Neo4j index, and the matched seed nodes are expanded into their graph neighborhood. The resulting subgraph is rendered to text and exposed to the prompt template as `{{RETRIEVAL}}`.
3. **Render the prompt.** Templates are rendered with [MiniJinja](https://docs.rs/minijinja) under strict undefined-variable behavior: `{{INPUT}}` is replaced by the user prompt and `{{RETRIEVAL}}` by the retrieved context. If a referenced variable is never provided, rendering fails rather than silently emitting an empty value.
4. **Generate.** The rendered prompt is sent to the completion model and the response is written to standard output.

When no retriever is configured, steps 2 is skipped entirely and the run proceeds as a standard generation.

## Usage

```
kag generation [OPTIONS] --provider <PROVIDER> --model <MODEL> --temperature <TEMPERATURE> <PROMPT>
```

| Flag | Short | Description | Default |
|---|---|---|---|
| `--config <FILE>` | `-c` | Path to the configuration file | `config.toml` |
| `--provider <NAME>` | `-p` | Completion provider component to use | *(required)* |
| `--model <MODEL>` | `-m` | Model identifier passed to the provider | *(required)* |
| `--temperature <T>` | `-t` | Sampling temperature; higher is more creative, lower more deterministic | *(required)* |
| `--tokens <N>` | `-n` | Maximum completion tokens. Maps to `max_tokens` on OpenAI and to `num_ctx` (context window) on Ollama | *(provider default)* |
| `--system-prompt <FILE>` | `-s` | System prompt template file structuring the context and question | *(none)* |
| `--retriever <NAME>` | `-r` | Retriever component; enables KAG. Must be paired with `--database` | *(none)* |
| `--database <NAME>` | `-d` | Database component to retrieve from. Must be paired with `--retriever` | *(none)* |
| `<PROMPT>` | | The user prompt. Pass `-` to read the prompt from standard input | *(required)* |

`--retriever` and `--database` form a group: either supply both to enable KAG, or neither. Temperatures above `1.5` are accepted but emit a warning, as the output tends to become erratic.

## Prompt templates

The system prompt (and the user prompt, if no system prompt is given) is a MiniJinja template. Two placeholders are substituted at render time:

| Placeholder | Replaced with | Available when |
|---|---|---|
| `{{INPUT}}` | The user's input prompt | a system prompt is supplied |
| `{{RETRIEVAL}}` | The retrieved graph context | KAG is enabled and the retrieval returned a non-empty result |

Because rendering is strict, guard optional values with MiniJinja filters — for example `{{ RETRIEVAL | default('<empty>') }}` so the template still renders when retrieval is empty or disabled. A ready-to-use prompt is provided in [`examples/prompt/umls_prompt.md`](../../../examples/prompt/umls_prompt.md).

## Examples

Standard generation, no retrieval:

```bash
kag generation \
  --provider ollama --model qwen3.5:9b --temperature 0.0 \
  "Explain what a knowledge graph is in two sentences."
```

Knowledge-augmented generation against Neo4j, with a system prompt and the user prompt piped from standard input:

```bash
echo "What conditions are related to Rheumatoid Arthritis?" | \
kag generation \
  --provider ollama --model qwen3.5:9b --temperature 0.0 \
  --system-prompt examples/prompt/umls_prompt.md \
  --retriever neo4j-embedding-triplet-formal --database neo4j \
  -
```

Cap the OpenAI completion length and raise creativity:

```bash
kag generation \
  --provider openai --model gpt-4o-mini --temperature 0.7 --tokens 800 \
  "Draft a one-paragraph abstract about graph-augmented language models."
```

## Notes

- Increasing log verbosity with `-v`/`-vv` surfaces the resolved components, the final rendered prompt, and timing for the retrieval and generation phases.
- The retrieval context, the final prompt, and per-phase statistics are tracked internally; in the `benchmark` subcommand the same pipeline serializes them to disk for later analysis.

See the [project README](../../../README.md) for configuration, retriever, and translation-strategy details.
