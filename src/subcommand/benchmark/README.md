# Benchmark Subcommand

Evaluates a collection of questions against several generation setups and writes one structured JSON result file per (dataset, question, setup). It reuses the exact same generation pipeline as the [`generation`](../generation/README.md) subcommand, so any setup can itself be Knowledge-Augmented. Runs are parallel and resumable.

## How it works

For every question in every dataset, the runner executes each applicable setup and persists the outcome:

```
<output>/<dataset>/<question>/<prefix><setup>.json
```

Each result is written atomically (to a `.tmp` file that is renamed on success), so an interrupted run never leaves a half-written file. With `--continue`, setups whose result file already exists and is non-empty are skipped, letting you resume a long run without redoing finished work.

A setup may restrict itself to a subset of datasets via its `datasets` field; when a setup does not list the current dataset, it is skipped for that dataset. Multiple-choice questions expose their options to the prompt through the `{{CHOICE}}` placeholder.

## Usage

```
kag benchmark [OPTIONS] --datasets <DATASETS> --benchmark <BENCHMARK>
```

| Flag | Short | Description | Default |
|---|---|---|---|
| `--config <FILE>` | `-c` | Path to the configuration file | `config.toml` |
| `--datasets <FILE>` | `-d` | JSON file describing the evaluation datasets | *(required)* |
| `--benchmark <FILE>` | `-b` | TOML file describing the setups to compare | *(required)* |
| `--parallel <N>` | `-p` | Number of parallel tasks | `1` |
| `--continue` | | Resume an interrupted run, preserving existing non-empty result files instead of overwriting them | *(disabled)* |
| `--output <DIR>` | `-o` | Root directory for result files | `.` (current directory) |
| `--prefix <STRING>` | | Prefix prepended to each result filename | *(empty)* |
| `--skip <SETUP1,SETUP2>` | | List of setups that should not be performed (separated by commas) | *(empty)* |

The global `--config` file is still used to resolve the providers, databases, and retrievers that setups reference by name.

## Datasets file

A JSON file shaped as a nested `dataset → question → entry` map. Dataset and question keys must be valid component names (lowercase letters, digits, hyphens). Each entry has an `input` and an optional `output`:

```json
{
  "free-form": {
    "0": {
      "input": "How is Rheumatoid Arthritis classified, and what are its finding sites?"
    }
  },
  "mcq-form": {
    "0": {
      "input": "Which vitamin deficiency is most commonly associated with scurvy?",
      "output": {
        "type": "MCQ",
        "options": { "a": "Vitamin A", "b": "Vitamin C", "c": "Vitamin D", "d": "Vitamin K" }
      }
    }
  }
}
```

When `output` is omitted the question is treated as free-form. For an `MCQ` entry, the `options` map is flattened into a newline-separated list and injected into the prompt as `{{CHOICE}}`. An optional `answer` field (the identifier of the correct option) may be added; it is ignored at benchmark time but is consumed by the [`stats`](../stats/README.md) subcommand to grade the run.

See [`examples/dataset/dataset.sample.json`](../../../examples/dataset/dataset.sample.json).

## Benchmark file

A TOML file shaped as a `setup → entry` map, rendered with MiniJinja before parsing (so the `file()` and `env()` helpers are available). Each entry carries the same generation parameters as the `generation` subcommand, a `system_prompt`, and an optional dataset filter:

```toml
[qwen35-9b-ollama-native]
provider = "ollama"
model = "qwen3.5:9b"
temperature = 0.0
tokens = 10000
system_prompt = '''{{ file('examples/prompt/umls_prompt.md') }}'''
# datasets = ["free-form"]   # optional: restrict this setup to specific datasets
```

To make a setup Knowledge-Augmented, add a `retriever` and a `database` (both referencing components from the main configuration), exactly as on the command line. The setup's `system_prompt` may use `{{INPUT}}`, `{{RETRIEVAL}}`, and `{{CHOICE}}`.

See [`examples/config/benchmark.sample.toml`](../../../examples/config/benchmark.sample.toml).

## Result files

Each result file contains either the successful generation output (the produced text plus retrieval/generation statistics and the exact configuration used) or, on failure, the error message — tagged so both cases parse uniformly:

```json
{ "ok": { "result": "…", "stats": { "…": "…" } } }
```

```json
{ "error": "The requested retriever 'neo4j-…' is missing from the configuration." }
```

A failed task is logged and recorded, but does not stop the rest of the run.

## Examples

Run four tasks in parallel, writing results under `./results`:

```bash
kag benchmark \
  --datasets examples/dataset/dataset.sample.json \
  --benchmark examples/config/benchmark.sample.toml \
  --parallel 4 \
  --output ./results
```

Resume after an interruption without redoing finished work, and tag the filenames:

```bash
kag benchmark \
  --datasets examples/dataset/dataset.sample.json \
  --benchmark examples/config/benchmark.sample.toml \
  --output ./results \
  --prefix run1- \
  --continue
```

See the [project README](../../../README.md) for configuration and component details.
