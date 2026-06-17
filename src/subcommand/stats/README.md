# Stats Subcommand

Grades the result tree produced by the [`benchmark`](../benchmark/README.md) subcommand against the ground truth in the datasets file, then reports accuracy, precision, and coverage **per setup**. It reads only files already on disk — no model or database is contacted — so scoring is fast, deterministic, and repeatable. The report is rendered either as an aligned, human-readable table or as a machine-readable JSON document.

## How it works

The scorer walks the same directory layout the benchmark writes:

```
<results>/<dataset>/<question>/<prefix><setup>.json
```

For every result file it recovers the **setup** name from the filename (stripping `--prefix` and the `.json` extension), looks up the matching **ground truth** for that `<dataset>/<question>`, and grades the recorded generation output:

1. **Resolve ground truth.** The question's `output` is read from the datasets file. Only multiple-choice (`MCQ`) entries that declare an `answer` are scorable. An `MCQ` entry without an `answer` makes every result file under that question count as *skipped*; a free-form entry (no `output`) is ignored silently.
2. **Grade each file.** A successful result whose generation output equals the expected `answer` is **correct**; a successful result that does not match is **incorrect**; a result file that recorded a benchmark execution error is counted as an **error**.
3. **Aggregate.** Outcomes are accumulated per setup, both overall and broken down per dataset, alongside the mean generation time and — when the setup used a retriever — the mean retrieval time and the mean number of vertices, relationships, and properties pulled from the graph.
4. **Rank and render.** Setups are ordered by descending accuracy (ties broken by name) and printed as a table or serialized as JSON.

Because grading is an exact comparison between the model's output and the expected option identifier, a scorable setup is expected to emit just the option key (e.g. `b`) — typically enforced through its system prompt. Files that do not match the `<prefix>…json` naming scheme, or that fail to parse, are reported in the logs and skipped.

## Usage

```
kag stats [OPTIONS] --datasets <DATASETS>
```

| Flag | Short | Description | Default |
|---|---|---|---|
| `--datasets <FILE>` | `-d` | JSON datasets file used as ground truth — the same file passed to `benchmark` | *(required)* |
| `--results <DIR>` | `-r` | Root directory holding the benchmark result files; must match the benchmark's `--output` | `.` (current directory) |
| `--prefix <STRING>` | | Filename prefix that was prepended by the benchmark's `--prefix`, so the setup name can be recovered | *(empty)* |
| `--format <FORMAT>` | | Output format: `text` or `json` | `text` |

The global `--config` file is **not** consulted here; scoring depends only on the datasets file and the result tree.

## Ground truth

A question is graded only when its datasets entry is an `MCQ` that declares the correct option through an `answer` field. This is the same datasets file used by `benchmark`, with the `answer` added:

```json
{
  "mcq-form": {
    "0": {
      "input": "Which vitamin deficiency is most commonly associated with scurvy?",
      "output": {
        "type": "MCQ",
        "options": { "a": "Vitamin A", "b": "Vitamin C", "c": "Vitamin D", "d": "Vitamin K" },
        "answer": "b"
      }
    }
  }
}
```

| Datasets entry | Effect on scoring |
|---|---|
| `MCQ` with `answer` | Scored — each result file is graded correct / incorrect / error |
| `MCQ` without `answer` | Not scored — every result file under the question is counted as *skipped* |
| No `output` (free-form) | Ignored — not scorable and not counted |

See [`examples/dataset/dataset.sample.json`](../../../examples/dataset/dataset.sample.json) for the base format and the [benchmark README](../benchmark/README.md#datasets-file) for the full schema.

## Metrics

For each setup (and for each dataset within it) the following are computed:

| Metric | Definition | Notes |
|---|---|---|
| **OK** | correct | Output matched the expected option |
| **WRONG** | incorrect | Successful run, output did not match |
| **ERR** | errors | Result file recorded an execution error |
| **TOT** | correct + incorrect + errors | Total scored files |
| **ACC%** | correct / total | Accuracy over everything scored |
| **PREC%** | correct / committed | Precision over non-error results (committed = correct + incorrect) |
| **COVER%** | committed / total | Share of scored files that produced a usable answer |
| **GEN(s)** | mean generation time | Averaged over successful files, in seconds |
| **RET(s)** | mean retrieval time | Present only for KAG setups |
| **VERTICE / RELATION / PROPERTY** | mean graph elements retrieved | Mean vertices, relationships, and properties per question; KAG setups only |

A dash (`-`) is shown wherever a metric has no samples (except RAG metrics).

## Output formats

### Text (default)

An aligned table, one block per setup ranked by accuracy, with a `└` sub-row per dataset:

```
Datasets: 2 | Scorable questions: 1 | Skipped files: 0

SETUP                    ACC%    PREC%    COVER%     OK  WRONG   ERR  TOT    GEN(s)    RET(s)  VERTICE  RELATION  PROPERTY
qwen35-9b-kag           80.00    88.89     90.00      8      1     1   10     1.732     0.214    12.40     18.60     41.20
  └ mcq-form            80.00    88.89     90.00      8      1     1   10     1.732     0.214    12.40     18.60     41.20
qwen35-9b-native        60.00    66.67     90.00      6      3     1   10     0.918                                       

ACC% = correct / scored, PREC% = correct / committed, COVER% = committed / scored
GEN(s)/RET(s) = mean generation/retrieval time per question, in seconds
```

### JSON

A pretty-printed document with the same numbers plus the raw counts and totals, suitable for further processing:

```json
{
  "datasets": 2,
  "scorable_questions": 1,
  "skipped_files": 0,
  "setups": [
    {
      "setup": "qwen35-9b-kag",
      "overall": {
        "total": 10, "correct": 8, "incorrect": 1, "errors": 1, "parsed": 9,
        "accuracy": 0.8, "precision": 0.888, "coverage": 0.9,
        "avg_generation_secs": 1.732, "total_generation_secs": 15.59,
        "retrieval": {
          "time_secs": 1.93, "vertices": 124, "relationships": 186, "properties": 412, "samples": 9,
          "avg_time_secs": 0.214, "avg_vertices": 12.4, "avg_relationships": 18.6, "avg_properties": 41.2
        }
      },
      "per_dataset": { "mcq-form": { "...": "..." } }
    }
  ]
}
```

The top-level `scorable_questions` counts the `MCQ` ground-truth questions, and `skipped_files` counts the result files that were ignored for lack of a usable `answer`.

## Examples

Score a run sitting in `./results`, printing the table:

```bash
kag stats \
  --datasets examples/dataset/dataset.sample.json \
  --results ./results
```

Score a prefixed run and emit JSON for a report pipeline:

```bash
kag stats \
  --datasets examples/dataset/dataset.sample.json \
  --results ./results \
  --prefix run1- \
  --format json > report.json
```

## Notes

- Scoring never mutates the result tree; it can be re-run as often as needed and is safe to run while a benchmark is still in progress (only finished files are graded).
- The `--prefix` here must match the benchmark's `--prefix`, otherwise setup names cannot be recovered and matching files are skipped.
- Increasing log verbosity with `-v`/`-vv` traces every per-file grading decision and reports files that were ignored or failed to parse.

See the [project README](../../../README.md) for configuration and the [benchmark README](../benchmark/README.md) for how the result tree is produced.
