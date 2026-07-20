# Stats Find Subcommand

Searches the result tree produced by the [`benchmark`](../../../benchmark/README.md) subcommand for every result whose model response validates a regular expression. Unlike [`stats`](../../README.md), it does **not** need a datasets file — it only looks at what the model actually answered, regardless of whether the question is scorable. In text format the matches are handed to an interactive [`fzf`](https://github.com/junegunn/fzf) prompt to browse; in JSON format it simply reports the count and the matching file paths.

## How it works

`find` walks the result tree rooted at `--results`, recursing through every subdirectory:

```
<results>/<dataset>/<question>/<prefix><setup>.json
```

For every entry whose filename matches the `<prefix>…json` scheme:

1. **Count it.** It is added to the total number of inspected result files, whether or not it parses or matches.
2. **Parse it.** Files that fail to parse, or that recorded a benchmark execution error, are skipped (logged at `warn` level) and hold no response to match against.
3. **Match it.** The trimmed model response is tested against `<REGEX>`. A validating file is kept as a **match**, along with its path and its full response text.

## Usage

```
kag stats find [OPTIONS] <REGEX>
```

| Argument / Flag | Short | Description | Default |
|---|---|---|---|
| `<REGEX>` | | Regular expression the (trimmed) model response must validate | *(required)* |
| `--results <DIR>` | `-r` | Root directory holding the benchmark result files; must match the benchmark's `--output` | `.` (current directory) |
| `--prefix <STRING>` | | Filename prefix that was prepended by the benchmark's `--prefix` | *(empty)* |
| `--format <FORMAT>` | | Output format: `text` or `json` | `text` |

`--datasets` is not required for `find`: subcommands negate the top-level required flags of `stats`.

## Output formats

### Text (default)

Requires [`fzf`](https://github.com/junegunn/fzf) to be installed and on `PATH`. Every matching response is streamed to `fzf` as a single, searchable line (embedded newlines are flattened to `⏎`), with a preview pane (`cat`) showing the full, raw result file for the currently highlighted match. Once you quit `fzf` (whether or not you picked a line), the occurrence count is printed:

```
12 occurrence(s) out of 340 result file(s) matched the pattern.
```

If nothing matches, `fzf` is never launched and `No result matched the pattern.` is printed instead.

### JSON

A pretty-printed document with the match count and the path of every matching file, suitable for piping into other tools:

```json
{
  "matches": 12,
  "paths": [
    "results/mcq-form/0/qwen35-9b-kag.json",
    "results/mcq-form/3/qwen35-9b-native.json"
  ]
}
```

## Examples

Browse every response that mentions "I cannot" or "I don't know" across a run:

```bash
kag stats find --results ./results "(?i)i (cannot|don't know)"
```

List, as JSON, every prefixed result file whose response is empty or blank:

```bash
kag stats find \
  --results ./results \
  --prefix run1- \
  --format json \
  '^\s*$' > blanks.json
```

## Notes

- The regular expression follows the [`regex`](https://docs.rs/regex) crate syntax (a Rust-flavored subset of PCRE); remember to quote it so your shell does not expand it.
- Increasing log verbosity with `-v`/`-vv` reports every result file that failed to read or parse.
- Unlike `stats`, `find` does not distinguish datasets, questions, or ground truth — it only inspects the raw model response, so it also surfaces free-form (non-`MCQ`) results that plain `stats` would skip.

See the [stats README](../../README.md) for grading benchmark results against ground truth, and the [project README](../../../../../README.md) for configuration.
