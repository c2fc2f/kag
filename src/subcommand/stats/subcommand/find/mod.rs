//! `stats find` subcommand: regex search over benchmark results
//!
//! Walks the result tree produced by the
//! [`benchmark`](crate::subcommand::benchmark) subcommand and keeps every
//! result file whose model response validates a user-supplied regular
//! expression. In text format, the matching responses are handed to an
//! interactive `fzf` prompt so they can be browsed and fuzzy-searched; once
//! the prompt is closed, the match count is printed. In JSON format, the
//! match count and the paths of every matching file are printed as a JSON
//! document instead.

use std::{
  io::Write,
  path::{Path, PathBuf},
  process::{ExitCode, Stdio},
};

use anyhow::Context;
use log::warn;
use regex::Regex;
use serde::Serialize;

use crate::{
  cli::stats::{Args, Format, find},
  generation, match_err,
  subcommand::benchmark,
};

/// A single result file whose model response validated the search regex
struct Match {
  /// Path to the result file on disk
  path: PathBuf,
  /// The model's response text, as extracted from the result file
  response: String,
}

/// Recursively walks `dir`, invoking `visit` once for every regular file
/// found, regardless of depth.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if a directory in the tree cannot be read.
fn walk<F: FnMut(&Path)>(dir: &Path, visit: &mut F) -> std::io::Result<()> {
  for entry in std::fs::read_dir(dir)? {
    let entry = entry?;
    let path = entry.path();
    if entry.file_type()?.is_dir() {
      walk(&path, visit)?;
    } else {
      visit(&path);
    }
  }
  Ok(())
}

/// Walks the result tree rooted at `root`, keeping the result files whose
/// name matches the `<prefix>*.json` scheme and whose model response
/// validates `regex`.
///
/// Returns the collected matches alongside the total number of result files
/// that were inspected (regardless of whether they matched).
///
/// # Errors
///
/// Returns an [`std::io::Error`] if a directory in the result tree cannot be
/// read.
fn collect_matches(
  root: &Path,
  prefix: &str,
  regex: &Regex,
) -> std::io::Result<(Vec<Match>, usize)> {
  let mut matches = Vec::new();
  let mut total = 0;

  walk(root, &mut |path| {
    let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
      return;
    };
    if !file_name.ends_with(".json") || !file_name.starts_with(prefix) {
      return;
    }

    total += 1;

    let bytes = match std::fs::read(path) {
      Ok(bytes) => bytes,
      Err(e) => {
        warn!("Failed to read result file {path:?}: {e:#}");
        return;
      }
    };

    let parsed: benchmark::result::Result<generation::config::Output> =
      match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(e) => {
          warn!("Failed to parse result file {path:?}: {e:#}");
          return;
        }
      };

    let benchmark::result::Result::Ok(output) = parsed else {
      return;
    };

    if regex.is_match(output.result.trim()) {
      matches.push(Match {
        path: path.to_path_buf(),
        response: output.result,
      });
    }
  })?;

  Ok((matches, total))
}

/// Opens an interactive `fzf` prompt over the matching responses, letting the
/// user browse and fuzzy-search them.
///
/// Each candidate line carries the result file's path as a hidden first
/// field (hidden via `--with-nth`), which the preview window reuses to `cat`
/// the full raw result file next to the flattened response text. Embedded
/// newlines in the response are flattened so every match stays a single
/// selectable line.
///
/// # Errors
///
/// Returns an error if `fzf` cannot be spawned (for example, if it is not
/// installed) or if writing the candidates to its standard input fails.
fn run_fzf(matches: &[Match]) -> anyhow::Result<()> {
  let mut child = std::process::Command::new("fzf")
    .args([
      "--delimiter",
      "\t",
      "--with-nth",
      "2..",
      "--preview",
      "cat {1}",
    ])
    .stdin(Stdio::piped())
    .spawn()
    .context("Failed to spawn `fzf`; is it installed and on your PATH?")?;

  let mut stdin = child.stdin.take().expect("stdin was piped");
  for m in matches {
    let flattened = m.response.replace('\n', " \u{23ce} ");
    if let Err(e) = writeln!(stdin, "{}\t{flattened}", m.path.display()) {
      // The user may quit `fzf` before the full candidate list has been
      // streamed in; the resulting broken pipe is expected, not a failure.
      if e.kind() == std::io::ErrorKind::BrokenPipe {
        break;
      }
      return Err(e)
        .context("Failed to write candidates to `fzf`'s standard input");
    }
  }
  drop(stdin);

  child
    .wait()
    .context("Failed to wait on the `fzf` process")?;

  Ok(())
}

/// The JSON report produced by the `find` subcommand
#[derive(Serialize)]
struct FindReport {
  /// Number of result files whose response validated the regex
  matches: usize,
  /// Number of total result files that were inspected
  total: usize,
  /// Paths to every result file that validated the regex
  paths: Vec<PathBuf>,
}

/// Executes the `find` subcommand.
///
/// # Arguments
///
/// * `args` - The shared statistics arguments, used here for the result
///   directory, the filename prefix, and the output format.
/// * `sargs` - The `find`-specific arguments, holding the search regex.
///
/// # Returns
///
/// Returns `ExitCode::SUCCESS` once the search has been rendered, or
/// `ExitCode::FAILURE` if the result tree cannot be walked, `fzf` cannot be
/// run, or the JSON report cannot be serialized.
pub fn run(args: Args, sargs: find::Args) -> ExitCode {
  let (matches, total) = match_err!(
    collect_matches(&args.results, &args.prefix, &sargs.regex),
    "Failed to walk the results directory ({:?})",
    args.results
  );

  match args.format {
    Format::Text => {
      if matches.is_empty() {
        eprintln!("No result matched the pattern.");
        return ExitCode::SUCCESS;
      }

      match_err!(run_fzf(&matches), "Failed to run `fzf`");

      eprintln!(
        "\n{} occurrence(s) out of {total} result file(s) matched the \
         pattern.",
        matches.len()
      );
    }
    Format::Json => {
      let report = FindReport {
        matches: matches.len(),
        total,
        paths: matches.into_iter().map(|m| m.path).collect(),
      };
      let json = match_err!(
        serde_json::to_string_pretty(&report),
        "Failed to serialize the report as JSON"
      );
      println!("{json}");
    }
  }

  ExitCode::SUCCESS
}
