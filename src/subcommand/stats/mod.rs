//! Benchmark scoring engine module
//!
//! This module consumes the result tree produced by the
//! [`benchmark`](super::benchmark) subcommand and grades it against the
//! ground truth held in the datasets file. For every question that defines an
//! `output`, it extracts the option the model committed to from the free-form
//! generation output and compares it against the expected option. The
//! per-setup outcomes are then aggregated into accuracy and precision metrics
//! and rendered either as a human-readable table or as a JSON document.

mod metric;

use std::{
  collections::{BTreeMap, BTreeSet},
  fs::{self, File},
  path::Path,
  process::ExitCode,
  str::FromStr,
  time::Duration,
};

use log::{debug, trace, warn};

use crate::{
  cli::{
    component::ComponentName,
    stats::{Args, Format},
  },
  generation, match_err,
  subcommand::{
    benchmark::{
      self,
      dataset::{self, Datasets},
    },
    stats::metric::{Metrics, Outcome, Report, SetupReport},
  },
};

/// Recovers the setup name from a result filename by stripping the prefix and
/// the `.json` extension.
///
/// Returns [`None`] if the file does not have the `.json` extension, does not
/// start with the expected prefix, or is not a valid filename.
fn setup_name(file_name: &str, prefix: &str) -> Option<ComponentName> {
  file_name
    .strip_suffix(".json")?
    .strip_prefix(prefix)
    .and_then(|s| ComponentName::from_str(s).ok())
}

/// Recovers the setup name from a directory entry, but only when the entry is
/// a regular file whose name matches the `<prefix>…json` result scheme.
///
/// Returns `Ok(None)` for anything that should be ignored (directories,
/// symlinks, files that do not match the naming scheme). This is the single
/// definition of "a gradeable result file", shared by both the grading path
/// and the skipped-file counter so the two never disagree.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the entry's file type cannot be read.
fn result_file_setup(
  entry: &fs::DirEntry,
  prefix: &str,
) -> std::io::Result<Option<ComponentName>> {
  if !entry.file_type()?.is_file() {
    return Ok(None);
  }
  Ok(setup_name(&entry.file_name().to_string_lossy(), prefix))
}

/// Walks the result tree and accumulates per-setup metrics.
///
/// The expected layout is
/// `<root>/<dataset>/<question>/<prefix><setup>.json`. Files that have no
/// matching scorable ground-truth entry are counted as skipped and otherwise
/// ignored.
///
/// # Errors
///
/// Returns an [`std::io::Error`] when a directory in the result tree cannot
/// be read.
#[allow(clippy::type_complexity)]
fn collect_metrics(
  root: &Path,
  prefix: &str,
  datasets: &Datasets,
) -> std::io::Result<(
  BTreeMap<ComponentName, (Metrics, BTreeMap<ComponentName, Metrics>)>,
  BTreeSet<ComponentName>,
  usize,
)> {
  let mut setups: BTreeMap<
    ComponentName,
    (Metrics, BTreeMap<ComponentName, Metrics>),
  > = BTreeMap::new();
  let mut seen_datasets = BTreeSet::new();
  let mut skipped = 0;

  for dataset_entry in fs::read_dir(root)? {
    let dataset_entry = dataset_entry?;
    if !dataset_entry.file_type()?.is_dir() {
      continue;
    }
    let dataset = match ComponentName::from_str(
      dataset_entry.file_name().to_string_lossy().as_ref(),
    ) {
      Ok(name) => name,
      Err(_) => {
        warn!(
          "Ignoring directory with invalid dataset name: {:?}",
          dataset_entry.file_name()
        );
        continue;
      }
    };

    for question_entry in fs::read_dir(dataset_entry.path())? {
      let question_entry = question_entry?;
      if !question_entry.file_type()?.is_dir() {
        continue;
      }
      let question = match ComponentName::from_str(
        question_entry.file_name().to_string_lossy().as_ref(),
      ) {
        Ok(name) => name,
        Err(_) => {
          warn!(
            "Ignoring directory with invalid question name: {:?}",
            question_entry.file_name()
          );
          continue;
        }
      };

      let answer = match datasets
        .0
        .get(&dataset)
        .and_then(|questions| questions.get(&question))
        .and_then(|entry| entry.output.as_ref())
      {
        Some(dataset::Output::Mcq {
          options: _,
          answer: Some(answer),
        }) => answer,
        Some(_) => {
          for file_entry in fs::read_dir(question_entry.path())? {
            let file_entry = file_entry?;
            if result_file_setup(&file_entry, prefix)?.is_some() {
              skipped += 1;
            }
          }
          continue;
        }
        None => {
          continue;
        }
      };

      for file_entry in fs::read_dir(question_entry.path())? {
        let file_entry = file_entry?;
        if !file_entry.file_type()?.is_file() {
          continue;
        }
        let file_name = file_entry.file_name();
        let Some(setup) = result_file_setup(&file_entry, prefix)? else {
          trace!("Ignoring non-result entry: {:?}", file_entry.file_name());
          continue;
        };

        let parsed: benchmark::result::Result<generation::config::Output> =
          match serde_json::from_reader(File::open(file_entry.path())?) {
            Ok(parsed) => parsed,
            Err(e) => {
              warn!("Failed to parse result file {file_name:?}: {e:#}");
              continue;
            }
          };

        let outcome = match &parsed {
          benchmark::result::Result::Ok(r) => {
            if r.result.trim().eq_ignore_ascii_case(answer) {
              Outcome::Correct
            } else {
              Outcome::Incorrect
            }
          }
          benchmark::result::Result::Error(_) => Outcome::Error,
        };

        let (timing, retrieval) = match parsed {
          benchmark::result::Result::Error(_) => (None, None),
          benchmark::result::Result::Ok(p) => (
            Some(p.stats.time),
            p.stats.retrieval.map(|r| match r.stats {
              crate::retrieval::Stats::Embedder {
                embedding: _,
                database,
                time,
              } => match database {
                crate::retrieval::database::Stats::Neo4j {
                  vertices,
                  relationships,
                  properties,
                  time: _,
                } => (time, vertices, relationships, properties),
              },
            }),
          ),
        };

        trace!("[{dataset}/{question}] setup '{setup}' graded as {outcome:?}");

        let (overall, per_dataset): &mut (_, _) =
          setups.entry(setup).or_default();

        overall.record(outcome, timing.as_ref(), retrieval.as_ref());
        per_dataset.entry(dataset.clone()).or_default().record(
          outcome,
          timing.as_ref(),
          retrieval.as_ref(),
        );
      }
    }

    seen_datasets.insert(dataset);
  }

  Ok((setups, seen_datasets, skipped))
}

/// Assembles the final [`Report`] from the accumulated per-setup metrics.
fn build_report(
  setups: BTreeMap<ComponentName, (Metrics, BTreeMap<ComponentName, Metrics>)>,
  seen_datasets: &BTreeSet<ComponentName>,
  scorable_questions: usize,
  skipped_files: usize,
) -> Report {
  let mut setup_reports: Vec<SetupReport> = setups
    .into_iter()
    .map(|(setup, (overall, per_dataset))| SetupReport {
      setup,
      overall,
      per_dataset,
    })
    .collect();

  // Order by descending accuracy, then by setup name for a stable ranking.
  setup_reports.sort_by(|a, b| {
    b.overall
      .accuracy()
      .unwrap_or(f64::NEG_INFINITY)
      .total_cmp(&a.overall.accuracy().unwrap_or(f64::NEG_INFINITY))
      .then_with(|| a.setup.cmp(&b.setup))
  });

  Report {
    datasets: seen_datasets.len(),
    scorable_questions,
    skipped_files,
    setups: setup_reports,
  }
}

/// Formats an optional ratio as a percentage, or a dash when absent.
///
/// The returned string is not padded; callers align it through the table's
/// own width specifiers.
fn pct(value: Option<f64>) -> String {
  value.map_or_else(|| "-".to_string(), |v| format!("{:.2}", v * 100.0))
}

/// Formats an optional duration in seconds, or a dash when absent.
///
/// The returned string is not padded; callers align it through the table's
/// own width specifiers.
fn secs(value: Option<Duration>) -> String {
  value.map_or_else(|| "-".to_string(), |d| format!("{:.3}", d.as_secs_f64()))
}

/// Renders the report as an aligned, human-readable table on standard output.
fn render_text(report: &Report) {
  println!(
    "Datasets: {} | Scorable questions: {} | Skipped files: {}\n",
    report.datasets, report.scorable_questions, report.skipped_files
  );

  if report.setups.is_empty() {
    println!("No scorable results were found.");
    return;
  }

  let name_width = report
    .setups
    .iter()
    .map(|s| s.setup.len())
    .max()
    .unwrap_or(5)
    .max(5);

  let any_retrieval =
    report.setups.iter().any(|s| s.overall.retrieval.is_some());

  print!(
    "{:<width$}  {:>7}  {:>7}  {:>8}  {:>5}  {:>5}  {:>4}  {:>3}  {:>8}",
    "SETUP",
    "ACC%",
    "PREC%",
    "COVER%",
    "OK",
    "WRONG",
    "ERR",
    "TOT",
    "GEN(s)",
    width = name_width
  );
  if any_retrieval {
    print!(
      "  {:>8}  {:>7}  {:>8}  {:>8}",
      "RET(s)", "VERTICE", "RELATION", "PROPERTY"
    );
  }
  println!();

  for setup in &report.setups {
    let m = &setup.overall;
    print!(
      "{:<width$}  {:>7}  {:>7}  {:>8}  {:>5}  {:>5}  {:>4}  {:>3}  {:>8}",
      setup.setup,
      pct(m.accuracy()),
      pct(m.precision()),
      pct(m.coverage()),
      m.correct,
      m.incorrect,
      m.errors,
      m.total(),
      secs(m.avg_generation()),
      width = name_width
    );

    if let Some(r) = &m.retrieval {
      print!(
        "  {:>8}  {:>7.2}  {:>8.2}  {:>8.2}",
        secs(Some(r.avg_time())),
        r.avg_vertices(),
        r.avg_relationships(),
        r.avg_properties(),
      );
    }

    println!();

    for (dataset, dm) in &setup.per_dataset {
      print!(
        "  └ {:<width$}  {:>7}  {:>7}  {:>8}  {:>5}  {:>5}  {:>4}  {:>3}  {:>8}",
        dataset,
        pct(dm.accuracy()),
        pct(dm.precision()),
        pct(dm.coverage()),
        dm.correct,
        dm.incorrect,
        dm.errors,
        dm.total(),
        secs(dm.avg_generation()),
        width = name_width.saturating_sub(4)
      );

      if let Some(r) = &dm.retrieval {
        print!(
          "  {:>8}  {:>7.2}  {:>8.2}  {:>8.2}",
          secs(Some(r.avg_time())),
          r.avg_vertices(),
          r.avg_relationships(),
          r.avg_properties(),
        );
      }

      println!();
    }
  }

  println!(
    "\nACC% = correct / scored, PREC% = correct / committed, COVER% = \
     committed / scored\nGEN(s)/RET(s) = mean generation/retrieval time per \
     question, in seconds"
  );
}

/// Executes the statistics/precision computation.
///
/// # Arguments
///
/// * `args` - The parsed command-line arguments selecting the datasets file,
///   the result directory, the filename prefix, and the output format.
///
/// # Returns
///
/// Returns `ExitCode::SUCCESS` once the report has been rendered, or
/// `ExitCode::FAILURE` if the datasets file cannot be loaded, the result tree
/// cannot be walked, or the JSON report cannot be serialized.
pub fn run(args: Args) -> ExitCode {
  let datasets: Datasets = match_err!(
    serde_json::from_reader(match_err!(
      std::fs::File::open(&args.datasets),
      "Failed to open the datasets ({:?}) file",
      args.datasets
    )),
    "Failed to parse the datasets"
  );
  debug!("Datasets successfully loaded for scoring");

  let (setups, seen_datasets, skipped) = match_err!(
    collect_metrics(&args.results, &args.prefix, &datasets),
    "Failed to walk the results directory ({:?})",
    args.results
  );

  let scorable_questions = datasets
    .0
    .values()
    .flat_map(|questions| questions.values())
    .filter(|entry| {
      matches!(
        entry.output,
        Some(dataset::Output::Mcq {
          answer: Some(_),
          ..
        })
      )
    })
    .count();

  let report =
    build_report(setups, &seen_datasets, scorable_questions, skipped);

  match args.format {
    Format::Text => render_text(&report),
    Format::Json => {
      let json = match_err!(
        serde_json::to_string_pretty(&report),
        "Failed to serialize the report as JSON"
      );
      println!("{json}");
    }
  }

  ExitCode::SUCCESS
}
