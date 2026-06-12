//! Benchmark execution engine module
//!
//! This module coordinates the execution of benchmark setups against loaded
//! evaluation datasets. It manages command-line arguments parsing, runtime
//! orchestration via Tokio, concurrent task scheduling, and outputs the
//! final evaluated metrics into structured JSON result files

mod config;
mod dataset;
mod result;

use std::{
  borrow::Cow,
  num::NonZero,
  ops::Deref,
  path::{Path, PathBuf},
  process::ExitCode,
  sync::Arc,
};

use futures::{StreamExt, stream};
use log::{debug, error, info, trace};
use tokio::{
  fs::{OpenOptions, create_dir_all, metadata, try_exists},
  io::AsyncWriteExt,
};

use crate::{
  config::{ComponentName, Config, load_config},
  match_err,
  subcommand::benchmark::{
    config::Benchmark,
    dataset::{DatasetEntry, Datasets, Output},
  },
};

/// Command-line arguments for the benchmark execution
#[derive(clap::Args, Debug)]
pub struct Args {
  /// Path to the JSON file containing the collection of evaluation datasets
  #[arg(short, long)]
  pub datasets: PathBuf,

  /// Path to the configuration file defining the different techniques and
  /// models to benchmark
  #[arg(short, long)]
  pub benchmark: PathBuf,

  /// Number of parallel tasks to use for the benchmark execution
  #[arg(short, long, default_value = "1")]
  pub parallel: NonZero<usize>,

  /// Flag to resume a previously interrupted benchmark run, preserving
  /// existing result files instead of overwriting them
  #[arg(short, long, action)]
  pub r#continue: bool,

  /// Directory path where the generated benchmark result files will be saved
  #[arg(short, long, default_value = ".")]
  pub output: PathBuf,

  /// Optional naming prefix to append to the generated JSON result filenames
  #[arg(long, default_value = "")]
  pub prefix: String,
}

/// Executes the benchmark lifecycle with the provided arguments and system
/// configuration
///
/// # Arguments
///
/// * `args` - The parsed command-line arguments containing paths and runtime
///   constraints
/// * `config` - The global system configuration containing component
///   definitions
///
/// # Returns
///
/// Returns `ExitCode::SUCCESS` if the orchestrator finishes its loop,
/// regardless of individual task failures (which are logged as errors)
pub fn run(args: Args, config: Config) -> ExitCode {
  let rt = tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .worker_threads(args.parallel.get())
    .build()
    .expect("Failed building the Runtime");

  let datasets: Datasets = match_err!(
    serde_json::from_reader(match_err!(
      std::fs::File::open(&args.datasets),
      "Failed to open the datasets ({:?}) file",
      args.datasets
    )),
    "Failed to parse the datasets"
  );
  debug!("Datasets successfully loaded: {datasets:?}");
  let benchmark: Arc<Benchmark> = Arc::new(match_err!(
    load_config(args.benchmark),
    "Unable to load the benchmark configuration file"
  ));
  debug!("Final benchmark configuration: {benchmark:?}");

  let config = Arc::new(config);
  let output = Arc::new(args.output);
  let prefix = Arc::new(args.prefix);

  rt.block_on(async {
    for (dname, dataset) in datasets.0 {
      info!("Starting dataset: {dname}");

      let dname = Arc::new(dname);

      stream::iter(dataset)
        .for_each_concurrent(args.parallel.get(), |(qname, question)| {
          let config = Arc::clone(&config);
          let dname = Arc::clone(&dname);
          let benchmark = Arc::clone(&benchmark);
          let output = Arc::clone(&output);
          let prefix = Arc::clone(&prefix);

          async move {
            debug!("Starting query '{qname}' in dataset '{dname}'");

            match execute_benchmark(
              config.as_ref(),
              dname.as_ref(),
              &qname,
              &question,
              benchmark.as_ref(),
              &output,
              &prefix,
              !args.r#continue,
            )
            .await
            {
              Ok(_) => debug!("Task finished successfully"),
              Err(e) => error!("Task ({dname}/{qname}) failed: {e:#}"),
            }
          }
        })
        .await;

      info!("Finished dataset: {dname}");
    }
  });

  ExitCode::SUCCESS
}

/// Runs all applicable benchmark setups for a specific dataset entry and
/// writes the output
///
/// # Arguments
///
/// * `config` - Reference to the system configuration definitions
/// * `dname` - The identifier name of the current dataset
/// * `qname` - The identifier name of the current question/query
/// * `question` - The actual dataset item containing input prompts
/// * `benchmark` - Reference to the loaded benchmark setups
/// * `output` - Root path where results should be persisted
/// * `prefix` - String slice to prefix the JSON filenames
/// * `override` - If `true`, overwrites existing files; if `false`, skips
///   processing when files exist
///
/// # Errors
///
/// Returns a `std::io::Result<()>` if directory creation, file opening, or
/// writing fails
#[allow(clippy::too_many_arguments)]
async fn execute_benchmark(
  config: &Config,
  dname: &ComponentName,
  qname: &ComponentName,
  question: &DatasetEntry,
  benchmark: &Benchmark,
  output: &Path,
  prefix: &str,
  r#override: bool,
) -> std::io::Result<()> {
  info!(
    "Starting benchmark execution [Dataset: {}, Question: {}]",
    dname, qname
  );

  let mut base_p = output.join(dname.deref());
  base_p.push(qname.deref());

  create_dir_all(&base_p).await?;

  for (sname, setup) in benchmark.as_ref() {
    if let Some(dataset) = &setup.datasets
      && dataset.contains(dname)
    {
      trace!(
        "\
          [Dataset: {}] Skipping setup '{}' for question '{}': dataset is \
          explicitly excluded\
        ",
        dname, sname, qname
      );
      continue;
    }

    let path = base_p.join(format!("{prefix}{sname}.json"));

    if !r#override
      && try_exists(&path).await?
      && metadata(&path).await?.len() > 0
    {
      continue;
    }

    let mut f = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(path)
      .await?;

    debug!(
      "[Dataset: {}] Generating configuration for setup '{}' on question '{}'",
      dname, sname, qname
    );

    let system_prompt: Cow<str> = match &question.output {
      None => {
        if setup.system_prompt.contains("{{CHOICE}}") {
          Cow::from(setup.system_prompt.replace("{{CHOICE}}", ""))
        } else {
          Cow::from(&setup.system_prompt)
        }
      }
      Some(Output::Mcq(choices)) => {
        let choices = choices.iter().fold(String::new(), |mut acc, (k, v)| {
          if !acc.is_empty() {
            acc.push('\n')
          }
          acc.push_str(k);
          acc.push_str(": ");
          acc.push_str(v);

          acc
        });

        if setup.system_prompt.contains("{{CHOICE}}") {
          Cow::from(setup.system_prompt.replace("{{CHOICE}}", &choices))
        } else {
          Cow::from(format!("{}\n\nChoice:\n{}", setup.system_prompt, choices))
        }
      }
    };

    let response: result::Result<_> = setup
      .config
      .generate(config, Some(system_prompt), &question.input)
      .await
      .map_err(|e| {
        error!(
          "\
            [Dataset: {}] Failed to generate configuration for setup '{}' on \
            question '{}'. Error: {:?}\
          ",
          dname, sname, qname, e
        );
        e
      })
      .into();

    let json_bytes = serde_json::to_vec_pretty(&response)?;
    f.write_all(&json_bytes).await?;
    f.flush().await?;

    trace!(
      "[Dataset: {}] Successfully completed setup '{}' for question '{}'",
      dname, sname, qname
    );
  }

  info!(
    "Successfully finished benchmark execution [Dataset: {}, Question: {}]",
    dname, qname
  );
  Ok(())
}

