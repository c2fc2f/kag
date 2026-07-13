//! Benchmark execution engine module
//!
//! This module coordinates the execution of benchmark setups against loaded
//! evaluation datasets. It manages command-line arguments parsing, runtime
//! orchestration via Tokio, parallel task scheduling, and outputs the
//! final evaluated metrics into structured JSON result files

pub mod config;
pub mod dataset;
pub mod result;

use std::{ops::Deref, path::Path, process::ExitCode, sync::Arc};

use futures::{StreamExt, stream};
use log::{debug, error, info, trace};
use minijinja::Environment;
use tokio::{
  fs::{OpenOptions, create_dir_all, metadata, rename, try_exists},
  io::AsyncWriteExt,
};

use crate::{
  cli::{benchmark::Args, component::ComponentName},
  config::{Config, load_config},
  match_err,
  subcommand::benchmark::{
    self,
    config::Benchmark,
    dataset::{DatasetEntry, Datasets, Output},
  },
};

/// Executes the benchmark lifecycle with the provided arguments and system
/// configuration
///
/// # Arguments
///
/// * `args` - The parsed command-line arguments containing paths and runtime
///   constraints
///
/// # Returns
///
/// Returns `ExitCode::SUCCESS` if the orchestrator finishes its loop,
/// regardless of individual task failures (which are logged as errors)
pub fn run(args: Args) -> ExitCode {
  let rt = tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .worker_threads(args.parallel.get())
    .build()
    .expect("Failed building the Runtime");

  let config: Config = match_err!(
    load_config(args.config),
    "Unable to load the configuration file"
  );

  debug!("Final configuration: {config:?}");

  let _ = rustls::crypto::ring::default_provider().install_default();

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
        .map(|(qname, question)| {
          let config = Arc::clone(&config);
          let dname = Arc::clone(&dname);
          let benchmark = Arc::clone(&benchmark);
          let output = Arc::clone(&output);
          let prefix = Arc::clone(&prefix);
          // one task per query -> schedulable on any worker thread
          tokio::spawn(async move {
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
          })
        })
        .buffer_unordered(args.parallel.get())
        .for_each(|joined| async {
          if let Err(e) = joined {
            error!("Task panicked or was cancelled: {e}");
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
/// * `overwrite` - If `true`, overwrites existing files; if `false`, skips
///   processing when files exist
///
/// # Errors
///
/// Returns a `std::io::Result<()>` if directory creation, file opening, or
/// writing fails
#[allow(clippy::too_many_arguments)]
async fn execute_benchmark(
  config: &Config<'_>,
  dname: &ComponentName,
  qname: &ComponentName,
  question: &DatasetEntry,
  benchmark: &Benchmark,
  output: &Path,
  prefix: &str,
  overwrite: bool,
) -> std::io::Result<()> {
  info!(
    "Starting benchmark execution [Dataset: {}, Question: {}]",
    dname, qname
  );

  let mut base_p = output.join(dname.deref());
  base_p.push(qname.deref());

  let mut dir_created = false;

  for (sname, setup) in benchmark.as_ref() {
    if let Some(dataset) = &setup.datasets
      && !dataset.contains(dname)
    {
      trace!(
        "\
          [Dataset: {}] Skipping setup '{}' for question '{}': dataset is \
          not included in the allowed list\
        ",
        dname, sname, qname
      );
      continue;
    }

    let path = base_p.join(format!("{prefix}{sname}.json"));

    if !overwrite
      && try_exists(&path).await?
      && metadata(&path).await?.len() > 0
    {
      continue;
    }

    if !dir_created {
      create_dir_all(&base_p).await?;
      dir_created = true;
    }

    let tmp_path = path.with_added_extension("tmp");

    let mut f = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(&tmp_path)
      .await?;

    debug!(
      "[Dataset: {}] Generating configuration for setup '{}' on question '{}'",
      dname, sname, qname
    );

    let mut env = Environment::new();

    match &question.output {
      None => (),
      Some(Output::Mcq { options, answer: _ }) => {
        let choices = options.iter().fold(String::new(), |mut acc, (k, v)| {
          if !acc.is_empty() {
            acc.push('\n')
          }
          acc.push_str(k);
          acc.push_str(": ");
          acc.push_str(v);

          acc
        });
        env.add_global("CHOICE", choices);
      }
    };

    let response: benchmark::result::Result<_> = setup
      .config
      .generate(
        config,
        Some(setup.system_prompt.as_str()),
        &question.input,
        env,
      )
      .await
      .map_err(|e| {
        error!(
          "\
            [Dataset: {}] Failed to generate configuration for setup '{}' on \
            question '{}'. Error: {:#}\
          ",
          dname, sname, qname, e
        );
        e
      })
      .into();

    let json_bytes = serde_json::to_vec_pretty(&response)?;
    f.write_all(&json_bytes).await?;
    f.flush().await?;
    rename(tmp_path, path).await?;

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
