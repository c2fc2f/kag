//! A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented
//! Generation (KAG), featuring generation pipelines and evaluation benchmarks

mod config;
mod subcommand;

use std::{fs, io::Write, process::ExitCode};

use clap::{Parser, Subcommand};
use clap_verbosity_flag::Verbosity;
use log::{LevelFilter, debug, error};
use minijinja::Environment;

use crate::{config::Config, subcommand::generation};

/// A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented
/// Generation (KAG), featuring generation pipelines and evaluation benchmarks
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The specific operation to perform with the toolkit
    #[command(subcommand)]
    command: Command,

    /// Path to the config file
    #[arg(short, long, global = true, default_value = "config.toml")]
    config: std::path::PathBuf,

    /// Control the output verbosity (-v, -q)
    #[command(flatten)]
    verbosity: Verbosity,
}

/// List of available subcommands in the toolkit
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Command {
    /// Generate text using a specified model, with optional
    /// Knowledge-Augmented Generation (KAG).
    ///
    /// This subcommand runs either a standard text generation workflow or an
    /// augmented generation workflow (KAG/RAG) when a retriever component is
    /// provided.
    Generation(generation::Args),
}

fn main() -> ExitCode {
    let args = Args::parse();

    let log_level = args.verbosity.log_level_filter();
    let mut builder = env_logger::Builder::new();
    builder.filter_level(log_level);
    builder.format(move |buf, record| {
        if log_level > LevelFilter::Error {
            writeln!(
                buf,
                "[{timestamp} {level} {target}] {message}",
                timestamp = buf.timestamp(),
                level = record.level(),
                target = record.target(),
                message = record.args()
            )
        } else {
            writeln!(
                buf,
                "{level}: {message}",
                level = record.level().to_string().to_lowercase(),
                message = record.args()
            )
        }
    });
    builder.init();

    let config = match fs::read_to_string(args.config) {
        Ok(c) => c,
        Err(e) => {
            error!("The configuration file could not be read: {e:#}.");
            return ExitCode::FAILURE;
        }
    };

    let mut env = Environment::new();
    env.add_function("file", |f: String| {
        fs::read_to_string(&f).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("The file {f} could not be read: {e:#}."),
            )
        })
    });
    env.add_function("env", |e: String| std::env::var(&e).unwrap_or_default());

    let config = match env.render_str(&config, minijinja::context!()) {
        Ok(c) => c,
        Err(e) => {
            error!(
                "The special syntax in the configuration file failed to render: {e:#}."
            );
            return ExitCode::FAILURE;
        }
    };

    let config: Config = match toml::from_str(&config) {
        Ok(c) => c,
        Err(e) => {
            error!(
                "The configuration file could not be parsed as valid TOML: {e:#}."
            );
            return ExitCode::FAILURE;
        }
    };

    debug!("Final Config: {config:#?}");

    match args.command {
        Command::Generation(args) => generation::run(args, config),
    }
}
