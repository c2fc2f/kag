#![allow(missing_docs)]

#[path = "src/cli/mod.rs"]
mod cli;

use clap::CommandFactory;

/// Generates man pages for the CLI
fn generate_man_pages() -> std::io::Result<()> {
  let out_dir = std::path::PathBuf::from(
    std::env::var_os("OUT_DIR").ok_or(std::io::ErrorKind::NotFound)?,
  );

  let cmd = cli::Args::command();
  let name = cmd.get_name();

  // main man page
  let mut buffer = Vec::default();
  clap_mangen::Man::new(cmd.clone()).render(&mut buffer)?;
  std::fs::write(out_dir.join(format!("{name}.1")), buffer)?;

  // subcommand man pages
  for subcmd in cmd.get_subcommands().filter(|c| !c.is_hide_set()) {
    let subcmd_name = format!("{name}-{}", subcmd.get_name());
    let subcmd = subcmd.clone().name(&subcmd_name);

    let mut buffer = Vec::default();

    clap_mangen::Man::new(subcmd)
      .title(subcmd_name.to_uppercase())
      .render(&mut buffer)?;

    std::fs::write(out_dir.join(subcmd_name + ".1"), buffer)?;
  }

  Ok(())
}

fn main() -> std::io::Result<()> {
  if let Ok(val) = std::env::var("NIX_RELEASE_VERSION") {
    println!("cargo:rustc-env=CARGO_PKG_VERSION={val}");
  }
  println!("cargo:rerun-if-env-changed=NIX_RELEASE_VERSION");

  println!("cargo:rerun-if-changed=src/cli");
  generate_man_pages()
}
