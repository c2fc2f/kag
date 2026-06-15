{
  version,
  lib,
  installShellFiles,
  rustPlatform,
  buildFeatures ? [ ],
}:

rustPlatform.buildRustPackage rec {
  pname = "kag";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../src
      ../build.rs
      ../Cargo.lock
      ../Cargo.toml
    ];
  };

  inherit buildFeatures;
  inherit version;

  # inject version from nix into the build
  env.NIX_RELEASE_VERSION = version;

  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    installShellFiles

    rustPlatform.bindgenHook
  ];

  buildInputs = [ ];

  postInstall = ''
    installShellCompletion --cmd ${meta.mainProgram} \
      --bash <($out/bin/${meta.mainProgram} completion bash) \
      --fish <($out/bin/${meta.mainProgram} completion fish) \
      --zsh <($out/bin/${meta.mainProgram} completion zsh)

    installManPage target/man/*
  '';

  meta = with lib; {
    description = "A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented Generation (KAG), featuring generation pipelines and evaluation benchmarks";
    mainProgram = "kag";
    homepage = "https://github.com/c2fc2f/kag";
    license = licenses.mit;
    maintainers = [ maintainers.c2fc2f ];
  };
}
