{
  version,
  lib,
  installShellFiles,
  rustPlatform,
  buildFeatures ? [ ],
}:

rustPlatform.buildRustPackage {
  pname = "kag";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../src
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

  meta = with lib; {
    description = "A comprehensive toolkit for Knowledge Graph Enhanced Retrieval-Augmented Generation (KAG), featuring generation pipelines and evaluation benchmarks";
    mainProgram = "kag";
    homepage = "https://github.com/c2fc2f/KAG";
    license = licenses.mit;
    maintainers = [ maintainers.c2fc2f ];
  };
}
