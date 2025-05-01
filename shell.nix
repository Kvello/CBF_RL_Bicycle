{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.11") {} }:
let
  python = pkgs.python39;
  fetchurl = pkgs.fetchurl;
  fetchgit = pkgs.fetchgit;
  fetchPypi = python.fetchPypi;
  fetchhg = pkgs.fetchhg;
  mujoco210 = import ./mujoco210.nix { 
    inherit (pkgs) lib stdenv fetchurl unzip;
  };
  # pythonPackages = import ./python-packages.nix {
  #   inherit pkgs fetchurl fetchgit fetchhg fetchPypi mujoco210;
  # } python python;
  pythonEnv = import ./python-packages.nix {
    inherit pkgs python;
  };
in
pkgs.mkShell {
  buildInputs = [ pythonEnv ];

  shellHook = ''
    export PYTHONPATH=$PWD
    echo "Python environment ready. Run 'pytest tests/'"
  '';
}
