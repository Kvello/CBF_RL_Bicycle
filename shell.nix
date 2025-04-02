{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.11") {} }:
let
  python = pkgs.python311;
in
pkgs.mkShellNoCC {
  packages = with pkgs; [
    (python.withPackages (ps: with ps; [
      numpy
      scipy
      pyqt5
      ipython
      matplotlib
      pytest
      flake8
      pylint
      autopep8
      tqdm
      pywayland
      gymnasium
      torch
      torchrl
      tensordict
      pytest
      ipykernel
      pip
      wandb
      ray
      tblib
    ]))
  ];
  shellHook = ''
    export PYTHONPATH=$PWD
    echo "Python environment ready. Run 'pytest tests/'"
  '';
}
