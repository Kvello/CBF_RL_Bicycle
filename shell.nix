{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/80c24eeb9ff46aa99617844d0c4168659e35175f.tar.gz") {} }:
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
    ]))
  ];
}
