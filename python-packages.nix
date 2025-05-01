{ pkgs, python }:

let 
  mujoco210 = import ./mujoco210.nix { 
    inherit (pkgs) lib stdenv fetchurl unzip;
  };
  mujoco_py = python.pkgs.buildPythonPackage rec {
    pname = "mujoco-py";
    version = "2.1.12.14";  # or your version

    src = pkgs.fetchFromGitHub {
      owner = "openai";
      repo = "mujoco-py";
      rev = "f1312cc";
      sha256 = "sha256-nwIJzLPhTZNlwk/NAiWCV/zYdwKeuQqhW6UIniGw8+k=";
    };

    propagatedBuildInputs = with python.pkgs; [
      numpy
      cffi
      glfw
      fasteners
      imageio
    ];
    nativeBuildInputs = with python.pkgs; [
      (cython.overridePythonAttrs (old: {
      version = "0.29.24";
      src = pkgs.fetchPypi {
        pname = "Cython";
        version = "0.29.24";
        sha256 = "sha256-zfBNB8NgCGDowuuq1Oj1KsP+shJFPBdkpJrAjIJ+hEM=";
        };
      }))
      setuptools
      wheel
    ];
    preBuild = ''
      export MUJOCO_PY_MUJOCO_PATH=${mujoco210}
      export MUJOCO_PY_MJKEY_PATH=${mujoco210}/mjkey.txt
      export LD_LIBRARY_PATH=${mujoco210}/bin:$LD_LIBRARY_PATH
      echo "MUJOCO_PY_MUJOCO_PATH during build: $MUJOCO_PY_MUJOCO_PATH"
      echo "LD_LIBRARY_PATH during build: $LD_LIBRARY_PATH"
    '';
    makeWrapperArgs = [
      "--set MUJOCO_PY_MUJOCO_PATH ${mujoco210}"
      "--set MUJOCO_PY_MJKEY_PATH ${mujoco210}/mjkey.txt"
    ];

    # You can optionally patch your environment at runtime via shellHook too
    meta = with pkgs.lib; {
      description = "MuJoCo Python bindings";
      homepage = "https://github.com/openai/mujoco-py";
      license = licenses.mit;
    };
  };
   torchrl = python.pkgs.buildPythonPackage rec {
     pname = "torchrl";
     version = "0.5.0";
     format = "wheel";
     src = pkgs.fetchurl {
       inherit pname version;
       url = "https://files.pythonhosted.org/packages/49/44/d7d137b323c92b5a438d13fafade6edd2636ac22ab5c9e7e54e094006892/torchrl-0.5.0-cp39-cp39-manylinux1_x86_64.whl";
       sha256 = "fb1ac3788678a588533ad1aba67d4d2bd55fb8d1344ddfed5b0c1855b6720af1";
     };
     propagatedBuildInputs = with python.pkgs; [ torch numpy tensordict ];
   };
 
   tensordict = python.pkgs.buildPythonPackage rec {
     pname = "tensordict";
     version = "0.5.0";
     src = pkgs.fetchurl {
       inherit pname version;
       url = "https://files.pythonhosted.org/packages/ed/15/99835a6103b24a6c625c9f7917e930dfb215a6b345e83a39a716c0a6a8cc/tensordict-0.5.0-cp38-cp38-manylinux1_x86_64.whl";
       sha256 = "sha256-+ocZvQICVBLQt2cs5Vf8tRReikhzFxc8UPVWjAgJ+OI=";
     };
     propagatedBuildInputs = with python.pkgs; [ torch numpy ];
   };
#   gymnasium = python.pkgs.buildPythonPackage rec {
#     pname = "gymnasium";
#     version = "1.0.0";
#     src = pkgs.fetchPypi {
#       inherit pname version;
#       sha256 = "9d2b66f30c1b34fe3c2ce7fae65ecf365d0e9982d2b3d860235e773328a3b403";
#     };
#     
#     propagatedBuildInputs = with python.pkgs; [
#       numpy
#       cloudpickle
#       wheel
#     ];
#     
#     # Skip unnecessary build phases for wheel packages
#     dontUnpack = true;
#     dontBuild = true;
#     dontConfigure = true;
#   };
   matplotlib = python.pkgs.buildPythonPackage rec {
     pname = "matplotlib";
     version = "3.6.1";
     src = pkgs.fetchPypi {
       inherit pname version;
       sha256 = "sha256-4tG3IlZm9+G8yUwLycWHqC4+hpHaR1fjV+XCUVIi7jc=";
     };
     propagatedBuildInputs = with python.pkgs; [ numpy ];
   };
#   pylint = python.pkgs.buildPythonPackage rec {
#     pname = "pylint";
#     version = "2.15.0";
#     src = pkgs.fetchPypi {
#       inherit pname version;
#       sha256 = "sha256-+ocZvQICVBLQt2cs5Vf8tRReikhzFxc8UPVWjAgJ+OI=";
#     };
#     propagatedBuildInputs = with python.pkgs; [ astroid ];
#   };
   torch = python.pkgs.buildPythonPackage rec {
     pname = "torch";
     version = "2.5.1";
     format = "wheel";
     src = pkgs.fetchurl {
       inherit pname version;
       url = "https://files.pythonhosted.org/packages/a9/18/81c399e8f4f1580d34bf99d827cb5fb5cf7a18a266bb5d30ca3ec2e89ba6/torch-2.5.1-cp39-cp39-manylinux1_x86_64.whl";
       sha256 = "1f3b7fb3cf7ab97fae52161423f81be8c6b8afac8d9760823fd623994581e1a3";
     };
     propagatedBuildInputs = with python.pkgs; [ numpy ];
   };
# 
#   time-machine = python.pkgs.buildPythonPackage rec {
#     pname = "time-machine";
#     version = "2.15.0";
#     src = pkgs.fetchPypi {
#       inherit pname version;
#       sha256 = pkgs.lib.fakeSha256;
#     };
#     propagatedBuildInputs = with python.pkgs; [ ];
#   };
#   wandb = python.pkgs.buildPythonPackage rec {
#     pname = "wandb";
#     version = "0.18.5";
#     src = pkgs.fetchPypi {
#       inherit pname version;
#       sha256 = pkgs.lib.fakeSha256;
#     };
#     propagatedBuildInputs = with python.pkgs; [ numpy ];
#   };
numpy = python.pkgs.buildPythonPackage rec {
  pname = "numpy";
  version = "1.26.4";
  src = pkgs.fetchPypi {
    inherit pname version;
    sha256 = "2a02aba9ed12e4ac4eb3ea9421c420301a0c6460d9830d74a9df87efa4912010";
  };
  meta = with pkgs.lib; {
    description = "NumPy is the fundamental package for array computing with Python.";
    homepage = "https://numpy.org/";
    license = licenses.bsd3;
  };
};

in
python.withPackages (ps: with ps; [
  numpy
  scipy
  matplotlib
  gymnasium
  torch
  torchrl
  tensordict
  pytest
  pip
  mujoco_py
])