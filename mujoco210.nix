{ lib, stdenv, fetchurl, unzip }:

stdenv.mkDerivation rec {
  pname = "mujoco210";
  version = "2.1.0";

  src = fetchurl {
    url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz";
    sha256 = "sha256-pDbKL0FEw4uDcgVjW71g/+EWLVtEyH3yIjJ5WXjX0BI=";
  };
  mjkey = fetchurl {
    url = "https://www.roboti.us/file/mjkey.txt";
    sha256 = "sha256-v/5AO85peNMpI5yD6HTg/UEnQNFJg0uMBRaJukqa3sw=";
  };

  nativeBuildInputs = [ unzip ];

  installPhase = ''
    mkdir -p $out
    cp -r ./* $out/
    cp ${mjkey} $out/mjkey.txt
  '';

  meta = {
    description = "MuJoCo physics engine binaries";
    homepage = "https://mujoco.org/";
    license = lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
  };
}