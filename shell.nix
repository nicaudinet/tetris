{ pkgs ? import <nixpkgs> {} }:

let

  pythonLibs = with pkgs.python38Packages; [
    numpy
    pygame
    h5py
  ];

  customPython = pkgs.python38.buildEnv.override {
    extraLibs = pythonLibs;
  };

in

  pkgs.mkShell {
    buildInputs = [ customPython ];
  }
