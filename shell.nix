{ pkgs ? import <nixpkgs> {} }:

let

  pythonLibs = with pkgs.python38Packages; [
    numpy
    matplotlib
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
