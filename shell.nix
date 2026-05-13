{
  pkgs ? import <nixpkgs> {}
}:

(pkgs.buildFHSEnv {
  name = "pippy";
  targetPkgs = pps: (with pps; [
    python312
    python312Packages.virtualenv
    libz
  ]);
}).env
