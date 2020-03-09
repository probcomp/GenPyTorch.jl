#!/bin/sh

# run this script from the GenTF/ directory with the PYTHON environment variable set:

# PYTHON=/path-to-python-env-with-torch-installed/bin/python ./docs/build_docs_locally.sh

# it will generate HTML documentation under docs/build.

julia --project=docs/ -e '
    using Pkg;
    Pkg.add(PackageSpec(url="https://github.com/probcomp/Gen"));
    Pkg.develop(PackageSpec(path="$(pwd())"));
    Pkg.instantiate();
    Pkg.add("PyCall");
    Pkg.build("PyCall");
    using PyCall;
    println(ENV["PYTHON"]);
    println(PyCall.python)'

julia --project=docs/ docs/make.jl
