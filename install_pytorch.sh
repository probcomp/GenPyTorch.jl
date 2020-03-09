#!/bin/sh
set -ex
apt-get update -qq && apt-get install -qq \
    build-essential \
    python3 \
    python-virtualenv
virtualenv -p python3 pytorch_env 
. pytorch_env/bin/activate
pip install --progress-bar off --upgrade pip
pip install --progress-bar off --upgrade torch
deactivate
