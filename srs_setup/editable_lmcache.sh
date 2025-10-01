#!/bin/bash
set -e

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# if [ -d "$SCRIPT_DIR/LMCache" ]; then
#     rm -rf "$SCRIPT_DIR/LMCache"
# fi

# git clone git@github.com:moyiii-ai/LMCache.git
cd LMCache
# git checkout v0.3.3

pip uninstall -y lmcache


pip install --upgrade pip setuptools wheel setuptools_scm
pip install -r requirements/build.txt

# uv pip install vllm==0.10.1

pip install --editable . --no-build-isolation