#!/bin/bash

# Re-install vLLM from source in editable mode.

set -e

pip uninstall -y vllm

rm -rf /vllm-workspace/vllm/build /vllm-workspace/vllm/*.so

pip install --upgrade pip setuptools wheel setuptools_scm jinja2 ninja

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

export VLLM_COMMIT=6d8d0a24c02bfd84d46b3016b865a44f048ae84b 
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

VLLM_USE_PRECOMPILED=1 uv pip install --system --editable /vllm-workspace/vllm

pip install -r /vllm-workspace/vllm/srs_setup/srs_requirements.txt


# Install Nsight Systems for profiling

apt-get update && apt-get install -y wget gnupg

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

apt-get update

apt-get install -y nsight-systems-2025.3.2

# Ensure the vLLM workspace is a safe directory for git operations

cd vllm

git config --global --add safe.directory /vllm-workspace/vllm