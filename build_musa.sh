#!/bin/bash

set -x
set -e

pip install -r requirements-build.txt
pip install -r requirements-musa.txt

export VLLM_TARGET_DEVICE=musa
export CMAKE_BUILD_TYPE=Debug
export VERBOSE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

rm -rf build
rm -rf dist
rm -rf vllm.egg-info
pip uninstall -y vllm

python setup.py bdist_wheel
pip install dist/*