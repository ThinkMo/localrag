#!/bin/bash

export BASE_URL=
export API_KEY=
export MODEL=
export INIT_DATA=true

export PYTHONPATH="./site-packages:."
export HF_ENDPOINT=https://hf-mirror.com

python evaluation/eval.py
