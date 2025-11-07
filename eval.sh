#!/bin/bash

export PYTHONPATH="./site-packages:."
export INID_DATA=true

python evaluation/eval.py
