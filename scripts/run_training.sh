#!/bin/bash

# Set the PYTHONPATH to the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Python script
python3 scripts/model_training.py