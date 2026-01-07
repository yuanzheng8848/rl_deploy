#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python train_pick_place.py --actor --exp_name=pick_place_right
