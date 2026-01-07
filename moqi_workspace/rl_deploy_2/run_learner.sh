#!/bin/bash
export WANDB_API_KEY=1f289522076032788139a1e79857ebd462ac83aa
export PYTHONPATH=$PYTHONPATH:$(pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Find the latest demo file
LATEST_DEMO=$(ls -t pick_place_right_*_demos_*.pkl 2>/dev/null | head -n 1)

if [ -z "$LATEST_DEMO" ]; then
    echo "No demo file found! Please run run_record.sh first."
    exit 1
fi

# Export LD_LIBRARY_PATH to include nvidia libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia/nvjitlink/lib

echo "Using demo file: $LATEST_DEMO"

python train_pick_place.py --learner --exp_name=pick_place_right --demo_path="$LATEST_DEMO"
