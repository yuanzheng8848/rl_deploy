#!/bin/bash
export WANDB_API_KEY=1f289522076032788139a1e79857ebd462ac83aa
export PYTHONPATH=$PYTHONPATH:$(pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Find the latest demo file
# Use the specific demo file
LATEST_DEMO="rl_success_demos.pkl"

if [ ! -f "$LATEST_DEMO" ]; then
    echo "Demo file $LATEST_DEMO not found!"
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

python train_pick_place.py --learner --demo_path="$LATEST_DEMO"
