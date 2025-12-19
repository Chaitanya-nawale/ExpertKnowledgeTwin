#!/usr/bin/zsh

#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/aimodel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5:00:00
#SBATCH --output=logs/hostLLM%j.log

cd /home/nawale/ExpertDigitalTwin/aimodel

# Activate virtualenv
source /home/nawale/ExpertDigitalTwin/.venv/bin/activate

# HF cache
export HF_HOME=/home/nawale/.hf_cache

# Optional: verbose logging for PyTorch
export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"

# Launch Python server
python serve_MiniStral.py
