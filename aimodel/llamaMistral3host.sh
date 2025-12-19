#!/bin/bash

### Job Parameters
#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/aimodel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/llamaM3%j.log

# Initialize module system
source /etc/profile.d/modules.sh

# Load required modules
module load gcc
module load cmake
module load cuda

cd /home/nawale/ExpertDigitalTwin/aimodel/llama.cpp

# Ensure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MODEL_PATH=/home/nawale/models/ministral-gguf/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf

./build/bin/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 16520 \
  -ngl 80 \
  -c 32768 \
  -n 256 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --threads 8
