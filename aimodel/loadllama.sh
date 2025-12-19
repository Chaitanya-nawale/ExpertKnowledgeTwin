#!/bin/bash

### Job Parameters
#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/aimodel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5:00:00
#SBATCH --output=logs/hostLLM%j.log

# Initialize module system
source /etc/profile.d/modules.sh

# Load required modules
module load gcc
module load cmake
module load cuda

cd /home/nawale/ExpertDigitalTwin/aimodel

# Ensure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# hf download unsloth/Ministral-3-14B-Instruct-2512-GGUF \
#   --repo-type model \
#   --include "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf" \
#   --local-dir ~/models/ministral-gguf

# hf download Linq-AI-Research/Linq-Embed-Mistral \
#   --repo-type model \
#   --local-dir ~/models/embeddings/linq

# python convert_hf_to_gguf.py \
#   ~/models/embeddings/linq \
#   --outfile ~/models/embeddings/linq/linq-q4_k_m.gguf \
#   --outtype q4_k_m

MODEL_PATH=/home/nawale/models/ministral-gguf/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf

git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

# Build with CMake and CUDA support (disable CURL since we have the model)
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build . -j16
cd ..
