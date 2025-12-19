#!/usr/bin/zsh

### Job Parameters
#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/aimodel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5:00:00
#SBATCH --output=logs/hostLLM%j.log

cd /home/nawale/ExpertDigitalTwin/aimodel

# Activate python environment
source /home/nawale/ExpertDigitalTwin/.venv/bin/activate

# HuggingFace cache on large scratch space
export HF_HOME=/scratch/izar/nawale/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Enable FlashAttention v1 on V100 (FA2 unsupported)
export VLLM_USE_FLASH_ATTENTION=1

# Optional: allow vLLM to flexibly choose kernels
export VLLM_FLEX_ATTENTION=1

# Ensure CUDA_HOME is set (required for TVM ops on many clusters)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run vLLM with optimized settings
vllm serve unsloth/Ministral-3-14B-Instruct-2512-GGUF \
  --host 0.0.0.0 \
  --port 16520 \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --tool-call-parser mistral \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill 