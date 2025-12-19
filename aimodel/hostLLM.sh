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

source /home/nawale/ExpertDigitalTwin/.venv/bin/activate

export HF_HOME=/home/nawale/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export VLLM_FLEX_ATTENTION=1

vllm serve mistralai/Ministral-8B-Instruct-2410 \
  --host 0.0.0.0 \
  --port 16520 \
  --quantize 8bit \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
