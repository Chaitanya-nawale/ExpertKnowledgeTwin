#!/usr/bin/zsh

### Job Parameters
#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/embeddingModel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5:00:00
#SBATCH --output=logs/testEmbedding%j.log

cd /home/nawale/ExpertDigitalTwin/embeddingModel

source /home/nawale/ExpertDigitalTwin/.venv/bin/activate


export HF_HOME=/home/nawale/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

python testEmbedding.py