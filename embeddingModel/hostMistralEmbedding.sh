#!/bin/bash
#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/embeddingModel
#SBATCH -J linq-embed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=logs/linq_%j.log

source /home/nawale/ExpertDigitalTwin/.venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /home/nawale/ExpertDigitalTwin/embeddingModel
# python embedding_server.py
python combined_server.py
