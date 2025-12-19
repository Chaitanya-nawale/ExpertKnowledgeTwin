#!/bin/bash

#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/embeddingModel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5:00:00
#SBATCH --output=logs/testEmbedding%j.log

# Set your actual HF token here
export HF_TOKEN=""

export HF_HOME=/scratch/izar/nawale/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export APP_PORT=16022

# Unset deprecated variable
unset TRANSFORMERS_CACHE

cd /home/nawale/ExpertDigitalTwin/embeddingModel

source /home/nawale/ExpertDigitalTwin/.venv/bin/activate

echo "========================================="
echo "Starting EmbeddingGemma Server"
echo "Node: $(hostname)"
echo "Port: $APP_PORT"
echo "HF_HOME: $HF_HOME"
echo "HF_TOKEN set: $([ -z "$HF_TOKEN" ] && echo "NO" || echo "YES")"
echo "========================================="

# Start server with explicit environment variables
nohup env HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" uvicorn app:app --host 0.0.0.0 --port $APP_PORT > embedding_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
sleep 5

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server started successfully!"
    # Show last 20 lines of log
    tail -20 embedding_server.log
else
    echo "Server failed to start!"
    tail -50 embedding_server.log
fi

# Keep job running
sleep infinity
