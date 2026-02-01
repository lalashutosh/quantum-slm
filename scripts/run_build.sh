#!/bin/bash
# scripts/run_build.sh

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/run_build.sh v1"
    exit 1
fi

echo "------------------------------------------"
echo "SLM: $VERSION"
echo "------------------------------------------"

# 0. Aalto Infrastructure Setup
# This prevents your home quota from exploding
export HF_HOME="/tmp/$USER_hf_cache"
mkdir -p $HF_HOME
echo "Model Cache directed to: $HF_HOME"

# 1. Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual Environment Activated."
else
    echo "venv not found. Run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# 2. Data Specialization Step
echo "Step 1: Running Data Engine (Context-Aware)..."
python3 src/data_engine.py || { echo "Data Prep Failed"; exit 1; }

# 3. Training Step (The knowledge injection)
echo "Step 2: Launching Trainer (4-bit LoRA)..."
python3 src/trainer.py || { echo "Training Failed"; exit 1; }

echo "------------------------------------------"
echo "BUILD $VERSION COMPLETE"
echo "Model saved in models/${VERSION}_weights"
echo "------------------------------------------"