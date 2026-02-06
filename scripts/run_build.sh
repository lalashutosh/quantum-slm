#!/bin/bash
# scripts/run_build.sh

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/run_build.sh v1_2"
    exit 1
fi

echo "------------------------------------------"
echo "🚀 STARTING CLEAN BUILD: $VERSION"
echo "------------------------------------------"

# 0. Infrastructure Setup
export HF_HOME="/tmp/$USER_hf_cache"
mkdir -p $HF_HOME

# 1. Activate Environment
source venv/bin/activate

# 2. Data Ingestion (Aggregates all .txt files in data/raw)
echo "Step 1: Running Data Engine on Research Corpus..."
# Note: Ensure your data_engine.py output_file path matches $VERSION
python3 src/data_engine.py || { echo "❌ Data Ingestion Failed"; exit 1; }

# 3. Training (Full re-train from base weights)
echo "Step 2: Launching Trainer (4-bit QLoRA)..."
python3 src/trainer.py $VERSION || { echo "❌ Training Failed"; exit 1; }

# 4. Evaluation (Immediate validation)
echo "Step 3: Running Automated Evaluation Suite..."
python3 src/evaluator.py $VERSION v1_1 || { echo "❌ Evaluation Failed"; exit 1; }

echo "------------------------------------------"
echo "✅ BUILD $VERSION COMPLETE"
echo "Check logs/ for the comparison report."
echo "------------------------------------------"