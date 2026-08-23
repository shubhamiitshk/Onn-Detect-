#!/bin/bash
set -e

echo "Starting training pipeline..."

cd ml
echo "Installing requirements..."
pip install -r requirements.txt
pip install -e .

echo "Training Visual Model..."
python scripts/train_srm_cnn.py

echo "Training Audio Model..."
python scripts/train_lcnn_audio.py

echo "Training Meta Classifier..."
python scripts/train_xgboost_meta.py

echo "Training pipeline completed."
cd ..
