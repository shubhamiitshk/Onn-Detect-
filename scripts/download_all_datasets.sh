#!/bin/bash
set -e

echo "Downloading all datasets..."

echo "Downloading ASVSpoof..."
python datasets/download_asvspoof.py

echo "Downloading GenImage..."
python datasets/download_genimage.py

echo "Downloading OpenFake..."
python datasets/download_openfake.py

echo "All datasets downloaded successfully."
