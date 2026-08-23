$ErrorActionPreference = 'Stop'

Write-Host "Setting up Python environment using uv..."
$env:UV_HTTP_TIMEOUT = "3600"

# Create a clear virtual environment using Python 3.10
uv venv --python 3.10 

# Install PyTorch CUDA (full-fledged) for RTX A4000
Write-Host "Installing PyTorch with CUDA support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the dependencies
Write-Host "Installing remaining requirements..."
uv pip install -r ml/requirements.txt requests
uv pip install -e ml/

Write-Host "Downloading all datasets..."
uv run python ml/scripts/download_all_datasets.py

Write-Host "Starting training pipeline..."
cd ml

Write-Host "Training Visual Model..."
uv run python scripts/train_srm_cnn.py --device cuda

Write-Host "Training Audio Model..."
uv run python scripts/train_lcnn_audio.py --device cuda

Write-Host "Training Meta Classifier..."
uv run python scripts/train_xgboost_meta.py

Write-Host "Exporting models + Android constants..."
uv run python scripts/export_all_models.py
uv run python scripts/generate_android_constants.py

Write-Host "Running honest evaluation..."
uv run python scripts/run_full_pipeline.py

Write-Host "Training pipeline completed!"
cd ..
