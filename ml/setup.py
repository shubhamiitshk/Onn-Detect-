from setuptools import setup, find_packages

setup(
    name="onndetect",
    version="4.0.0",
    packages=find_packages(where="src"),
    package_dir={"" : "src"},
    install_requires=[
        "torch==2.4.0",
        "torchvision",
        "torchaudio",
        "ai-edge-torch",
        "tensorflow",
        "librosa",
        "soundfile",
        "opencv-python",
        "albumentations",
        "xgboost",
        "scikit-learn",
        "datasets",
        "huggingface-hub",
        "wandb",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "pytest",
        "jupyter"
    ],
)
