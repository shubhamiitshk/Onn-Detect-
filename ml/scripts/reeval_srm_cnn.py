"""Compatibility entrypoint; delegates to the configured strict training path."""
from train_srm_cnn import run
if __name__ == "__main__":
    run("visual")
