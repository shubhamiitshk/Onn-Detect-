"""Compatibility entry point for the strict, explicit model parity diagnostic.

Use --help for required verified checkpoint, dataset, TFLite and threshold.
"""
from diagnose_models import main

if __name__ == "__main__":
    main(default_modality='audio')
