PYTHON ?= python
ARGS ?=
.PHONY: setup test train-visual train-audio train-all export parity validate-release
setup:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-export.txt -r requirements-data.txt -r requirements-dev.txt
	$(PYTHON) -m pip install -e .
test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
train-visual:
	$(PYTHON) scripts/train_srm_cnn.py $(ARGS)
train-audio:
	$(PYTHON) scripts/train_lcnn_audio.py $(ARGS)
train-all:
	$(PYTHON) scripts/run_training_pipeline.py $(ARGS)
export:
	$(PYTHON) scripts/export_all_models.py $(ARGS)
parity:
	$(PYTHON) ../scripts/verify_front_end_parity.py --require-torch
validate-release:
	$(PYTHON) scripts/release_bundle.py --validate-only $(ARGS)
