.PHONY: setup train-visual train-audio train-meta train-all export eval package clean

setup:
	pip install -r requirements.txt
	pip install -e .

train-visual:
	python scripts/train_srm_cnn.py

train-audio:
	python scripts/train_lcnn_audio.py

train-meta:
	python scripts/train_xgboost_meta.py

train-all: train-visual train-audio train-meta

export:
	python scripts/export_all_models.py

eval:
	python scripts/run_full_pipeline.py

package:
	cd .. && ./scripts/package_for_android.sh

clean:
	rm -rf checkpoints/* exported/*
