# Thin wrappers; every target is one command you can also paste into PowerShell (see README).
# Override on the command line, e.g.: make replay VIDEO=data/recordings/P01_s01.mp4 OUT=out/P01_s01
PY ?= python
VIDEO ?= data/recordings/sample.mp4
OUT ?= out
CFG = --config configs/pipeline.yaml --config configs/policy.yaml

.PHONY: setup test replay eval train-detector train-temporal web serve

setup:
	$(PY) -m pip install -e ".[perception,train,dev]"

test:
	$(PY) tests/run.py

replay:
	proctorlens replay $(VIDEO) $(CFG) --out $(OUT)

eval:
	$(PY) -m ml.evaluation.run_eval

train-detector:
	$(PY) -m ml.training.train_detector

train-temporal:
	$(PY) -m ml.training.train_temporal_gbm
	$(PY) -m ml.training.train_temporal_tcn

web:
	$(PY) -m pip install -e ".[perception,web]"
	cd frontend && npm install && npm run build

serve:
	proctorlens serve $(CFG)
