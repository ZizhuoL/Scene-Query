.PHONY: setup data model prepare train semantics evaluate test verify serve
PYTHON := .venv/bin/python

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --cache-dir cache/pip -r requirements.lock.txt
	npm ci --cache cache/npm --ignore-scripts
data:
	$(PYTHON) scripts/download_data.py
model:
	$(PYTHON) scripts/download_model.py
prepare:
	$(PYTHON) scripts/prepare.py
train:
	$(PYTHON) scripts/train.py
semantics:
	$(PYTHON) scripts/cache_semantics.py
evaluate:
	$(PYTHON) scripts/inspect_geometry.py
	$(PYTHON) scripts/evaluate.py --run-id reproduction
test:
	$(PYTHON) -m pytest -q tests
verify:
	$(PYTHON) scripts/verify_artifacts.py
serve:
	./run.sh

.PHONY: train-neural evaluate-neural
train-neural:
	$(PYTHON) scripts/train_radiance.py --config config/neural/rgbd_s42.json
	$(PYTHON) scripts/evaluate_radiance.py --run rgbd_s42 --split development
	$(PYTHON) scripts/train_radiance.py --config config/neural/rgb_s42.json
	$(PYTHON) scripts/evaluate_radiance.py --run rgb_s42 --split development
	$(PYTHON) scripts/train_radiance.py --config config/neural/rgbd_s43.json
	$(PYTHON) scripts/evaluate_radiance.py --run rgbd_s43 --split development

evaluate-neural:
	$(PYTHON) scripts/radiance_recovery.py
	$(PYTHON) scripts/finish_neural_experiments.py

.PHONY: train-brush evaluate-brush
train-brush:
	$(PYTHON) scripts/download_brush.py
	$(PYTHON) scripts/prepare_brush.py
	$(PYTHON) scripts/prepare_brush_opaque.py
	$(PYTHON) scripts/prepare_brush_dense.py
	$(PYTHON) scripts/run_brush.py --config config/neural/brush_full.json
	$(PYTHON) scripts/run_brush_opaque.py --config config/neural/brush_opaque.json
	$(PYTHON) scripts/run_brush_dense.py --config config/neural/brush_dense.json

evaluate-brush:
	$(PYTHON) scripts/evaluate_brush.py --final
	$(PYTHON) scripts/evaluate_brush_opaque.py --final
	$(PYTHON) scripts/evaluate_brush_dense.py --final
	$(PYTHON) scripts/deliver_reconstruction.py
