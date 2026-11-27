.PHONY: install test test-network all notebooks

install:
	pip install -e ".[all]"

test:
	python -m pytest -q

# Optional: tests that download public data from Zenodo (deselected by default)
test-network:
	python -m pytest -q -m network

# Tests + execute all five notebooks (same as: python run_all.py)
all:
	python run_all.py

# Re-execute notebooks and save outputs (regenerates docs/capture_curve.png)
notebooks:
	python run_all.py --inplace
