.PHONY: install test all notebooks

install:
	pip install -e ".[all]"

test:
	python -m pytest -q

# Tests + execute all four notebooks (same as: python run_all.py)
all:
	python run_all.py

# Re-execute notebooks and save outputs (regenerates docs/capture_curve.png)
notebooks:
	python run_all.py --inplace
