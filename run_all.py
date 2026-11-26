"""Run the test suite and execute all four notebooks top to bottom.

Usage (from a fresh clone):
    pip install -e ".[all]"
    python run_all.py              # tests + notebooks (outputs kept in memory)
    python run_all.py --inplace    # also write executed outputs back into notebooks/

Notebooks run on a Jupyter kernel that uses THIS Python interpreter, so they import the
installed `yieldkit` without any PYTHONPATH. Notebook 04 also regenerates docs/capture_curve.png
(the README headline figure).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    "01_rt_to_frequency.ipynb",
    "02_synthetic_wafer.ipynb",
    "03_collision_yield.ipynb",
    "04_die_ranking.ipynb",
]
KERNEL = "yieldkit-run"


def run_tests() -> None:
    print("== pytest ==", flush=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


def register_local_kernel(tmp: Path) -> None:
    """Point a temporary kernelspec at this interpreter and put it first on JUPYTER_PATH."""
    spec = tmp / "kernels" / KERNEL
    spec.mkdir(parents=True)
    spec.joinpath("kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "yieldkit (this interpreter)",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = os.pathsep.join([str(tmp), os.environ.get("JUPYTER_PATH", "")])


def run_notebooks(inplace: bool) -> None:
    import nbformat
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory() as tmp:
        register_local_kernel(Path(tmp))
        for name in NOTEBOOKS:
            path = ROOT / "notebooks" / name
            t0 = time.time()
            print(f"== {name} ==", flush=True)
            nb = nbformat.read(path, as_version=4)
            NotebookClient(
                nb, kernel_name=KERNEL, timeout=1800, resources={"metadata": {"path": str(path.parent)}}
            ).execute()
            if inplace:
                nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
                nbformat.write(nb, path)
            print(f"   ok ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    run_tests()
    run_notebooks(inplace="--inplace" in sys.argv)
    print("All tests passed and all notebooks ran.")
