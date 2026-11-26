"""Qubit lattices and frequency patterns -- REPRODUCTION for research purposes.

Transcribed from Hertzberg et al. 2021, Supplementary Information, Supplementary
Figures 1-9 (npj Quantum Information 7, 129; arXiv:2009.00781).

Each lattice is a grid of characters, one string per drawn row:
    '1'..'5' = qubit with set-point frequency f1..f5,   '.' = no qubit.
In every SI figure, two qubits are coupled exactly when they are horizontal or
vertical neighbours on this grid (our reading of the figures; the heavy lattices'
bridge rows have qubits only every 2nd or 4th column, so no extra couplings appear).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# fmt: off
LATTICE_GRIDS: dict[str, dict] = {
    # --- square, 5-frequency pattern ---
    "square_d3": {"si_figure": 1, "grid": [
        "..53.",
        "3142.",
        "25314",
        ".4253",
        ".31..",
    ]},
    "square_d5": {"si_figure": 4, "grid": [
        "....45...",
        "..4512...",
        "..123451.",
        "12345123.",
        "345123451",
        ".12345123",
        ".345123..",
        "...2345..",
        "...45....",
    ]},
    "square_d7": {"si_figure": 7, "grid": [
        "......23.....",
        "....2345.....",
        "....451234...",
        "..45123451...",
        "..1234512345.",
        "123451234512.",
        "3451234512345",
        ".123451234512",
        ".3451234512..",
        "...23451234..",
        "...451234....",
        ".....3451....",
        ".....51......",
    ]},
    # --- heavy-square, 3-frequency pattern ---
    "heavy_square_d3": {"si_figure": 2, "grid": [
        "231323.",
        "3.3.3..",
        "1323132",
        "..3.3.3",
        ".313231",
    ]},
    "heavy_square_d5": {"si_figure": 5, "grid": [
        "2313231323.",
        "3.3.3.3.3..",
        "13231323132",
        "..3.3.3.3.3",
        "23132313231",
        "3.3.3.3.3..",
        "13231323132",
        "..3.3.3.3.3",
        ".3132313231",
    ]},
    "heavy_square_d7": {"si_figure": 8, "grid": [
        "23132313231323.",
        "3.3.3.3.3.3.3..",
        "132313231323132",
        "..3.3.3.3.3.3.3",
        "231323132313231",
        "3.3.3.3.3.3.3..",
        "132313231323132",
        "..3.3.3.3.3.3.3",
        "231323132313231",
        "3.3.3.3.3.3.3..",
        "132313231323132",
        "..3.3.3.3.3.3.3",
        ".31323132313231",
    ]},
    # --- heavy-hexagon, 3-frequency pattern ---
    "heavy_hex_d3": {"si_figure": 3, "grid": [
        ".323132",
        "..3...3",
        "2313231",
        "3...3..",
        "132313.",
    ]},
    "heavy_hex_d5": {"si_figure": 6, "grid": [
        ".3231323132",
        "..3...3...3",
        "23132313231",
        "3...3...3..",
        "13231323132",
        "..3...3...3",
        "23132313231",
        "3...3...3..",
        "1323132313.",
    ]},
    "heavy_hex_d7": {"si_figure": 9, "grid": [
        ".31323132313231",
        "..3...3...3...3",
        "132313231323132",
        "3...3...3...3..",
        "231323132313231",
        "..3...3...3...3",
        "132313231323132",
        "3...3...3...3..",
        "231323132313231",
        "..3...3...3...3",
        "132313231323132",
        "3...3...3...3..",
        "23132313231323.",
    ]},
}
# fmt: on

N_QUBITS_PAPER = {
    "square_d3": 17, "square_d5": 49, "square_d7": 97,
    "heavy_square_d3": 25, "heavy_square_d5": 73, "heavy_square_d7": 145,
    "heavy_hex_d3": 23, "heavy_hex_d5": 65, "heavy_hex_d7": 127,
}
"""Qubit counts, Hertzberg et al. 2021, Table 2 and Supp. Fig. captions."""


@dataclass(frozen=True)
class Lattice:
    """A qubit lattice: grid positions, frequency-pattern index (1..5) and couplings."""

    name: str
    si_figure: int
    row: np.ndarray  # (N,) grid row of each qubit
    col: np.ndarray  # (N,) grid column of each qubit
    freq_index: np.ndarray  # (N,) 1..5 -> f1..f5
    edges: np.ndarray  # (E, 2) coupled pairs (undirected)

    @property
    def n_qubits(self) -> int:
        return len(self.freq_index)


def build_lattice(name: str) -> Lattice:
    """Parse one transcribed SI figure into qubits and couplings (grid neighbours)."""
    spec = LATTICE_GRIDS[name]
    cells = [(r, c, int(ch)) for r, line in enumerate(spec["grid"]) for c, ch in enumerate(line) if ch != "."]
    index = {(r, c): i for i, (r, c, _) in enumerate(cells)}
    edges = [
        (index[(r, c)], index[(r + dr, c + dc)])
        for (r, c, _) in cells
        for dr, dc in ((0, 1), (1, 0))
        if (r + dr, c + dc) in index
    ]
    rows, cols, fidx = (np.array(v) for v in zip(*cells))
    return Lattice(name, spec["si_figure"], rows, cols, fidx, np.array(edges, dtype=int))


def design_frequencies(lattice: Lattice, f1_hz: float, step_hz: float) -> np.ndarray:
    """Set-point frequencies f_n = f1 + (n - 1) * step.

    Hertzberg et al. 2021, Sec. IV.3: frequencies "spaced at regular intervals"
    with f1 = 5 GHz; the step is swept to minimise collisions at each sigma_f.
    """
    return f1_hz + (lattice.freq_index - 1) * step_hz
