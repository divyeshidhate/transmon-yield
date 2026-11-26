"""Tests for yieldkit.lattices and the collision model on full lattices.

Reproduction of Hertzberg et al., npj Quantum Information 7, 129 (2021), arXiv:2009.00781:
layouts from Supplementary Figs 1-9; benchmark numbers from Table 2 and the main text.
Monte Carlo tolerances are our own choices (1000 trials -> ~1.5% standard error on yield).
"""

import numpy as np
import pytest

from yieldkit.collisions import analytic_yield, best_spacing_yield, fit_window
from yieldkit.lattices import LATTICE_GRIDS, N_QUBITS_PAPER, build_lattice, design_frequencies

NAMES = list(LATTICE_GRIDS)


@pytest.mark.parametrize("name", NAMES)
def test_qubit_counts_match_paper(name):
    assert build_lattice(name).n_qubits == N_QUBITS_PAPER[name]


@pytest.mark.parametrize("name", NAMES)
def test_connected_and_no_equal_frequency_neighbours(name):
    lat = build_lattice(name)
    deg = np.bincount(lat.edges.ravel(), minlength=lat.n_qubits)
    assert (deg > 0).all()
    f = lat.freq_index
    assert not np.any(f[lat.edges[:, 0]] == f[lat.edges[:, 1]])


@pytest.mark.parametrize("name", [n for n in NAMES if n.startswith("heavy")])
def test_heavy_lattices_controls_on_degree_two_f3(name):
    """Main text: controls at f3 on degree-two vertices. Every coupling has exactly one f3 end,
    and each f3 qubit couples one f1 and one f2 (transcription check)."""
    lat = build_lattice(name)
    f, e = lat.freq_index, lat.edges
    assert np.all((f[e[:, 0]] == 3) != (f[e[:, 1]] == 3))
    for q in np.flatnonzero(f == 3):
        nbrs = np.concatenate([e[e[:, 0] == q, 1], e[e[:, 1] == q, 0]])
        assert len(nbrs) <= 2
        if len(nbrs) == 2:
            assert sorted(f[nbrs]) == [1, 2]


@pytest.mark.parametrize("name", [n for n in NAMES if n.startswith("square")])
def test_square_lattices_follow_one_stripe_pattern(name):
    """Transcription check: f-index = (a*col + b*row + k) mod 5 + 1 for a single (a, b, k)."""
    lat = build_lattice(name)
    sols = [
        (a, b, k)
        for a in range(5)
        for b in range(5)
        for k in range(5)
        if np.all((a * lat.col + b * lat.row + k) % 5 + 1 == lat.freq_index)
    ]
    assert len(sols) == 1


def test_design_frequencies():
    lat = build_lattice("heavy_hex_d3")
    f = design_frequencies(lat, 5.00e9, 70e6)
    assert set(np.round(f / 1e6)) == {5000, 5070, 5140}  # Hertzberg et al. Discussion example


@pytest.mark.parametrize(
    "name, paper_yield",
    [("heavy_hex_d3", 0.70), ("heavy_hex_d5", 0.33), ("heavy_square_d3", 0.67), ("heavy_square_d5", 0.27)],
)
def test_yield_at_14_mhz_matches_table2(name, paper_yield):
    """Table 2 (sigma_f = 14 MHz): collision-free yield within 5 percentage points."""
    y, _, _ = best_spacing_yield(build_lattice(name), 14e6, n_trials=1000, seed=0)
    assert y == pytest.approx(paper_yield, abs=0.05)


def test_square_d5_near_zero_at_14_mhz():
    """Table 2: square d=5 yield < 0.1% at 14 MHz."""
    y, _, _ = best_spacing_yield(build_lattice("square_d5"), 14e6, n_trials=1000, seed=0)
    assert y < 0.01


def test_as_fabricated_spread_gives_zero_yield():
    """Table 2: sigma_f = 132.3 MHz (as fabricated) -> yield ~0 (<0.1%, 0.1% for heavy-hex d=3)."""
    y, _, _ = best_spacing_yield(build_lattice("heavy_hex_d3"), 132.3e6, n_trials=1000, seed=0)
    assert y <= 0.005


def test_yield_falls_with_sigma_and_size():
    lat3, lat7 = build_lattice("heavy_hex_d3"), build_lattice("heavy_hex_d7")
    y3 = [best_spacing_yield(lat3, s, 500, 0)[0] for s in (6e6, 14e6, 30e6)]
    assert y3[0] > y3[1] > y3[2]
    assert best_spacing_yield(lat7, 14e6, 500, 0)[0] < y3[1]


def test_fit_window_recovers_known_window():
    sig = np.arange(4e6, 40e6, 2e6)
    ys = np.array([analytic_yield(65, 30e6, s) for s in sig])
    assert fit_window(65, sig, ys) == pytest.approx(30e6, abs=0.02e6)
