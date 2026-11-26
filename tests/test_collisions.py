"""Tests for yieldkit.collisions (reproduction of Hertzberg et al., npj QI 7, 129 (2021), arXiv:2009.00781).

Toy graphs with hand-picked frequencies check that each Table 1 condition fires
exactly where the paper says. Lattice-level tests come once the lattices are built.
"""

import numpy as np
import pytest

from yieldkit.collisions import (
    ALPHA_HZ,
    COLLISION_TABLE,
    analytic_yield,
    count_collisions,
    directed_edges,
    simulate_yield,
    spectator_triples,
)

MHZ = 1e6
GHZ = 1e9


def test_table_matches_paper():
    """Bounds as printed in Table 1 of Hertzberg et al. 2021; anharmonicity ~ -330 MHz (caption)."""
    assert [c.bound_hz for c in COLLISION_TABLE] == [17 * MHZ, 4 * MHZ, 30 * MHZ, None, 17 * MHZ, 25 * MHZ, 17 * MHZ]
    assert ALPHA_HZ == -330 * MHZ


def _pair(f_control_hz: float, f_target_hz: float) -> dict[int, int]:
    """Two coupled qubits; design frequencies fix qubit 0 as control (5.07 > 5.00 GHz)."""
    design = np.array([5.07, 5.00]) * GHZ
    directed = directed_edges(np.array([[0, 1]]), design)
    triples = spectator_triples(directed, 2)
    counts = count_collisions(np.array([f_control_hz, f_target_hz]), directed, triples)
    return {t: int(c) for t, c in counts.items()}


def test_control_is_higher_design_frequency():
    d = directed_edges(np.array([[1, 0], [1, 2]]), np.array([5.07, 5.00, 5.14]) * GHZ)
    assert d.tolist() == [[0, 1], [2, 1]]


def test_good_pair_has_no_collisions():
    """70 MHz detuning (the paper's example step) avoids every pair condition."""
    assert sum(_pair(5.07 * GHZ, 5.00 * GHZ).values()) == 0


def test_type1_degenerate_pair():
    c = _pair(5.010 * GHZ, 5.000 * GHZ)  # 10 MHz apart, bound 17 MHz
    assert c[1] == 1 and sum(c.values()) == 1


def test_type2_two_photon():
    """f_j02 = 2 f_k01  <=>  f_j - f_k = -alpha/2 = 165 MHz."""
    c = _pair(5.165 * GHZ, 5.000 * GHZ)
    assert c[2] == 1 and sum(c.values()) == 1


def test_type3_f01_equals_f12():
    """Target sits 10 MHz above the control's 1-2 line (bound 30 MHz), still inside the CR window."""
    c = _pair(5.320 * GHZ, 5.000 * GHZ)
    assert c[3] == 1 and sum(c.values()) == 1


def test_type4_target_above_control():
    c = _pair(5.000 * GHZ, 5.070 * GHZ)
    assert c[4] == 1 and sum(c.values()) == 1


def test_type4_target_below_control_12():
    """Target below f_j12 = f_j - 330 MHz is outside the CR window."""
    c = _pair(5.400 * GHZ, 5.000 * GHZ)
    assert c[4] == 1


def _line(fi: float, fj: float, fk: float) -> dict[int, int]:
    """Line i(0) - j(1) - k(2); design makes j the control of both neighbours."""
    design = np.array([5.00, 5.14, 5.07]) * GHZ
    directed = directed_edges(np.array([[0, 1], [1, 2]]), design)
    triples = spectator_triples(directed, 3)
    counts = count_collisions(np.array([fi, fj, fk]), directed, triples)
    return {t: int(c) for t, c in counts.items()}


def test_spectator_triples_need_a_control_centre():
    design = np.array([5.00, 5.14, 5.07]) * GHZ  # centre is control
    d = directed_edges(np.array([[0, 1], [1, 2]]), design)
    assert spectator_triples(d, 3).tolist() == [[0, 1, 2]]
    design_low_centre = np.array([5.07, 5.00, 5.14]) * GHZ  # centre is target of both
    d2 = directed_edges(np.array([[0, 1], [1, 2]]), design_low_centre)
    assert spectator_triples(d2, 3).size == 0


def test_type5_spectators_degenerate():
    c = _line(5.000 * GHZ, 5.140 * GHZ, 5.010 * GHZ)
    assert c[5] == 1


def test_type7_two_photon_with_spectator():
    """f_j02 = f_i + f_k with f_j02 = 2 f_j + alpha."""
    fj = 5.140 * GHZ
    fi = 4.900 * GHZ
    fk = 2 * fj + ALPHA_HZ - fi  # 5.050 GHz
    c = _line(fi, fj, fk)
    assert c[7] == 1


def test_simulate_yield_limits():
    design = np.array([5.00, 5.14, 5.07]) * GHZ
    edges = np.array([[0, 1], [1, 2]])
    assert simulate_yield(design, edges, sigma_hz=0.0, n_trials=10, seed=0) == (1.0, 0.0)
    y, mean = simulate_yield(design, edges, sigma_hz=500 * MHZ, n_trials=2000, seed=0)
    assert y < 0.5 and mean > 0.5


def test_simulate_yield_seed_reproducible():
    design = np.array([5.00, 5.14, 5.07]) * GHZ
    edges = np.array([[0, 1], [1, 2]])
    a = simulate_yield(design, edges, sigma_hz=30 * MHZ, n_trials=500, seed=1)
    assert a == simulate_yield(design, edges, sigma_hz=30 * MHZ, n_trials=500, seed=1)


def test_analytic_yield():
    assert analytic_yield(1, window_hz=1.0, sigma_hz=1.0) == pytest.approx(0.8413, abs=1e-4)
    # Hertzberg Fig. 4 caption: heavy-hex window 27.99 MHz used for 300 qubits.
    assert analytic_yield(300, 27.99 * MHZ, 14 * MHZ) < analytic_yield(100, 27.99 * MHZ, 14 * MHZ)
