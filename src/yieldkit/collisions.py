"""Frequency-collision model -- REPRODUCTION of a published model for research purposes.

Source: Hertzberg et al., "Laser-annealing Josephson junctions for yielding
scaled-up superconducting quantum processors", npj Quantum Information 7, 129 (2021),
arXiv:2009.00781 (v4). Table 1 (collision types and bounds), Sec. IV.3 (Monte Carlo).

Conventions (Table 1): j = control, k = target, i = spectator; all three are
fixed-frequency transmons with cross-resonance (CR) gates.
Frequencies are in Hz (SI); bounds are stored in Hz.

Transition frequencies used by Table 1:
    f12 = f01 + alpha            (alpha = anharmonicity = f12 - f01, Sec. IV.1)
    f02 = f01 + f12 = 2 f01 + alpha
The second line follows from the definition of alpha; the paper does not write it out.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

ALPHA_HZ: float = -330e6
"""Anharmonicity f12 - f01 ~ -330 MHz (Hertzberg et al. 2021, Table 1 caption and Sec. IV.1)."""


@dataclass(frozen=True)
class CollisionType:
    """One row of Hertzberg et al. 2021, Table 1."""

    number: int
    condition: str  # as printed (j = control, k = target, i = spectator)
    participants: str
    bound_hz: float | None  # +/- window; None for the type-4 forbidden zone


COLLISION_TABLE: tuple[CollisionType, ...] = (
    CollisionType(1, "f_j01 = f_k01", "nearest neighbours j, k", 17e6),
    CollisionType(2, "f_j02 = 2 f_k01", "control j, target k", 4e6),
    CollisionType(3, "f_j01 = f_k12", "nearest neighbours j, k", 30e6),
    CollisionType(4, "f_k01 < f_j12 or f_j01 < f_k01", "control j, target k", None),
    CollisionType(5, "f_i01 = f_k01", "j controls i and/or k, NN to both", 17e6),
    CollisionType(6, "f_i01 = f_k12 or f_i12 = f_k01", "j controls i and/or k, NN to both", 25e6),
    CollisionType(7, "f_j02 = f_k01 + f_i01", "j controls i and/or k, NN to both", 17e6),
)
"""Hertzberg et al. 2021, Table 1. Bounds for types 1-3: region where gate error from the
collision exceeds ~1% (their ref. 22); types 5-7 based on types 1 and 3. Type 4 is a
forbidden zone ("ZXZX coupling is too low"), with no bound."""

BOUND = {c.number: c.bound_hz for c in COLLISION_TABLE}


def directed_edges(edges: np.ndarray, design_hz: np.ndarray) -> np.ndarray:
    """Orient every coupled pair as (control, target).

    Rule: Table 1 type 4 "requires f_control > f_target" (Hertzberg et al. 2021,
    Discussion), so the qubit with the higher DESIGN frequency is the control.

    Args:
        edges: (E, 2) int array of coupled qubit pairs (undirected).
        design_hz: (N,) design f01 of each qubit [Hz]. Coupled qubits must differ.

    Returns:
        (E, 2) int array of (control j, target k).
    """
    edges = np.asarray(edges)
    a, b = edges[:, 0], edges[:, 1]
    if np.any(design_hz[a] == design_hz[b]):
        raise ValueError("coupled qubits must have different design frequencies")
    a_is_ctrl = design_hz[a] > design_hz[b]
    return np.column_stack([np.where(a_is_ctrl, a, b), np.where(a_is_ctrl, b, a)])


def spectator_triples(directed: np.ndarray, n_qubits: int) -> np.ndarray:
    """All (i, j, k) with j a nearest neighbour of both i and k and control to i and/or k.

    Table 1 rows 5-7 participants: "Q_j is control to Q_i and/or Q_k & is
    nearest-neighbor to both" (Hertzberg et al. 2021). Each unordered {i, k} is listed once.

    Returns:
        (T, 3) int array of (i, j, k).
    """
    neighbours: list[set[int]] = [set() for _ in range(n_qubits)]
    controls: list[set[int]] = [set() for _ in range(n_qubits)]  # targets that j controls
    for j, k in directed:
        neighbours[j].add(int(k))
        neighbours[k].add(int(j))
        controls[j].add(int(k))
    triples = [
        (i, j, k)
        for j in range(n_qubits)
        for i, k in combinations(sorted(neighbours[j]), 2)
        if i in controls[j] or k in controls[j]
    ]
    return np.array(triples, dtype=int).reshape(-1, 3)


def count_collisions(
    f01_hz: np.ndarray, directed: np.ndarray, triples: np.ndarray, alpha_hz: float = ALPHA_HZ
) -> dict[int, np.ndarray]:
    """Count Table 1 collisions for one or many frequency samples.

    Each condition "a = b" with bound B counts when |a - b| < B (Hertzberg et al.
    2021, Table 1). Type 4 counts when the target lies outside (f_j12, f_j01).

    Args:
        f01_hz: (..., N) qubit frequencies [Hz]; leading axes are Monte Carlo trials.
        directed: (E, 2) (control, target) pairs from directed_edges.
        triples: (T, 3) (i, j, k) from spectator_triples.
        alpha_hz: anharmonicity [Hz].

    Returns:
        {type number: (...,) int array of counts}.
    """
    f = np.asarray(f01_hz, dtype=float)
    f12 = f + alpha_hz
    f02 = 2 * f + alpha_hz
    j, k = directed[:, 0], directed[:, 1]
    ti, tj, tk = triples[:, 0], triples[:, 1], triples[:, 2]

    def n(mask: np.ndarray) -> np.ndarray:
        return mask.sum(axis=-1)

    return {
        1: n(np.abs(f[..., j] - f[..., k]) < BOUND[1]),
        2: n(np.abs(f02[..., j] - 2 * f[..., k]) < BOUND[2]),
        # Type 3 pairs are unordered nearest neighbours: test both orientations.
        3: n(np.abs(f[..., j] - f12[..., k]) < BOUND[3]) + n(np.abs(f[..., k] - f12[..., j]) < BOUND[3]),
        4: n((f[..., k] < f12[..., j]) | (f[..., j] < f[..., k])),
        5: n(np.abs(f[..., ti] - f[..., tk]) < BOUND[5]),
        6: n((np.abs(f[..., ti] - f12[..., tk]) < BOUND[6]) | (np.abs(f12[..., ti] - f[..., tk]) < BOUND[6])),
        7: n(np.abs(f02[..., tj] - f[..., tk] - f[..., ti]) < BOUND[7]),
    }


def simulate_yield(
    design_hz: np.ndarray,
    edges: np.ndarray,
    sigma_hz: float,
    n_trials: int,
    seed: int,
    alpha_hz: float = ALPHA_HZ,
) -> tuple[float, float]:
    """Monte Carlo collision-free yield (Hertzberg et al. 2021, Sec. IV.3).

    Method: each qubit's f01 is drawn independently from Normal(design, sigma_f);
    count Table 1 collisions; yield = fraction of trials with zero collisions.

    Args:
        design_hz: (N,) design frequencies [Hz].
        edges: (E, 2) coupled pairs.
        sigma_hz: frequency spread sigma_f [Hz].
        n_trials: Monte Carlo repetitions (paper: 1000 by default, Sec. IV.3).
        seed: random seed.

    Returns:
        (collision-free yield, mean number of collisions per trial).
    """
    directed = directed_edges(edges, design_hz)
    triples = spectator_triples(directed, len(design_hz))
    rng = np.random.default_rng(seed)
    f = design_hz + rng.normal(0.0, sigma_hz, size=(n_trials, len(design_hz)))
    total = sum(count_collisions(f, directed, triples, alpha_hz).values())
    return float(np.mean(total == 0)), float(np.mean(total))


SPACING_SWEEP_HZ: np.ndarray = np.arange(30e6, 155e6, 5e6)
"""OUR CHOICE: frequency steps tried (30-150 MHz, 5 MHz grid). Hertzberg et al. 2021
(Sec. IV.3) sweep the spacing "to find the spacing that minimizes 'frequency collisions'
at each value of sigma_f" but do not state the range."""


def best_spacing_yield(
    lattice,
    sigma_hz: float,
    n_trials: int,
    seed: int,
    f1_hz: float = 5e9,
    steps_hz: np.ndarray = SPACING_SWEEP_HZ,
) -> tuple[float, float, float]:
    """Yield at the frequency step that minimises mean collisions (Hertzberg et al. 2021, Sec. IV.3).

    Design: f_n = f1 + (n - 1) * step with f1 = 5 GHz (Sec. IV.3). Ties in mean collisions
    are broken by higher yield. The same seed is used for every step (common random numbers).

    Args:
        lattice: a yieldkit.lattices.Lattice.
        sigma_hz: frequency spread sigma_f [Hz].
        n_trials: Monte Carlo repetitions per step.
        seed: random seed.

    Returns:
        (collision-free yield, mean collisions, chosen step [Hz]).
    """
    from yieldkit.lattices import design_frequencies

    best = None
    for step in steps_hz:
        y, m = simulate_yield(design_frequencies(lattice, f1_hz, step), lattice.edges, sigma_hz, n_trials, seed)
        if best is None or (m, -y) < (best[1], -best[0]):
            best = (y, m, float(step))
    return best


def fit_window(n_qubits: int, sigma_hz: np.ndarray, yields: np.ndarray, min_yield: float = 0.01) -> float:
    """Fit Delta_f of the fixed-window model Y = Phi(Delta_f/sigma_f)^N to Monte Carlo yields.

    Least squares on log(yield) for points with yield >= min_yield, by grid search over
    Delta_f = 1-100 MHz in 10 kHz steps (our choices; the paper does not state its fitting
    method -- Hertzberg et al. 2021, Table 2 / Fig. 4).
    """
    from scipy.stats import norm

    s, y = np.asarray(sigma_hz), np.asarray(yields)
    keep = y >= min_yield
    windows = np.arange(1e6, 100e6, 1e4)
    model = n_qubits * norm.logcdf(windows[:, None] / s[keep][None, :])
    loss = np.sum((model - np.log(y[keep])[None, :]) ** 2, axis=1)
    return float(windows[np.argmin(loss)])


def analytic_yield(n_qubits: int, window_hz: float, sigma_hz: float) -> float:
    """Fixed-window yield model of Hertzberg et al. 2021 (Fig. 4 caption and text).

    Equation: Y = [ Phi(Delta_f / sigma_f) ]^N,  Phi = standard normal CDF
    (the paper writes the integral of exp(-x^2/2) without the 1/sqrt(2 pi); its text
    calls it "the cumulative distribution function", so we use the normalised CDF).
    Delta_f is fitted per lattice (Table 2).
    """
    from scipy.stats import norm

    return float(norm.cdf(window_hz / sigma_hz) ** n_qubits)
