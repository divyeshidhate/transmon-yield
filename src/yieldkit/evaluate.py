"""Score die rankings against the hidden ground truth (synthetic data only).

This is the only module that reads the `truth` table. Rankings are produced by
yieldkit.ranking from the `observed` table alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yieldkit.collisions import count_collisions, directed_edges, spectator_triples
from yieldkit.ranking import (
    DIE,
    calibrate,
    naive_scores,
    order_naive,
    order_ours,
    score_dies,
)
from yieldkit.scenario import Scenario


def true_collision_free(truth: pd.DataFrame, design_hz: np.ndarray, edges: np.ndarray) -> pd.Series:
    """Is each die truly collision-free? (Table 1 of Hertzberg et al. 2021 on f01_true.)"""
    n_q = len(design_hz)
    t = truth.sort_values(DIE + ["qubit_id"])
    f = t["f01_true_hz"].to_numpy().reshape(-1, n_q)
    directed = directed_edges(edges, design_hz)
    total = sum(count_collisions(f, directed, spectator_triples(directed, n_q)).values())
    idx = pd.MultiIndex.from_frame(t[DIE].drop_duplicates())
    return pd.Series(total == 0, index=idx, name="collision_free")


def capture_curve(order: list[tuple[int, int]], good: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of dies cooled (x) vs fraction of truly good dies captured (y), starting at (0, 0)."""
    g = good.loc[order].to_numpy().astype(float)
    x = np.arange(len(g) + 1) / len(g)
    y = np.concatenate([[0.0], np.cumsum(g)]) / max(g.sum(), 1.0)
    return x, y


def _auc(order: list[tuple[int, int]], good: pd.Series) -> float:
    x, y = capture_curve(order, good)
    return float(np.trapezoid(y, x))


def ranking_gain(order: list[tuple[int, int]], good: pd.Series) -> float:
    """(AUC - AUC_random) / (AUC_oracle - AUC_random): 0 = random, 1 = oracle (our metric).

    AUC_random = 0.5 (the expected capture curve of a random order is the diagonal).
    """
    sub = good.loc[order]
    if sub.sum() == 0 or sub.all():
        return float("nan")
    oracle = list(sub.sort_values(ascending=False, kind="stable").index)
    return (_auc(order, good) - 0.5) / (_auc(oracle, good) - 0.5)


def cooldowns_to_n_good(order: list[tuple[int, int]], good: pd.Series, n: int) -> float:
    """Number of dies cooled (in this order) until n good ones are found; NaN if never."""
    c = np.cumsum(good.loc[order].to_numpy())
    hit = np.flatnonzero(c >= n)
    return float(hit[0] + 1) if hit.size else float("nan")


def expected_random_cooldowns(n_dies: int, n_good: int, n: int) -> float:
    """Expected dies cooled at random until n good are found: n (N + 1) / (G + 1) (negative hypergeometric mean)."""
    return n * (n_dies + 1) / (n_good + 1) if n_good >= n else float("nan")


def evaluate_wafer(
    observed: pd.DataFrame,
    truth: pd.DataFrame,
    sc: Scenario,
    n_good_target: int = 5,
    n_samples: int = 200,
    seed: int = 0,
    calibration=None,
) -> dict:
    """Calibrate (unless given), rank, and score one wafer against truth.

    Returns a dict with base yield, calibration, ranking gains and cooldowns-to-N for
    ours / naive / random / oracle, and the absolute prediction bias on uncooled qubits.
    """
    cal = calibration if calibration is not None else calibrate(observed, sc.cq_f)
    scores = score_dies(observed, cal, sc.design_hz, sc.lattice.edges, n_samples, seed)
    good = true_collision_free(truth, sc.design_hz, sc.lattice.edges)
    ours = order_ours(scores)
    naive = order_naive(naive_scores(observed, sc.rn_target_ohm))
    pool = good.loc[ours]
    oracle = list(pool.sort_values(ascending=False, kind="stable").index)

    from yieldkit.ranking import predict_f01

    unc = observed[~observed["is_cooled"]]
    merged = unc.merge(truth[DIE + ["qubit_id", "f01_true_hz"]], on=DIE + ["qubit_id"])
    err = predict_f01(merged, cal) - merged["f01_true_hz"].to_numpy()
    return {
        "n_dies": len(pool),
        "n_good": int(pool.sum()),
        "base_yield": float(pool.mean()),
        "tc_fit_k": cal.tc_k,
        "sigma_qubit_est_mhz": cal.sigma_qubit_hz / 1e6,
        "sigma_die_est_mhz": cal.sigma_die_hz / 1e6,
        "mean_abs_error_mhz": float(np.mean(np.abs(err))) / 1e6,
        "mean_bias_mhz": float(np.mean(err)) / 1e6,
        "gain_ours": ranking_gain(ours, good),
        "gain_naive": ranking_gain(naive, good),
        "cool_ours": cooldowns_to_n_good(ours, good, n_good_target),
        "cool_naive": cooldowns_to_n_good(naive, good, n_good_target),
        "cool_random": expected_random_cooldowns(len(pool), int(pool.sum()), n_good_target),
        "cool_oracle": cooldowns_to_n_good(oracle, good, n_good_target),
        "_orders": {"ours": ours, "naive": naive, "oracle": oracle},
        "_good": good,
    }
