"""Rank dies for cooldown from MEASURED data only.

Inputs are the analyst's `observed` table (RT resistance for every qubit, cryo f01 for
the few cooled calibration dies) and the fixed chip design. This module never reads the
hidden ground truth; scoring against truth lives in yieldkit.evaluate.

Scope: we rank dies from measured data for a fixed chip design. The collision checker is
used only to score predictions for measured dies; this code does not choose or optimise
chip designs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from yieldkit.collisions import ALPHA_HZ, count_collisions, directed_edges, spectator_triples
from yieldkit.physics import (
    alpha_scaled_to_design,
    charging_energy,
    critical_current,
    f01_from_rn_second_order,
    gap_from_tc,
    josephson_energy,
)

DIE = ["wafer_id", "die_id"]


@dataclass(frozen=True)
class Calibration:
    """What we learn from the cooled dies."""

    tc_k: float  # fitted effective Tc
    cq_f: float  # capacitance used (design value, not fitted)
    sigma_qubit_hz: float  # within-die, per-qubit residual std (drives collisions)
    sigma_die_hz: float  # die-shared residual std (absolute error only)
    n_dies: int


def calibrate(observed: pd.DataFrame, cq_f: float, tc_bounds_k: tuple[float, float] = (0.3, 1.5)) -> Calibration:
    """Fit effective Tc on cooled dies, then split the residual into die-shared and per-qubit parts.

    1. Tc: least squares  min_Tc sum (f_meas - f2(Rn_meas; Tc, Cq))^2, f2 = Ambegaokar-Baratoff +
       second-order transmon formula (Krizan, PhD thesis, Chalmers 2026, Eq. 4.3; Eq. 1 of
       Mayer, Weber et al., arXiv:2505.08424 with the next two terms; Tc as the only free parameter,
       as in Lang et al. 2025, Sec. IV.2). The fitted Tc is effective: Tc_eff = Tc / (R_cold/R_RT),
       because Tc and the cold/warm resistance ratio enter EJ only as Tc / (ratio * R_RT)
       (yieldkit.physics.gap_from_tc). One design here, so one scale.
    2. Residual r = f_meas - f_pred, one-way random-effects split (standard ANOVA estimators):
         sigma_qubit^2 = pooled within-die variance (dof = n - 1 per die)
         sigma_die^2   = max(0, var(die means) - sigma_qubit^2 / n)

    Args:
        observed: analyst table (only rows with is_cooled are used).
        cq_f: qubit capacitance [F].

    Returns:
        Calibration.
    """
    cooled = observed[observed["is_cooled"]]
    rn = cooled["rn_measured_ohm"].to_numpy()
    fm = cooled["f01_measured_hz"].to_numpy()

    def sse(tc: float) -> float:
        return float(np.sum((f01_from_rn_second_order(rn, cq_f, gap_from_tc(tc)) - fm) ** 2))

    tc = float(minimize_scalar(sse, bounds=tc_bounds_k, method="bounded", options={"xatol": 1e-7}).x)
    r = pd.Series(fm - f01_from_rn_second_order(rn, cq_f, gap_from_tc(tc)), index=cooled.index)
    groups = r.groupby([cooled["wafer_id"], cooled["die_id"]])
    n = groups.size().to_numpy()
    within_ss = float(((r - groups.transform("mean")) ** 2).sum())
    sigma_q2 = within_ss / float(np.sum(n - 1))
    means = groups.mean().to_numpy()
    sigma_d2 = max(0.0, float(np.var(means, ddof=1)) - sigma_q2 / float(np.mean(n))) if len(means) > 1 else 0.0
    return Calibration(tc, cq_f, float(np.sqrt(sigma_q2)), float(np.sqrt(sigma_d2)), len(means))


def predict_f01(observed: pd.DataFrame, cal: Calibration) -> np.ndarray:
    """Predicted f01 [Hz] for every row: second-order formula with measured Rn and calibrated Tc."""
    return f01_from_rn_second_order(observed["rn_measured_ohm"].to_numpy(), cal.cq_f, gap_from_tc(cal.tc_k))


def _uncooled_matrix(observed: pd.DataFrame, n_qubits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uncooled rows sorted by die then qubit, plus the die index table."""
    unc = observed[~observed["is_cooled"]].sort_values(DIE + ["qubit_id"])
    dies = unc[DIE].drop_duplicates().reset_index(drop=True)
    if len(unc) != len(dies) * n_qubits:
        raise ValueError("every uncooled die must have all qubits")
    return unc, dies


def score_dies(
    observed: pd.DataFrame,
    cal: Calibration,
    design_hz: np.ndarray,
    edges: np.ndarray,
    n_samples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """P(collision-free) and expected collisions for every uncooled die.

    For each die, draw n_samples chips from the predictive distribution
        f_q = f_pred,q + d + e_q,   d ~ N(0, sigma_die),  e_q ~ N(0, sigma_qubit)
    and count Table 1 collisions (Hertzberg et al. 2021) with the fixed design's control/target roles.
    Each qubit's anharmonicity is predicted from its measured Rn (calibrated EJ) and the design EC with
    Eq. 4.11 of Krizan, PhD thesis, Chalmers 2026 (physics.alpha_scaled_to_design): -330 MHz on design
    (Hertzberg Table 1 caption), moving with EJ through xi. Same physics as the truth; only measured data used.

    Returns:
        DataFrame: wafer_id, die_id, p_collision_free, expected_collisions.
    """
    n_q = len(design_hz)
    unc, dies = _uncooled_matrix(observed, n_q)
    f_pred = predict_f01(unc, cal).reshape(len(dies), 1, n_q)
    ej = josephson_energy(critical_current(unc["rn_measured_ohm"].to_numpy(), gap_from_tc(cal.tc_k)))
    f_d = np.asarray(design_hz)[unc["qubit_id"].to_numpy()]
    ec_d = charging_energy(cal.cq_f)
    alpha = alpha_scaled_to_design(ej, ec_d, f_d, ec_d, ALPHA_HZ).reshape(len(dies), 1, n_q)
    rng = np.random.default_rng(seed)
    shape = (len(dies), n_samples)
    f = (
        f_pred
        + rng.normal(0.0, cal.sigma_die_hz, shape)[..., None]
        + rng.normal(0.0, cal.sigma_qubit_hz, shape + (n_q,))
    )
    directed = directed_edges(edges, design_hz)
    triples = spectator_triples(directed, n_q)
    total = sum(count_collisions(f, directed, triples, alpha).values())
    out = dies.copy()
    out["p_collision_free"] = np.mean(total == 0, axis=1)
    out["expected_collisions"] = np.mean(total, axis=1)
    return out


def naive_scores(observed: pd.DataFrame, rn_target_ohm: np.ndarray) -> pd.DataFrame:
    """Baseline: sum over qubits of |Rn_meas - Rn_target| / Rn_target per uncooled die (lower = better)."""
    unc, dies = _uncooled_matrix(observed, len(rn_target_ohm))
    tgt = rn_target_ohm[unc["qubit_id"].to_numpy()]
    dev = np.abs(unc["rn_measured_ohm"].to_numpy() - tgt) / tgt
    out = dies.copy()
    out["naive_score"] = dev.reshape(len(dies), -1).sum(axis=1)
    return out


def order_ours(scores: pd.DataFrame) -> list[tuple[int, int]]:
    """Cooling order: highest P(collision-free) first, ties by fewest expected collisions."""
    s = scores.sort_values(["p_collision_free", "expected_collisions"], ascending=[False, True], kind="stable")
    return list(map(tuple, s[DIE].to_numpy()))


def order_naive(naive: pd.DataFrame) -> list[tuple[int, int]]:
    """Cooling order for the naive baseline: smallest total |Rn - target| first."""
    s = naive.sort_values("naive_score", kind="stable")
    return list(map(tuple, s[DIE].to_numpy()))
