"""Tests for the Step 4 die-ranking pipeline (scenario, ranking, evaluate).

Published numbers used: Hertzberg et al. 2021 (alpha = -330 MHz, 14.5 MHz residual,
5.00/5.07/5.14 GHz example, Rn measurable to better than 0.1%), Lang et al. 2025 (Tc = 0.71 K).
Tolerances are our own choices.
"""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yieldkit.ranking as ranking_module
from yieldkit.collisions import ALPHA_HZ
from yieldkit.evaluate import (
    capture_curve,
    cooldowns_to_n_good,
    evaluate_wafer,
    expected_random_cooldowns,
    true_collision_free,
)
from yieldkit.physics import H_PLANCK, charging_energy, cq_for_anharmonicity, transmon_f01_f12_exact
from yieldkit.ranking import calibrate, order_ours, score_dies
from yieldkit.scenario import RESID_IBM_HZ, RN_RSD_AS_FAB, make_scenario, wafer_config
from yieldkit.wafer import make_wafer

SC = make_scenario()


def test_cq_gives_hertzberg_anharmonicity():
    """Derived Cq: exact anharmonicity -330 MHz at 5.07 GHz (Hertzberg et al. 2021, Table 1 caption)."""
    cq = cq_for_anharmonicity(5.07e9, ALPHA_HZ)
    assert 60e-15 < cq < 75e-15
    ec = charging_energy(cq)
    from scipy.optimize import brentq

    ej = brentq(lambda e: transmon_f01_f12_exact(e, ec)[0] - 5.07e9, 2 * ec, 2000 * ec, xtol=1e-40)
    f01, f12 = transmon_f01_f12_exact(ej, ec)
    assert f12 - f01 == pytest.approx(ALPHA_HZ, abs=0.01e6)


def test_exact_levels_match_scqubits():
    scq = pytest.importorskip("scqubits")
    ec = charging_energy(SC.cq_f)
    ej = 30 * ec
    f01, _ = transmon_f01_f12_exact(ej, ec)
    t = scq.Transmon(EJ=ej / H_PLANCK / 1e9, EC=ec / H_PLANCK / 1e9, ng=0.25, ncut=30, truncated_dim=3)
    ev = t.eigenvals(evals_count=2)
    assert f01 == pytest.approx((ev[1] - ev[0]) * 1e9, abs=1e3)


def test_scenario_design():
    assert SC.lattice.n_qubits == 23
    assert set(np.round(SC.design_hz / 1e6)) == {5000, 5070, 5140}
    by_f = [SC.rn_target_ohm[SC.design_hz == f][0] for f in sorted(set(SC.design_hz))]
    assert np.all(np.diff(by_f) < 0)  # higher design f -> lower target Rn


def test_per_qubit_design_is_the_median():
    obs, truth = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, 0.0), seed=0)
    med = truth.groupby("qubit_id")["rn_true_ohm"].median().to_numpy()
    assert med == pytest.approx(SC.rn_target_ohm, rel=0.01)


def test_calibration_recovers_tc_and_residual():
    obs, truth = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ, tc_true_k=(0.68,)), seed=3)
    cal = calibrate(obs, SC.cq_f)
    assert cal.tc_k == pytest.approx(0.68, abs=0.002)
    assert cal.sigma_qubit_hz == pytest.approx(RESID_IBM_HZ, rel=0.15)


def test_calibration_splits_die_shared_residual():
    cfg = wafer_config(SC, RN_RSD_AS_FAB, 10e6, resid_die_hz=10e6, n_cooled_dies=40)
    obs, _ = make_wafer(cfg, seed=4)
    cal = calibrate(obs, SC.cq_f)
    assert cal.sigma_qubit_hz == pytest.approx(10e6, rel=0.15)
    assert cal.sigma_die_hz == pytest.approx(10e6, rel=0.4)


def test_perfect_information_equals_oracle():
    """Zero residual, zero probe and cryo noise -> P(collision-free) is 0/1 and matches truth exactly."""
    cfg = replace(wafer_config(SC, RN_RSD_AS_FAB, 0.0, probe_noise_rsd=0.0), f01_residual_rsd=0.0)
    obs, truth = make_wafer(cfg, seed=5)
    res = evaluate_wafer(obs, truth, SC, n_samples=20)
    assert res["n_good"] > 0
    assert res["gain_ours"] == pytest.approx(1.0)
    assert res["cool_ours"] == res["cool_oracle"]


def test_random_order_is_diagonal_on_average():
    obs, truth = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=6)
    good = true_collision_free(truth, SC.design_hz, SC.lattice.edges)
    rng = np.random.default_rng(0)
    keys = list(good.index)
    curves = [capture_curve([keys[i] for i in rng.permutation(len(keys))], good)[1] for _ in range(300)]
    x = capture_curve(keys, good)[0]
    assert np.max(np.abs(np.mean(curves, axis=0) - x)) < 0.05


def test_cooldown_helpers():
    good = pd.Series([False, True, False, True], index=pd.MultiIndex.from_tuples([(0, i) for i in range(4)]))
    assert cooldowns_to_n_good(list(good.index), good, 2) == 4
    assert np.isnan(cooldowns_to_n_good(list(good.index), good, 3))
    assert expected_random_cooldowns(4, 2, 1) == pytest.approx(5 / 3)


def test_ranking_runs_without_any_truth_columns():
    obs, _ = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=7)
    assert not any("true" in c for c in obs.columns)
    cal = calibrate(obs, SC.cq_f)
    scores = score_dies(obs, cal, SC.design_hz, SC.lattice.edges, n_samples=20)
    assert len(order_ours(scores)) == obs.loc[~obs.is_cooled, ["wafer_id", "die_id"]].drop_duplicates().shape[0]


def test_ranking_module_never_names_truth():
    """By construction: no identifier or string literal in ranking.py refers to truth columns."""
    tree = ast.parse(Path(ranking_module.__file__).read_text(encoding="utf-8"))
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.append(n.id)
        elif isinstance(n, ast.Attribute):
            names.append(n.attr)
        elif isinstance(n, ast.arg):
            names.append(n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            names.append(n.value)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names += [a.name for a in n.names] + [getattr(n, "module", "") or ""]
    assert not [s for s in names if "true" in s.lower() or "truth" in s.lower() or "evaluate" in s.lower()]
