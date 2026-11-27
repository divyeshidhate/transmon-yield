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
    """Zero residual, zero probe and cryo noise -> P(collision-free) is 0/1 and matches truth exactly.
    Needs the scorer and the truth to share the alpha model (Eq. 4.11 from EJ and EC, Step 8c)."""
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


# ---------------- Step 6a: capacitor (EC) scatter hidden from the RT observation ----------------

from yieldkit.evaluate import true_alpha_hz  # noqa: E402
from yieldkit.physics import f01_ec_sensitivity  # noqa: E402
from yieldkit.scenario import EC_RSD_OSMAN  # noqa: E402


def test_ec_sensitivity_matches_numerical_derivative():
    """d ln f / d ln EC of the transmon formula (Koch et al. 2007): analytic vs finite difference."""
    from yieldkit.physics import transmon_f01

    ec = charging_energy(SC.cq_f)
    f = 5.07e9
    ej = ((f * H_PLANCK + ec) ** 2) / (8 * ec)  # invert f = (sqrt(8 EJ EC) - EC)/h
    h = 1e-6
    num = (np.log(transmon_f01(ej, ec * (1 + h))) - np.log(transmon_f01(ej, ec * (1 - h)))) / (2 * h)
    assert f01_ec_sensitivity(f, ec) == pytest.approx(num, rel=1e-6)
    assert f01_ec_sensitivity(f, ec) == pytest.approx(0.472, abs=0.001)


def _expected_ec_extra_hz() -> float:
    """RMS over the 23 qubits of EC_rsd * sensitivity * f (each qubit at its own set-point)."""
    ec = charging_energy(SC.cq_f)
    per_q = EC_RSD_OSMAN * np.array([f01_ec_sensitivity(f, ec) for f in SC.design_hz]) * SC.design_hz
    return float(np.sqrt(np.mean(per_q**2)))


def test_ec_scatter_grows_residual_by_expected_amount():
    """Deterministic: seed 11, 100 cooled dies (2300 qubits), IBM 14.5 MHz other-residual.

    Osman et al. 2023's 0.8% EC scatter should add sqrt(sigma_other^2 + ~19.2^2) in quadrature.
    Tolerances (our choice): extra residual within 7%, total within 3%.
    """
    expected_extra = _expected_ec_extra_hz()
    assert expected_extra == pytest.approx(19.2e6, abs=0.1e6)
    base = wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ, n_cooled_dies=100)
    c0 = calibrate(make_wafer(base, seed=11)[0], SC.cq_f)
    with_ec = wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ, n_cooled_dies=100, ec_rsd=EC_RSD_OSMAN)
    c1 = calibrate(make_wafer(with_ec, seed=11)[0], SC.cq_f)
    extra = np.sqrt(c1.sigma_qubit_hz**2 - c0.sigma_qubit_hz**2)
    assert extra == pytest.approx(expected_extra, rel=0.07)
    assert c1.sigma_qubit_hz == pytest.approx(np.hypot(c0.sigma_qubit_hz, expected_extra), rel=0.03)


def test_ec_die_split_moves_variance_to_die_part():
    """With half the EC variance die-shared, calibration sees ~half of it per qubit and ~half per die."""
    cfg = wafer_config(SC, RN_RSD_AS_FAB, 0.0, n_cooled_dies=100, ec_rsd=EC_RSD_OSMAN, ec_die_frac=0.5)
    cal = calibrate(make_wafer(cfg, seed=11)[0], SC.cq_f)
    half = _expected_ec_extra_hz() / np.sqrt(2)
    assert cal.sigma_die_hz == pytest.approx(half, rel=0.25)
    assert cal.sigma_qubit_hz < 0.8 * _expected_ec_extra_hz()


def test_true_alpha_follows_ec_and_ej():
    """Step 8c: alpha_q from thesis Eq. 4.11 with the qubit's own EJ and EC (Krizan, PhD thesis, Chalmers 2026).
    (i) a qubit exactly on design keeps alpha = -330 MHz (Hertzberg et al. 2021, Table 1 caption);
    (ii) with 0.8% EC scatter (Osman et al. 2023) the alpha spread is ~0.9%, not 0.8%: Eq. 4.11 gives
         d ln|alpha| / d ln EC = 1.09 and d ln|alpha| / d ln EJ = -0.09 at this design (EJ/EC ~ 46), so
         sqrt((1.09 * 0.8%)^2 + (0.09 * 2% Rn)^2) = 0.90% (band 0.85-0.95%: our tolerance);
    (iii) without EC scatter, alpha now moves with the Rn scatter through EJ, but by < 1 MHz rms (our bound)."""
    from yieldkit.scenario import TC_NOMINAL_K

    on_design = pd.DataFrame({"qubit_id": np.arange(len(SC.design_hz)), "rn_cooldown_ohm": SC.rn_target_ohm,
                              "tc_eff_true_k": TC_NOMINAL_K, "cq_true_f": SC.cq_f})
    assert np.allclose(true_alpha_hz(on_design, SC.cq_f, SC.design_hz), ALPHA_HZ, rtol=1e-9)

    obs, truth = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, 0.0, ec_rsd=EC_RSD_OSMAN), seed=12)
    alpha = true_alpha_hz(truth, SC.cq_f, SC.design_hz)
    assert 0.85e-2 < np.std(alpha) / abs(np.mean(alpha)) < 0.95e-2

    obs0, truth0 = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=12)
    a0 = true_alpha_hz(truth0, SC.cq_f, SC.design_hz)
    assert 0 < np.std(a0) < 1e6


# ---------------- Step 6b: junction ageing between RT probe and cooldown ----------------

from yieldkit.ranking import predict_f01  # noqa: E402


def _f_matrix(truth: pd.DataFrame, col: str = "f01_true_hz") -> np.ndarray:
    return truth.sort_values(["wafer_id", "die_id", "qubit_id"])[col].to_numpy().reshape(-1, SC.lattice.n_qubits)


def _counts(f: np.ndarray) -> np.ndarray:
    from yieldkit.collisions import count_collisions, directed_edges, spectator_triples

    d = directed_edges(SC.lattice.edges, SC.design_hz)
    return sum(count_collisions(f, d, spectator_triples(d, SC.lattice.n_qubits)).values())


def test_additive_common_offset_leaves_collisions_identical():
    """Table 1 conditions compare frequencies on one chip with a fixed alpha, so adding the SAME
    offset to every qubit of a die cannot change any collision count (exact)."""
    _, truth = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=0)
    f = _f_matrix(truth)
    offsets = np.random.default_rng(0).normal(0, 100e6, (f.shape[0], 1))
    assert np.array_equal(_counts(f), _counts(f + offsets))


def test_shared_drift_compresses_gaps_and_biases_absolute_prediction():
    """A shared Rn increase d is NOT a pure common offset: f + EC/h scales as (1+d)^(-1/2), so the
    gaps between qubits shrink by that factor while alpha (set by EC) does not. Expected:
    (i) RT-explained gaps scale by (1+d)^(-1/2) -- exactly for Eq. 1; with the second-order terms of
    the truth model (Krizan thesis 2026, Eq. 4.3) to within 0.1% (measured: 45 kHz, 0.017%);
    (ii) with a calibration from an un-aged wafer, prediction is biased by ~(f + EC/h) d / 2;
    (iii) recalibrating absorbs it exactly (Tc -> Tc/(1+d)), because Rn and Tc enter EJ only as Tc/Rn."""
    d = 0.04  # Osman et al. 2023: +4% over 21 days
    o0, t0 = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=0)
    o1, t1 = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ, drift_shared_rel=d), seed=0)
    resid = _f_matrix(t0, "f01_resid_qubit_hz") + _f_matrix(t0, "f01_resid_die_hz")
    F0, F1 = _f_matrix(t0) - resid, _f_matrix(t1) - resid  # RT-explained part
    e = SC.lattice.edges
    assert np.allclose(F1[:, e[:, 0]] - F1[:, e[:, 1]], (F0[:, e[:, 0]] - F0[:, e[:, 1]]) * (1 + d) ** -0.5, rtol=1e-3)

    cal0, cal1 = calibrate(o0, SC.cq_f), calibrate(o1, SC.cq_f)
    unc = ~o1["is_cooled"].to_numpy()
    true_f = t1["f01_true_hz"].to_numpy()[unc]
    ec_hz = charging_energy(SC.cq_f) / H_PLANCK
    expected_bias = np.mean((SC.design_hz + ec_hz) / 2) * d
    assert np.mean(predict_f01(o1[unc], cal0) - true_f) == pytest.approx(expected_bias, rel=0.05)
    assert cal1.tc_k == pytest.approx(cal0.tc_k / (1 + d), rel=1e-3)
    assert abs(np.mean(predict_f01(o1[unc], cal1) - true_f)) < 2e6


def test_junction_drift_adds_residual_in_quadrature():
    """Per-junction drift s (ASSUMED value) shifts f by (f + EC/h)/2 * s per qubit (d f / d ln Rn).
    Deterministic: seed 11, 100 cooled dies; tolerance 7% (our choice)."""
    s = 0.01
    ec_hz = charging_energy(SC.cq_f) / H_PLANCK
    expected = s * np.sqrt(np.mean(((SC.design_hz + ec_hz) / 2) ** 2))
    c0 = calibrate(make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ, n_cooled_dies=100), seed=11)[0], SC.cq_f)
    cfg = wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ, n_cooled_dies=100, drift_junction_rsd=s)
    c1 = calibrate(make_wafer(cfg, seed=11)[0], SC.cq_f)
    assert np.sqrt(c1.sigma_qubit_hz**2 - c0.sigma_qubit_hz**2) == pytest.approx(expected, rel=0.07)


def test_defaults_headline_robust_properties():
    """Notebook 04 defaults (no EC scatter, no ageing), seeds 0-4: properties that hold on every seed, not
    single-seed values. Over 10 seeds (Step 8c, 2026-11-27) gain was 0.631 +/- 0.086 (min 0.50) and our
    ranking beat naive and random on 10/10; one die flipping at a +/-4 MHz window edge moves a single seed's
    cooldowns-to-5 by up to 20, so exact per-seed values are not pinned."""
    for seed in range(5):
        obs, truth = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=seed)
        r = evaluate_wafer(obs, truth, SC, seed=seed)
        assert r["n_good"] >= 5
        assert r["gain_ours"] > r["gain_naive"]
        assert r["gain_ours"] > 0.4  # our floor, below the 10-seed minimum 0.50
        assert r["cool_ours"] < r["cool_random"]
        assert r["cool_oracle"] == 5


def test_fitted_tc_absorbs_cold_warm_ratio():
    """Step 8b: if the truth's cold resistance is k * R_RT, the Tc fitted on R_RT comes out as Tc/k and the
    predictions are unchanged -- a separate k parameter could not be fitted. k = 1.1375 (Krizan thesis p. 49)."""
    k = 1.1375
    obs, _ = make_wafer(wafer_config(SC, RN_RSD_AS_FAB, RESID_IBM_HZ), seed=0)
    cal = calibrate(obs, SC.cq_f)
    scaled = obs.assign(rn_measured_ohm=obs["rn_measured_ohm"] * k)  # analyst given cold-like resistances
    cal_k = calibrate(scaled, SC.cq_f)
    assert cal_k.tc_k == pytest.approx(cal.tc_k * k, rel=1e-5)
    assert np.allclose(predict_f01(scaled, cal_k), predict_f01(obs, cal), rtol=1e-6)
