"""Tests for yieldkit.wafer (synthetic wafer).

Published numbers used:
- Lang et al., Phys. Rev. Applied 24, 054052 (2025), arXiv:2504.18173: 12.4% Rn RSD, mean 6.89 kOhm,
  375 dies of 10 x 7 mm (Sec. II, III.2).
- Krizan et al. 2026 (DOI 10.1088/2633-4356/ae7fe1), citing their ref. [9]: ~2% RSD.
- Mayer, Weber et al., arXiv:2505.08424: 4 qubits per chip; 1.6% average prediction error.
Tolerances below are our own choices.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from yieldkit.wafer import (
    F01_RESIDUAL_RSD_TUNED,
    KEYS,
    RSD_LANG_2025,
    RSD_MODERN,
    WaferConfig,
    die_layout,
    make_wafer,
    mean_abs_rel_error,
    tune_f01_residual,
)

CFG = WaferConfig()


def test_seed_reproduces_exactly():
    obs1, tr1 = make_wafer(CFG, seed=42)
    obs2, tr2 = make_wafer(CFG, seed=42)
    pd.testing.assert_frame_equal(obs1, obs2)
    pd.testing.assert_frame_equal(tr1, tr2)
    obs3, _ = make_wafer(CFG, seed=43)
    assert not np.allclose(obs1["rn_measured_ohm"], obs3["rn_measured_ohm"])


@pytest.mark.parametrize("rsd", [RSD_LANG_2025, RSD_MODERN])
def test_generated_rsd_within_10pct_of_target(rsd):
    _, truth = make_wafer(replace(CFG, rn_rsd=rsd), seed=1)
    rn = truth["rn_true_ohm"]
    assert rn.std() / rn.mean() == pytest.approx(rsd, rel=0.10)


def test_mean_rn_near_lang():
    _, truth = make_wafer(CFG, seed=1)
    assert truth["rn_true_ohm"].mean() == pytest.approx(6.89e3, rel=0.03)


def test_truth_never_in_observed():
    obs, truth = make_wafer(CFG, seed=3)
    shared = set(obs.columns) & set(truth.columns)
    assert shared == set(KEYS)
    # Probe noise: measured Rn differs from true Rn for every junction.
    assert np.all(obs["rn_measured_ohm"].to_numpy() != truth["rn_true_ohm"].to_numpy())
    # No observed numeric column is an exact copy of any truth column.
    for oc in obs.columns.difference(KEYS):
        for tc in truth.columns.difference(KEYS):
            assert not np.array_equal(obs[oc].to_numpy(), truth[tc].to_numpy())


def test_f01_measured_only_on_cooled_dies():
    obs, _ = make_wafer(CFG, seed=4)
    assert obs.loc[~obs["is_cooled"], "f01_measured_hz"].isna().all()
    assert obs.loc[obs["is_cooled"], "f01_measured_hz"].notna().all()
    assert obs["is_cooled"].sum() == CFG.n_cooled_dies_per_wafer * CFG.qubits_per_die  # 40 qubits


def test_die_count_near_lang_375():
    """4 mm edge exclusion is our assumption, chosen to land near Lang's 375 dies."""
    assert 360 <= len(die_layout(CFG)) <= 390


def test_dies_inside_edge_exclusion():
    d = die_layout(CFG)
    r_max = CFG.wafer_diameter_m / 2 - CFG.edge_exclusion_m
    for sx in (-1, 1):
        for sy in (-1, 1):
            r = np.hypot(d["x_m"] + sx * CFG.die_width_m / 2, d["y_m"] + sy * CFG.die_height_m / 2)
            assert (r <= r_max + 1e-12).all()


def test_four_qubits_per_die():
    obs, _ = make_wafer(CFG, seed=5)
    assert (obs.groupby(["wafer_id", "die_id"]).size() == 4).all()


def test_per_wafer_tc():
    cfg = replace(CFG, n_wafers=2, tc_eff_true_k=(0.71, 0.68))
    _, truth = make_wafer(cfg, seed=6)
    assert truth.groupby("wafer_id")["tc_eff_true_k"].first().tolist() == [0.71, 0.68]


def test_tuned_residual_gives_1p6_percent():
    """Default residual was TUNED so mean |error| ~ 1.6% (arXiv:2505.08424); check on a large sample."""
    big = replace(CFG, n_wafers=10, n_cooled_dies_per_wafer=10**6)
    obs, _ = make_wafer(big, seed=7)
    assert mean_abs_rel_error(obs, 0.71, CFG.cq_design_f) == pytest.approx(0.016, abs=0.001)


def test_tuning_function_recovers_default():
    assert tune_f01_residual(CFG) == pytest.approx(F01_RESIDUAL_RSD_TUNED, abs=0.0005)
