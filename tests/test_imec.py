"""Tests for notebook 05 (imec test arrays vs qubits): yieldkit.imec and the two physics additions.

Published numbers used:
- Mayer, Weber et al., arXiv:2505.08424 (2025): 7.1 kOhm, Cq = 86 fF, Tc = 0.71 K -> 4.42 GHz design.
- Hertzberg et al., npj QI 7, 129 (2021), Table 1: alpha = -330 MHz.
- Van Damme, Zenodo doi:10.5281/zenodo.13143313 (2024), README.txt: 7872 wafer-map junctions,
  393 functional qubits (data of Van Damme et al., Nature 634, 74-79 (2024)).
The statistics functions are checked on synthetic data with planted effects; tolerances are our choices.
"""

import urllib.error

import numpy as np
import pandas as pd
import pytest

from yieldkit.imec import (
    add_design_offset_lodo,
    decompose_wafer,
    fit_tc,
    good_junctions,
    load_imec,
    same_design_across_dies,
    split_die_within,
)
from yieldkit.physics import (
    charging_energy,
    cq_for_anharmonicity,
    f01_from_rn,
    f01_from_rn_exact,
    gap_from_tc,
    transmon_f01_f12_exact,
)


# ---------------------------------------------------------------- physics


def test_f01_exact_at_fraunhofer_design_point():
    """7.1 kOhm, 86 fF, Tc 0.71 K: the paper's 4.42 GHz design within 1%, and 5-25 MHz below the
    asymptotic Eq. 1 (the ~13 MHz gap found in Step 1)."""
    gap = gap_from_tc(0.71)
    exact = f01_from_rn_exact(7.1e3, 86e-15, gap)
    assert exact == pytest.approx(4.42e9, rel=0.01)
    assert 5e6 < f01_from_rn(7.1e3, 86e-15, gap) - exact < 25e6


def test_f01_exact_vectorised_matches_scalar():
    gap = gap_from_tc(0.71)
    rn, cq = np.array([6e3, 7e3, 8e3]), np.array([80e-15, 86e-15, 90e-15])
    vec = f01_from_rn_exact(rn, cq, gap)
    assert vec.shape == (3,)
    assert vec[1] == pytest.approx(f01_from_rn_exact(7e3, 86e-15, gap), rel=1e-12)


def test_cq_bracket_default_unchanged_and_low_f_works():
    """Default bracket still gives the Step 4 chip (alpha = -330 MHz at 5.07 GHz, Hertzberg Table 1).
    A low-frequency qubit (3.6 GHz, -250 MHz) needs a bracket near the alpha ~ -EC guess; check the root
    reproduces the target alpha by forward calculation."""
    c_default = cq_for_anharmonicity(5.07e9, -330e6)
    c0 = 1.602176634e-19**2 / (2 * 6.62607015e-34 * 330e6)
    assert cq_for_anharmonicity(5.07e9, -330e6, (0.8 * c0, 1.3 * c0)) == pytest.approx(c_default, rel=1e-9)
    with pytest.raises(ValueError):
        cq_for_anharmonicity(3.6e9, -250e6)  # default bracket reaches the charge regime
    c0 = 1.602176634e-19**2 / (2 * 6.62607015e-34 * 250e6)
    c = cq_for_anharmonicity(3.6e9, -250e6, (0.8 * c0, 1.3 * c0))
    ec = charging_energy(c)
    from scipy.optimize import brentq

    ej = brentq(lambda e: transmon_f01_f12_exact(e, ec)[0] - 3.6e9, 2 * ec, 2000 * ec, xtol=1e-40)
    f01, f12 = transmon_f01_f12_exact(ej, ec)
    assert f12 - f01 == pytest.approx(-250e6, abs=0.1e6)


# ---------------------------------------------------------------- statistics on synthetic data


def _synthetic_map(seed=0, s_trend=0.08, s_die=0.03, s_within=0.06, copies=8):
    """82-die-like grid; ln RA = quadratic radial trend + die effect + junction scatter."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(-4, 5), np.arange(-5, 6))
    keep = xs**2 / 20.0 + ys**2 / 30.0 <= 1.0
    xs, ys = xs[keep], ys[keep]
    t = xs**2 + ys**2
    t = (t - t.mean()) / t.std() * s_trend
    b = rng.normal(0, s_die, len(xs))
    rows = []
    for x, y, tt, bb in zip(xs, ys, t, b):
        for _ in range(copies):
            rows.append((x, y, 100, np.exp(tt + bb + rng.normal(0, s_within))))
    w = pd.DataFrame(rows, columns=["die_x", "die_y", "cd_nm", "ra_ohm_m2"])
    d = pd.DataFrame({"die": np.repeat([0, 1, 2], 400), "cd_nm": 100,
                      "ra_ohm_m2": np.exp(rng.normal(0, s_within, 1200) + np.repeat(rng.normal(0, 0.05, 3), 400))})
    return w, d


def test_decompose_recovers_planted_sigmas():
    w, d = _synthetic_map()
    r = decompose_wafer(w, d)
    assert r["sigma_trend"] == pytest.approx(0.08, rel=0.15)
    assert r["sigma_die"] == pytest.approx(0.03, rel=0.35)
    assert r["sigma_within_diced"] == pytest.approx(0.06, rel=0.08)  # die offsets of the diced dies removed
    assert r["sigma_within_copies"] == pytest.approx(0.06, rel=0.08)
    assert r["share_trend"] + r["share_die"] + r["share_within"] == pytest.approx(1.0)


def test_good_junctions_drops_opens_and_shorts():
    w, _ = _synthetic_map()
    w.loc[0, "ra_ohm_m2"] = 1e30 * 1e-13  # open
    w.loc[1, "ra_ohm_m2"] = 1e-6 * w["ra_ohm_m2"].median()  # short
    ok = good_junctions(w)
    assert not ok[0] and not ok[1]
    assert ok.sum() == len(w) - 2


def _synthetic_qubits(seed=1, s_die=100e6, s_q=250e6, n_dies=35, per_die=11):
    rng = np.random.default_rng(seed)
    designs = [f"D{k}" for k in range(15)]
    off = dict(zip(designs, rng.normal(0, 800e6, 15)))
    rows = []
    for d in range(n_dies):
        b = rng.normal(0, s_die)
        for des in rng.choice(designs, per_die, replace=False):
            rows.append((f"die{d}", des, off[des] + b + rng.normal(0, s_q)))
    return pd.DataFrame(rows, columns=["die", "design", "resid_hz"])


def test_split_and_pair_estimates_recover_planted_parts():
    """Average over 10 synthetic wafers (one wafer of 35 dies scatters by ~18% in sigma_die)."""
    est = []
    for seed in range(10):
        q = add_design_offset_lodo(_synthetic_qubits(seed=seed))
        s = split_die_within(q["resid_design_hz"], q["die"])
        c = same_design_across_dies(q["resid_hz"], q["die"], q["design"])  # centres each design itself
        assert s["n_dies"] == 35
        est.append([s["sigma_within"], s["sigma_die"], c["sigma_die_pairs"], c["sigma_design_across_dies"]])
    within, die, die_pairs, across = np.mean(est, axis=0)
    assert within == pytest.approx(250e6, rel=0.06)
    assert die == pytest.approx(100e6, rel=0.12)
    assert die_pairs == pytest.approx(100e6, rel=0.12)
    assert across == pytest.approx(np.hypot(100e6, 250e6), rel=0.05)


def test_design_offsets_inflate_raw_within_die_split():
    """Without the design-specific offset, 800 MHz design biases leak into the within-die part."""
    q = _synthetic_qubits()
    raw = split_die_within(q["resid_hz"], q["die"])
    assert raw["sigma_within"] > 2 * 250e6


def test_fit_tc_returns_tc_over_cold_ratio():
    """Step 8b: frequencies made with cold resistances k * R_RT and Tc = 1.16 K, fitted against R_RT, give
    Tc_eff = 1.16 / k. k = 1.1375 and Tc ~ 1.16 K: Krizan, PhD thesis, Chalmers 2026, p. 49."""
    k, tc = 1.1375, 1.16
    rn_rt = np.array([4e3, 5e3, 6e3, 7e3, 8e3])
    cq = np.full(5, 80e-15)
    f = f01_from_rn_exact(k * rn_rt, cq, gap_from_tc(tc))
    assert fit_tc(rn_rt, cq, f) == pytest.approx(tc / k, abs=2e-5)


def test_split_needs_two_dies():
    s = split_die_within(pd.Series([1.0, 2.0]), pd.Series(["a", "a"]))
    assert np.isnan(s["sigma_within"]) and s["n_dies"] == 1


# ---------------------------------------------------------------- real data (downloads)
# Marked `network`: deselected by default (pyproject addopts), run with  python -m pytest -m network
# If Zenodo is unreachable even then, the fixture skips instead of failing.


@pytest.fixture(scope="module")
def imec():
    try:
        return load_imec()
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        pytest.skip(f"imec Zenodo data not available offline: {e}")


@pytest.mark.network
def test_dataset_counts_match_zenodo_readme(imec):
    """README.txt of the Zenodo record: 7872 wafer-map junctions, 393/400 functional qubits.
    (The README says 1752 junctions on three diced dies; the file holds 1752 per die = 5256.)"""
    assert len(imec["wafer"]) == 7872
    assert len(imec["qubits"]) == 393
    assert len(imec["diced"]) == 3 * 1752
    assert imec["wafer"].groupby(["die_x", "die_y"]).ngroups == 82
    assert sorted(imec["wafer"]["cd_nm"].unique()) == [80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 280, 320]
