"""Tests for yieldkit.physics, checked against published numbers.

Published numbers used (see README for links):
- Mayer, Weber et al., arXiv:2505.08424 (2025), Sec. III.1, Eq. 1:
  design f01 = 4.42 GHz -> Rn_target ~ 7.1 kOhm; Tc = 0.71 K; Cq = 86 fF.
- Lang et al., arXiv:2504.18173 (2025), Sec. III.2: measured mean junction Rn = 6.89 kOhm.
- Krizan, PhD thesis, Chalmers (2026), DOI 10.63959/chalmers.dt/5949: Eq. 4.3 (second-order f01);
  Sec. 4.2: first-order form "introduces an error of 6.7 MHz"; Sec. 4.5.1: f01 = 5.4910 GHz
  from EJ = 21.63 GHz, EC = 186.7 MHz. p. 49: RN = 1.1375 * R_RT (Chalmers junctions, not universal).
"""

import numpy as np
import pytest

from yieldkit.physics import (
    E_CHARGE,
    H_PLANCK,
    K_BOLTZMANN,
    charging_energy,
    critical_current,
    cq_for_anharmonicity,
    ej_ec_from_f01_alpha,
    f01_from_rn,
    f01_from_rn_second_order,
    gap_from_tc,
    josephson_energy,
    rn_target_for_f01,
    transmon_f01,
    transmon_alpha_second_order,
    transmon_f01_f12_exact,
    transmon_f01_second_order,
)

TC_FRAUNHOFER = 0.71  # K, fitted effective Tc (arXiv:2504.18173 Sec. IV.2; used in 2505.08424 Eq. 1)
CQ_FRAUNHOFER = 86e-15  # F, design qubit capacitance (arXiv:2505.08424 Sec. III.1)
F01_DESIGN = 4.42e9  # Hz, design frequency (arXiv:2505.08424 Sec. III.1)
RN_TARGET_PAPER = 7.1e3  # ohm, (arXiv:2505.08424 Sec. III.1)
RN_MEAN_LANG = 6.89e3  # ohm, measured wafer mean (arXiv:2504.18173 Sec. III.2)


@pytest.fixture
def gap() -> float:
    return gap_from_tc(TC_FRAUNHOFER)


def eq1_as_printed(rn: float, tc: float, cq: float) -> float:
    """Eq. 1 of arXiv:2505.08424, typed exactly as printed in the paper."""
    return np.sqrt((1.0 / rn) * 0.882 * K_BOLTZMANN * tc / (H_PLANCK * cq)) - E_CHARGE**2 / (2 * H_PLANCK * cq)


def test_chain_equals_paper_eq1(gap):
    """Our step-by-step chain must reproduce the paper's closed-form Eq. 1."""
    for rn in (5.5e3, 7.1e3, 9.5e3):
        assert f01_from_rn(rn, CQ_FRAUNHOFER, gap) == pytest.approx(
            eq1_as_printed(rn, TC_FRAUNHOFER, CQ_FRAUNHOFER), rel=1e-12
        )


def test_design_point_7p1_kohm_gives_4p42_ghz(gap):
    """arXiv:2505.08424: Rn_target ~ 7.1 kOhm for the 4.42 GHz design (within 2%)."""
    assert f01_from_rn(RN_TARGET_PAPER, CQ_FRAUNHOFER, gap) == pytest.approx(F01_DESIGN, rel=0.02)


def test_lang_mean_6p89_kohm_gives_about_4p47_ghz(gap):
    """OUR OWN derived check (not a published frequency): Eq. 1 at Lang's mean Rn = 6.89 kOhm."""
    assert f01_from_rn(RN_MEAN_LANG, CQ_FRAUNHOFER, gap) == pytest.approx(4.47e9, abs=10e6)


def test_rn_target_inverse_matches_paper(gap):
    """Inverse of Eq. 1 at 4.42 GHz should give the paper's ~7.1 kOhm (within 3%)."""
    assert rn_target_for_f01(F01_DESIGN, CQ_FRAUNHOFER, gap) == pytest.approx(RN_TARGET_PAPER, rel=0.03)


def test_rn_target_round_trip(gap):
    for f in (4.0e9, 4.42e9, 5.0e9):
        rn = rn_target_for_f01(f, CQ_FRAUNHOFER, gap)
        assert f01_from_rn(rn, CQ_FRAUNHOFER, gap) == pytest.approx(f, rel=1e-9)


def test_rn_target_matches_closed_form(gap):
    """Eq. 1 can be inverted by hand: Rn = (Delta/2) / (h Cq (f + EC/h)^2)."""
    ec_hz = charging_energy(CQ_FRAUNHOFER) / H_PLANCK
    closed = (gap / 2) / (H_PLANCK * CQ_FRAUNHOFER * (F01_DESIGN + ec_hz) ** 2)
    assert rn_target_for_f01(F01_DESIGN, CQ_FRAUNHOFER, gap) == pytest.approx(closed, rel=1e-9)


def test_charging_energy_about_225_mhz():
    """OUR OWN calculation (paper does not state EC): e^2/(2 h * 86 fF) ~ 225 MHz."""
    assert charging_energy(CQ_FRAUNHOFER) / H_PLANCK == pytest.approx(225e6, abs=1e6)


def test_critical_current_and_ej_units(gap):
    """Sanity: EJ = Ic*Phi0/2pi equals Delta*h/(8 e^2 Rn) (algebra of Ambegaokar-Baratoff)."""
    rn = 7.1e3
    ej = josephson_energy(critical_current(rn, gap))
    assert ej == pytest.approx(gap * H_PLANCK / (8 * E_CHARGE**2 * rn), rel=1e-12)


def _scqubits_f01(ej_over_ec: float, ec_ghz: float) -> float:
    scq = pytest.importorskip("scqubits")
    t = scq.Transmon(EJ=ej_over_ec * ec_ghz, EC=ec_ghz, ng=0.25, ncut=40, truncated_dim=4)
    ev = t.eigenvals(evals_count=2)
    return (ev[1] - ev[0]) * 1e9  # Hz


def test_asymptotic_formula_vs_scqubits():
    """Compare sqrt(8 EJ EC) - EC with exact diagonalisation (scqubits, ng = 0.25).

    Finding: the asymptotic formula OVERestimates f01 by ~13 MHz (0.3%) at the
    Fraunhofer operating point (EJ/EC ~ 53) and ~23 MHz (0.9%) at EJ/EC = 20.
    The error shrinks as EJ/EC grows. Our tolerance (1% of f01) is our own choice:
    it keeps the formula error below the paper's 1.6% prediction accuracy
    (arXiv:2505.08424 Sec. III.1).
    """
    ec = charging_energy(CQ_FRAUNHOFER)
    ec_ghz = ec / H_PLANCK / 1e9
    errors = []
    for ratio in (20, 30, 50, 100):
        exact = _scqubits_f01(ratio, ec_ghz)
        asym = transmon_f01(ratio * ec, ec)
        assert asym > exact
        assert asym == pytest.approx(exact, rel=0.01)
        errors.append(asym - exact)
    assert all(np.diff(errors) < 0)  # error decreases as EJ/EC grows


# ---------------------------------------------------------------- second-order f01 (Step 8a)

EJ_KRIZAN = 21.63e9 * H_PLANCK  # J, thesis Sec. 4.5.1
EC_KRIZAN = 186.7e6 * H_PLANCK  # J, thesis Sec. 4.5.1
F01_KRIZAN = 5.4910e9  # Hz, measured mean of Q5-Q8, thesis Sec. 4.5.1


def test_second_order_reproduces_thesis_worked_example():
    """EJ = 21.63 GHz, EC = 186.7 MHz -> f01 = 5.4910 GHz (thesis Sec. 4.5.1).
    EJ and EC are printed to 4 digits, so we allow 1 MHz (our choice)."""
    assert transmon_f01_second_order(EJ_KRIZAN, EC_KRIZAN) == pytest.approx(F01_KRIZAN, abs=1e6)


def test_first_order_error_is_thesis_6p7_mhz():
    """Thesis Sec. 4.2: sqrt(8 EJ EC) - EC instead of Eq. 4.3 "introduces an error of 6.7 MHz"."""
    gap_mhz = (transmon_f01(EJ_KRIZAN, EC_KRIZAN) - transmon_f01_second_order(EJ_KRIZAN, EC_KRIZAN)) / 1e6
    assert gap_mhz == pytest.approx(6.7, abs=0.05)


def test_second_order_vs_scqubits():
    """Second-order formula against exact diagonalisation (scqubits, ng = 0.25), EC of the Fraunhofer chip.

    Finding: the first-order error (13 MHz at EJ/EC ~ 53) shrinks to < 1 MHz for EJ/EC >= 35, and the
    second-order formula stays above exact (the next term is negative). 1 MHz is our tolerance.
    """
    ec = charging_energy(CQ_FRAUNHOFER)
    ec_ghz = ec / H_PLANCK / 1e9
    for ratio in (35, 53, 66, 100, 190):
        exact = _scqubits_f01(ratio, ec_ghz)
        second = transmon_f01_second_order(ratio * ec, ec)
        first = transmon_f01(ratio * ec, ec)
        assert 0 < second - exact < 1e6
        assert (second - exact) < (first - exact) / 10


def test_second_order_at_fraunhofer_point():
    """7.1 kOhm, 86 fF, Tc 0.71 K: Eq. 1 is ~13 MHz high; second order is ~0.3 MHz high (vs scqubits)."""
    gap = gap_from_tc(TC_FRAUNHOFER)
    ej = josephson_energy(critical_current(RN_TARGET_PAPER, gap))
    ec = charging_energy(CQ_FRAUNHOFER)
    exact = _scqubits_f01(ej / ec, ec / H_PLANCK / 1e9)
    assert 10e6 < f01_from_rn(RN_TARGET_PAPER, CQ_FRAUNHOFER, gap) - exact < 16e6
    assert 0 < f01_from_rn_second_order(RN_TARGET_PAPER, CQ_FRAUNHOFER, gap) - exact < 1e6


def test_rn_target_inverts_second_order_model():
    gap = gap_from_tc(TC_FRAUNHOFER)
    rn = rn_target_for_f01(F01_DESIGN, CQ_FRAUNHOFER, gap, model=f01_from_rn_second_order)
    assert f01_from_rn_second_order(rn, CQ_FRAUNHOFER, gap) == pytest.approx(F01_DESIGN, rel=1e-9)
    # Lower f01 per ohm -> slightly lower Rn needed than with Eq. 1 (the paper's 7.1 kOhm target).
    assert rn < rn_target_for_f01(F01_DESIGN, CQ_FRAUNHOFER, gap)


# ---------------------------------------------------------------- Tc vs R_cold/R_RT degeneracy (Step 8b)

K_CHALMERS = 1.1375  # R_cold / R_RT, Krizan thesis p. 49 ("generally not to be considered universal")


@pytest.mark.parametrize("model", [f01_from_rn, f01_from_rn_second_order])
def test_cold_ratio_and_tc_enter_only_as_tc_over_ratio(model):
    """EJ = (hbar/2e) pi 1.764 kB Tc / (2 e k R_RT): a cold/warm ratio k is indistinguishable from Tc -> Tc/k.
    Checked for the two closed-form models at the Fraunhofer point, and for EJ itself."""
    for tc in (0.71, 1.16):
        cold = model(K_CHALMERS * RN_TARGET_PAPER, CQ_FRAUNHOFER, gap_from_tc(tc))
        warm_eff = model(RN_TARGET_PAPER, CQ_FRAUNHOFER, gap_from_tc(tc / K_CHALMERS))
        assert cold == pytest.approx(warm_eff, rel=1e-12)
    ej_cold = josephson_energy(critical_current(K_CHALMERS * RN_TARGET_PAPER, gap_from_tc(1.16)))
    ej_eff = josephson_energy(critical_current(RN_TARGET_PAPER, gap_from_tc(1.16 / K_CHALMERS)))
    assert ej_cold == pytest.approx(ej_eff, rel=1e-12)


def test_cold_ratio_degeneracy_exact_spectrum():
    from yieldkit.physics import f01_from_rn_exact

    rn = np.array([6e3, 7.1e3, 8e3])
    cold = f01_from_rn_exact(K_CHALMERS * rn, CQ_FRAUNHOFER, gap_from_tc(1.16))
    warm_eff = f01_from_rn_exact(rn, CQ_FRAUNHOFER, gap_from_tc(1.16 / K_CHALMERS))
    assert np.allclose(cold, warm_eff, rtol=1e-12)


# ---------------------------------------------------------------- eq. 4.11: EC from anharmonicity (Step 8c)


def test_ej_ec_from_f01_alpha_reproduces_thesis():
    """Thesis Sec. 4.5.1: f01 = 5.4910 GHz, eta/2pi = -203.0 MHz -> EJ = 21.63 GHz, EC = 186.7 MHz (Eqs. 4.3 + 4.11)."""
    ej, ec = ej_ec_from_f01_alpha(5.4910e9, -203.0e6)
    assert ej / H_PLANCK == pytest.approx(21.63e9, abs=0.005e9)
    assert ec / H_PLANCK == pytest.approx(186.7e6, abs=0.05e6)
    assert transmon_f01_second_order(ej, ec) == pytest.approx(5.4910e9, rel=1e-12)
    assert transmon_alpha_second_order(ej, ec) == pytest.approx(-203.0e6, rel=1e-12)


@pytest.mark.parametrize("ratio", [35, 53, 66, 100, 190])
def test_eq411_alpha_vs_exact(ratio):
    """Eq. 4.11 vs exact diagonalisation at the Fraunhofer EC: within 1 MHz for EJ/EC >= 35 (our tolerance),
    while the leading-order alpha = -EC is 15-45 MHz too small in magnitude."""
    ec = charging_energy(CQ_FRAUNHOFER)
    f01, f12 = transmon_f01_f12_exact(ratio * ec, ec)
    exact = f12 - f01
    assert transmon_alpha_second_order(ratio * ec, ec) == pytest.approx(exact, abs=1e6)
    assert abs(-ec / H_PLANCK - exact) > 10e6


@pytest.mark.parametrize("f01, alpha", [(5.07e9, -330e6), (4.5e9, -200e6), (6.0e9, -250e6)])
def test_ec_from_eq411_matches_exact_diagonalisation(f01, alpha):
    """EC from (f01, alpha) by Eqs. 4.3 + 4.11 agrees with the exact-diagonalisation EC (cq_for_anharmonicity)
    to 0.3% (our tolerance); the old EC = |alpha| is off by several percent."""
    c0 = E_CHARGE**2 / (2 * H_PLANCK * abs(alpha))
    ec_exact = charging_energy(cq_for_anharmonicity(f01, alpha, (0.8 * c0, 1.3 * c0)))
    _, ec_series = ej_ec_from_f01_alpha(f01, alpha)
    assert ec_series == pytest.approx(ec_exact, rel=0.003)
    assert abs(H_PLANCK * abs(alpha) / ec_exact - 1) > 0.03
