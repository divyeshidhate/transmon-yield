"""Tests for yieldkit.physics, checked against published numbers.

Published numbers used (see README for links):
- Mayer, Weber et al., arXiv:2505.08424 (2025), Sec. III.1, Eq. 1:
  design f01 = 4.42 GHz -> Rn_target ~ 7.1 kOhm; Tc = 0.71 K; Cq = 86 fF.
- Lang et al., arXiv:2504.18173 (2025), Sec. III.2: measured mean junction Rn = 6.89 kOhm.
"""

import numpy as np
import pytest

from yieldkit.physics import (
    E_CHARGE,
    H_PLANCK,
    K_BOLTZMANN,
    charging_energy,
    critical_current,
    f01_from_rn,
    gap_from_tc,
    josephson_energy,
    rn_target_for_f01,
    transmon_f01,
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
