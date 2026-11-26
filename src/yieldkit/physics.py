"""Physics core: room-temperature junction resistance -> transmon frequency.

All functions take and return SI units (ohm, ampere, joule, farad, kelvin, hertz).
Convert to kOhm / GHz only for display.

Chain implemented here (the "RT -> f01" model used by Fraunhofer EMFT):

    Tc --(BCS)--> gap Delta --(Ambegaokar-Baratoff)--> Ic --> EJ
    Cq --> EC
    (EJ, EC) --(transmon asymptotic formula)--> f01

Composed, this is exactly Eq. 1 of Mayer, Weber et al., arXiv:2505.08424 (2025):

    f01 = sqrt( (1/Rn) * 0.882 * kB * Tc / (h * Cq) ) - e^2 / (2 * h * Cq)
"""

from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.optimize import brentq

E_CHARGE: float = constants.e  # elementary charge [C], CODATA via scipy
H_PLANCK: float = constants.h  # Planck constant [J s]
K_BOLTZMANN: float = constants.k  # Boltzmann constant [J/K]
PHI0: float = constants.physical_constants["mag. flux quantum"][0]  # h/(2e) [Wb]

# BCS weak-coupling ratio Delta = 1.764 kB Tc. Eq. 1 of arXiv:2505.08424 contains
# 0.882 = 1.764 / 2 (Mayer, Weber et al. 2025; Lang et al. arXiv:2504.18173 Eq. 5).
BCS_GAP_RATIO: float = 1.764


def gap_from_tc(tc: float) -> float:
    """Superconducting gap from critical temperature (BCS weak coupling).

    Equation:  Delta = 1.764 * kB * Tc

    Args:
        tc: critical temperature [K]. For Fraunhofer EMFT Al junctions use the
            FITTED effective Tc = 0.71 K (Lang et al., arXiv:2504.18173, Sec. IV.2),
            not the bulk-Al 1.2 K.

    Returns:
        Gap Delta [J].

    Source: the factor 0.882 = 1.764/2 in Eq. 1 of Mayer, Weber et al.,
    arXiv:2505.08424 (2025).
    """
    return BCS_GAP_RATIO * K_BOLTZMANN * tc


def critical_current(rn: float, gap: float) -> float:
    """Junction critical current from normal-state resistance (Ambegaokar-Baratoff, T -> 0).

    Equation:  Ic = pi * Delta / (2 * e * Rn)

    Args:
        rn: normal-state (room-temperature) junction resistance [ohm].
        gap: superconducting gap Delta [J].

    Returns:
        Critical current Ic [A].

    Source: Ambegaokar & Baratoff (1963), as used in Hertzberg et al.,
    npj Quantum Inf. 7, 129 (2021), arXiv:2009.00781, and in Eq. 1 of
    Mayer, Weber et al., arXiv:2505.08424.
    """
    return np.pi * gap / (2.0 * E_CHARGE * rn)


def josephson_energy(ic: float) -> float:
    """Josephson energy from critical current.

    Equation:  EJ = Ic * Phi0 / (2 * pi),  Phi0 = h / (2e)

    Args:
        ic: critical current [A].

    Returns:
        EJ [J].

    Source: standard Josephson relation; it is the EJ implicit in Eq. 1 of
    Mayer, Weber et al., arXiv:2505.08424.
    """
    return ic * PHI0 / (2.0 * np.pi)


def charging_energy(c: float) -> float:
    """Single-electron charging energy of the qubit capacitance.

    Equation:  EC = e^2 / (2 * C)

    Args:
        c: total qubit capacitance [F]. Fraunhofer design value Cq = 86 fF
           (arXiv:2505.08424, Sec. III.1).

    Returns:
        EC [J].

    Source: the constant term e^2/(2 h Cq) in Eq. 1 of arXiv:2505.08424 is EC/h.
    The paper does not name it EC; for Cq = 86 fF, EC/h ~ 225 MHz is our own calculation.
    """
    return E_CHARGE**2 / (2.0 * c)


def transmon_f01(ej: float, ec: float) -> float:
    """Transmon 0->1 transition frequency, asymptotic (EJ >> EC) formula.

    Equation:  f01 = ( sqrt(8 * EJ * EC) - EC ) / h

    Args:
        ej: Josephson energy [J].
        ec: charging energy [J].

    Returns:
        f01 [Hz].

    Source: Koch et al., Phys. Rev. A 76, 042319 (2007), Sec. II (asymptotic
    transmon spectrum); the same form is Eq. 1 of arXiv:2505.08424.
    Accurate to a few MHz when EJ/EC is large; see tests for the scqubits check.
    """
    return (np.sqrt(8.0 * ej * ec) - ec) / H_PLANCK


def f01_from_rn(rn: float, cq: float, gap: float) -> float:
    """Predict transmon f01 from room-temperature junction resistance.

    Equation (Eq. 1 of Mayer, Weber et al., arXiv:2505.08424, with Delta = 1.764 kB Tc):
        f01 = sqrt( (1/Rn) * (Delta/2) / (h * Cq) ) - e^2 / (2 * h * Cq)

    Args:
        rn: normal-state junction resistance [ohm].
        cq: qubit capacitance [F].
        gap: superconducting gap Delta [J] (use gap_from_tc).

    Returns:
        f01 [Hz].
    """
    ej = josephson_energy(critical_current(rn, gap))
    ec = charging_energy(cq)
    return transmon_f01(ej, ec)


def transmon_levels(ej: float, ec: float, ng: float = 0.25, ncut: int = 30) -> np.ndarray:
    """Exact transmon energy levels by charge-basis diagonalisation.

    Equation:  H = 4 EC (n - ng)^2 - (EJ/2) * sum_n ( |n><n+1| + |n+1><n| ),  n = -ncut..ncut

    Args:
        ej: Josephson energy [J].
        ec: charging energy [J].
        ng: offset charge (0.25 sits mid-way in the charge dispersion).
        ncut: charge-basis cutoff.

    Returns:
        Sorted eigen-energies [J].

    Source: Koch et al., Phys. Rev. A 76, 042319 (2007), Sec. II (transmon Hamiltonian).
    """
    n = np.arange(-ncut, ncut + 1)
    scale = H_PLANCK * 1e9  # work in GHz units for numerical conditioning
    h = np.diag(4.0 * ec / scale * (n - ng) ** 2)
    off = -ej / scale / 2.0 * np.ones(len(n) - 1)
    h += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(h) * scale


def transmon_f01_f12_exact(ej: float, ec: float) -> tuple[float, float]:
    """Exact f01 and f12 [Hz] from transmon_levels (Koch et al. 2007)."""
    e = transmon_levels(ej, ec)
    return (e[1] - e[0]) / H_PLANCK, (e[2] - e[1]) / H_PLANCK


def cq_for_anharmonicity(f01: float, alpha: float) -> float:
    """Capacitance that gives a target anharmonicity at a target f01 (exact diagonalisation).

    Solves, for Cq:  alpha_exact(EJ*, EC(Cq)) = alpha,  where EJ* makes f01_exact = f01,
    EC = e^2/(2 Cq) and alpha = f12 - f01 (Koch et al. 2007).

    Used to make our synthetic chip consistent with the anharmonicity ~ -330 MHz assumed by
    the collision table of Hertzberg et al., npj QI 7, 129 (2021), Table 1 caption.

    Args:
        f01: target 0-1 frequency [Hz].
        alpha: target anharmonicity f12 - f01 [Hz] (negative).

    Returns:
        Cq [F].
    """

    def ej_for_f01(ec: float) -> float:
        return brentq(lambda ej: transmon_f01_f12_exact(ej, ec)[0] - f01, 2.0 * ec, 2000.0 * ec, xtol=1e-40)

    def alpha_minus_target(cq: float) -> float:
        ec = charging_energy(cq)
        f01_x, f12_x = transmon_f01_f12_exact(ej_for_f01(ec), ec)
        return (f12_x - f01_x) - alpha

    return brentq(alpha_minus_target, 20e-15, 300e-15, xtol=1e-20)


def rn_target_for_f01(f01: float, cq: float, gap: float) -> float:
    """Junction resistance needed to hit a target f01 (numerical inverse of f01_from_rn).

    Solves  f01_from_rn(Rn, Cq, Delta) - f01 = 0  for Rn with Brent's method.
    f01_from_rn decreases monotonically in Rn, so the root is unique.

    Args:
        f01: target transition frequency [Hz].
        cq: qubit capacitance [F].
        gap: superconducting gap Delta [J].

    Returns:
        Rn [ohm].

    Source: inverse of Eq. 1, Mayer, Weber et al., arXiv:2505.08424
    (paper: 4.42 GHz design -> Rn_target ~ 7.1 kOhm).
    """
    # Search bracket [10 ohm, 10 Mohm] spans every practical junction; it is a
    # numerical choice, not a physical constant.
    return brentq(lambda rn: f01_from_rn(rn, cq, gap) - f01, 1e1, 1e7, xtol=1e-9, rtol=1e-12)
