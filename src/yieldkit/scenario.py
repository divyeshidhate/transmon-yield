"""Step 4 scenario: one heavy-hex d=3 chip per die, on the synthetic wafer of Step 2.

We combine two papers here:
* chip and collision model: IBM-style 3-frequency heavy-hex chip at 5.00 / 5.07 / 5.14 GHz
  (example set-points in Hertzberg et al., npj QI 7, 129 (2021), Discussion; layout: SI Fig. 3),
  collision table assumes anharmonicity -330 MHz (Table 1 caption);
* RT -> f01 model: Eq. 1 of Mayer, Weber et al., arXiv:2505.08424, with Fraunhofer's fitted
  Tc = 0.71 K (Lang et al., PRApplied 24, 054052 (2025), Sec. IV.2).
This is NOT Fraunhofer's 4.42 GHz chip.

Cq is chosen so that the exact anharmonicity is -330 MHz at 5.07 GHz (derived, see
physics.cq_for_anharmonicity); it is not a published capacitance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from yieldkit.collisions import ALPHA_HZ
from yieldkit.lattices import Lattice, build_lattice, design_frequencies
from yieldkit.physics import cq_for_anharmonicity, gap_from_tc, rn_target_for_f01
from yieldkit.wafer import WaferConfig

F1_HZ = 5.00e9  # Hertzberg et al. 2021, Discussion example f1 = 5.0 GHz
STEP_HZ = 70e6  # Hertzberg et al. 2021, Discussion example f2 = 5.07, f3 = 5.14 GHz (fixed, not optimised)
TC_NOMINAL_K = 0.71  # Lang et al. 2025, Sec. IV.2 (fitted effective Tc)
CRYO_NOISE_HZ = 100e3  # Hertzberg et al. 2021: frequencies measured "to better than 100 kHz" (Ramsey)

RN_PROBE_NOISE_RSD = 0.001
"""Hertzberg et al. 2021: room-temperature Rn "measurable to better than 0.1%" (used as the probe noise)."""

RN_RSD_AS_FAB = 0.02
"""~2% RSD, as cited by Krizan et al. 2026 (Mater. Quantum Technol., DOI 10.1088/2633-4356/ae7fe1) from their ref. [9]."""
RN_RSD_TRIMMED = 0.005
"""0.50% wafer-level JJ resistance targeting after parallel ABAA trimming: Wang et al. (Rigetti),
arXiv:2608.27789, abstract / Table II (14 wafers, 150 mm)."""

RESID_IBM_HZ = 14.5e6
"""Hertzberg et al. 2021: 14.5 MHz (0.25%) scatter of f01 about their f(Rn) fit (31 qubits)."""
RESID_FRAUNHOFER_REL = 0.016
"""Mayer, Weber et al. 2025 (arXiv:2505.08424), Sec. III.1: 1.6% average deviation, 40 qubits."""


@dataclass(frozen=True)
class Scenario:
    """Everything fixed about the chip design (no truth here)."""

    lattice: Lattice
    design_hz: np.ndarray  # (23,) design f01 per qubit_id
    cq_f: float  # derived capacitance
    rn_target_ohm: np.ndarray  # (23,) design Rn per qubit_id


def make_scenario(tc_nominal_k: float = TC_NOMINAL_K) -> Scenario:
    """Heavy-hex d=3 chip, design frequencies, derived Cq and per-qubit Rn targets."""
    lat = build_lattice("heavy_hex_d3")
    design = design_frequencies(lat, F1_HZ, STEP_HZ)
    cq = cq_for_anharmonicity(F1_HZ + STEP_HZ, ALPHA_HZ)
    gap = gap_from_tc(tc_nominal_k)
    rn = np.array([rn_target_for_f01(f, cq, gap) for f in design])
    return Scenario(lat, design, cq, rn)


def wafer_config(
    sc: Scenario,
    rn_rsd: float,
    resid_qubit_hz: float,
    resid_die_hz: float = 0.0,
    tc_true_k: tuple[float, ...] = (TC_NOMINAL_K,),
    n_cooled_dies: int = 10,
    probe_noise_rsd: float = RN_PROBE_NOISE_RSD,
) -> WaferConfig:
    """WaferConfig for the Step 4 scenario (Step 2 geometry and spatial model reused).

    Cq spread is folded into the residual (Mayer et al. 2025 footnote 2: the 1.6% includes
    deviations from the design capacity), so cq_rsd = 0. The cryo measurement noise is
    Hertzberg's < 100 kHz.
    """
    return replace(
        WaferConfig(),
        qubits_per_die=sc.lattice.n_qubits,
        rn_design_per_qubit_ohm=tuple(sc.rn_target_ohm),
        rn_rsd=rn_rsd,
        rn_probe_noise_rsd=probe_noise_rsd,
        cq_design_f=sc.cq_f,
        cq_rsd=0.0,
        f01_residual_rsd=CRYO_NOISE_HZ / (F1_HZ + STEP_HZ),
        f01_resid_qubit_hz=resid_qubit_hz,
        f01_resid_die_hz=resid_die_hz,
        tc_eff_true_k=tc_true_k,
        n_wafers=len(tc_true_k),
        n_cooled_dies_per_wafer=n_cooled_dies,
    )
