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
EC_RSD_OSMAN = 0.008
"""Charging-energy scatter sigma_EC/EC = 0.8% (1.7 MHz on 195 MHz): Osman et al., Phys. Rev.
Research 5, 043001 (2023), Sec. III / Fig. 2(b), 32 qubits excluding 3 outliers. Measured across
4 dies (5 x 7 mm each), so it includes some die-to-die spread: treating all of it as per-qubit
(ec_die_frac = 0) is an upper bound for the collision-relevant part. The paper gives no split."""

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
    ec_rsd: float = 0.0,
    ec_die_frac: float = 0.0,
    drift_shared_rel: float = 0.0,
    drift_junction_rsd: float = 0.0,
) -> WaferConfig:
    """WaferConfig for the Step 4 scenario (Step 2 geometry and spatial model reused).

    resid_qubit_hz / resid_die_hz: frequency deviation RT data cannot explain, EXCLUDING the
    capacitor. ec_rsd: relative EC (= Cq) scatter, hidden from the RT observation, added on
    top (default 0 reproduces Step 4; Osman et al. 2023 measured 0.8%, see EC_RSD_OSMAN).
    It shifts f01 by (f - EC)/(2 f) * dEC/EC (physics.f01_ec_sensitivity), about 19 MHz per
    qubit at 0.8% and 5.07 GHz. Combining a measured lab residual (e.g. IBM's 14.5 MHz, which
    already contains that lab's EC scatter) with Osman's EC scatter is therefore a pessimistic
    scenario, not a correction. ec_die_frac: share of the EC variance common to the die.
    drift_shared_rel / drift_junction_rsd: Rn ageing between probe and cooldown (see
    WaferConfig.rn_drift_shared_rel / rn_drift_junction_rsd; the per-junction value is ASSUMED).
    The cryo measurement noise is Hertzberg's < 100 kHz.
    """
    return replace(
        WaferConfig(),
        qubits_per_die=sc.lattice.n_qubits,
        rn_design_per_qubit_ohm=tuple(sc.rn_target_ohm),
        rn_rsd=rn_rsd,
        rn_probe_noise_rsd=probe_noise_rsd,
        cq_design_f=sc.cq_f,
        cq_rsd=ec_rsd,
        cq_die_frac=ec_die_frac,
        rn_drift_shared_rel=drift_shared_rel,
        rn_drift_junction_rsd=drift_junction_rsd,
        f01_residual_rsd=CRYO_NOISE_HZ / (F1_HZ + STEP_HZ),
        f01_resid_qubit_hz=resid_qubit_hz,
        f01_resid_die_hz=resid_die_hz,
        tc_eff_true_k=tc_true_k,
        n_wafers=len(tc_true_k),
        n_cooled_dies_per_wafer=n_cooled_dies,
    )
