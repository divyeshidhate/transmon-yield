"""Synthetic 200 mm transmon wafer: hidden truth + what an analyst can observe.

Every parameter in WaferConfig is marked either with the paper it comes from or
as an ASSUMPTION (not published; chosen by us). All values are SI.

make_wafer() returns two separate tables that share the keys
(wafer_id, die_id, qubit_id):

* observed -- what a fab analyst sees: die position, room-temperature resistance
  (with probe noise), and cryo-measured f01 only for dies that were cooled.
* truth    -- hidden ground truth (true Rn and its components, true Cq, the
  wafer's effective Tc, true f01). Use it only to score methods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from yieldkit.physics import f01_from_rn, gap_from_tc

KEYS = ["wafer_id", "die_id", "qubit_id"]

# Rn spread presets (relative standard deviation of RT junction resistance).
RSD_LANG_2025 = 0.124
"""12.4% across a 200 mm wafer: Lang et al., Phys. Rev. Applied 24, 054052 (2025), arXiv:2504.18173, Sec. III.2."""
RSD_MODERN = 0.02
"""~2% RSD, as cited by Krizan et al. 2026 (Mater. Quantum Technol., DOI 10.1088/2633-4356/ae7fe1)
from their ref. [9]; literature claim (50 mm wafer), not their own measurement."""

# Tuned, NOT derived: makes mean |Eq.1 prediction - measured| / measured ~ 1.6%
# (Mayer, Weber et al., arXiv:2505.08424, Sec. III.1, 40 qubits) for the default
# config. Re-derive with tune_f01_residual(WaferConfig()).
F01_RESIDUAL_RSD_TUNED = 0.0174


@dataclass(frozen=True)
class WaferConfig:
    """All knobs of the synthetic wafer (SI units)."""

    # --- geometry ---
    wafer_diameter_m: float = 0.200  # 200 mm wafer (Lang et al. 2025; Mayer et al. 2025)
    edge_exclusion_m: float = 4e-3  # ASSUMPTION chosen to match Lang's ~375 dies (10 x 7 mm)
    die_width_m: float = 10e-3  # Lang et al. 2025, Sec. II: dies of 10 mm x 7 mm
    die_height_m: float = 7e-3  # Lang et al. 2025, Sec. II
    qubits_per_die: int = 4  # Mayer, Weber et al. 2025 (arXiv:2505.08424), Sec. II: four qubits per chip

    # --- room-temperature junction resistance (lognormal) ---
    rn_mean_ohm: float = 6.89e3  # measured wafer mean, Lang et al. 2025, Sec. III.2
    rn_rsd: float = RSD_LANG_2025
    frac_trend: float = 0.50  # ASSUMPTION: share of ln(Rn) variance from smooth wafer trend
    frac_die: float = 0.25  # ASSUMPTION: share from per-die offset
    frac_junction: float = 0.25  # ASSUMPTION: share from per-junction noise
    frac_radial_in_trend: float = 0.70  # ASSUMPTION: radial bowl vs linear tilt inside the trend
    rn_probe_noise_rsd: float = 0.005  # ASSUMPTION: RT probe measurement noise (~0.5%)

    # --- cryo physics ---
    n_wafers: int = 1
    tc_eff_true_k: tuple[float, ...] = (0.71,)  # per wafer; 0.71 K fitted by Lang et al. 2025, Sec. IV.2
    cq_design_f: float = 86e-15  # design capacitance, arXiv:2505.08424, Sec. III.1
    cq_rsd: float = 0.02  # ASSUMPTION: capacitance spread (paper: Cq deviations contribute, no number given)
    f01_residual_rsd: float = F01_RESIDUAL_RSD_TUNED  # TUNED to 1.6% (see constant above)
    ageing_shift_rel: float = 0.0  # ASSUMPTION: optional systematic shift of cryo f01 (none published)

    # --- Step 4 scenario options (defaults reproduce Step 2 exactly) ---
    # Per-qubit design resistance (index = qubit_id). If set, it is the MEDIAN of each
    # qubit's lognormal Rn (targeted process) and rn_mean_ohm is ignored.
    rn_design_per_qubit_ohm: tuple[float, ...] | None = None
    # Physical f01 deviation that RT data cannot explain, added to the TRUE f01 [Hz].
    # Split into a per-qubit part (drives collisions) and a die-shared part (absolute error only).
    f01_resid_qubit_hz: float = 0.0
    f01_resid_die_hz: float = 0.0

    # --- what gets cooled ---
    n_cooled_dies_per_wafer: int = 10  # 10 chips x 4 = 40 qubits, as in Lang et al. / Mayer et al.

    def tc_for_wafer(self, wafer_id: int) -> float:
        """Effective Tc [K] of one wafer (a single value is broadcast to all wafers)."""
        tcs = self.tc_eff_true_k
        return tcs[0] if len(tcs) == 1 else tcs[wafer_id]


def die_layout(cfg: WaferConfig) -> pd.DataFrame:
    """Centres of all whole dies inside the usable wafer area.

    A rectangular grid centred on the wafer; a die is kept only if all four of
    its corners lie within radius = diameter/2 - edge_exclusion.

    Returns:
        DataFrame with columns die_id, x_m, y_m (die centre, metres).
    """
    r_usable = cfg.wafer_diameter_m / 2 - cfg.edge_exclusion_m
    w, h = cfg.die_width_m, cfg.die_height_m
    nx = int(np.ceil(cfg.wafer_diameter_m / w)) + 1
    ny = int(np.ceil(cfg.wafer_diameter_m / h)) + 1
    xc = (np.arange(nx) - (nx - 1) / 2) * w
    yc = (np.arange(ny) - (ny - 1) / 2) * h
    x, y = np.meshgrid(xc, yc)
    x, y = x.ravel(), y.ravel()
    corner_r = np.max(
        [np.hypot(x + sx * w / 2, y + sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)], axis=0
    )
    keep = corner_r <= r_usable
    return pd.DataFrame({"die_id": np.arange(keep.sum()), "x_m": x[keep], "y_m": y[keep]})


def _unit_std(v: np.ndarray) -> np.ndarray:
    """Centre and scale to unit standard deviation."""
    v = v - v.mean()
    return v / v.std()


def lognormal_sigma(rsd: float) -> float:
    """Std of ln(X) for a lognormal X with relative standard deviation rsd.

    Equation: sigma_ln = sqrt( ln(1 + RSD^2) )   (standard lognormal identity)
    """
    return float(np.sqrt(np.log1p(rsd**2)))


def make_wafer(cfg: WaferConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic wafers.

    Model, per qubit q on die d of wafer w (all lognormal in Rn):
        ln Rn_true = ln Rn_median + trend_w(x_d, y_d) + offset_d + noise_q
        Rn_median  = Rn_mean / sqrt(1 + RSD^2)                  (lognormal mean -> median)
        trend      = radial bowl (r^2) + linear tilt (random direction), unit-normalised
        Rn_measured = Rn_true * (1 + probe noise)
        Cq_true    = Cq_design * (1 + cq noise)
        f01_true   = Eq. 1 (arXiv:2505.08424) with the wafer's true Tc and Cq_true
        f01_measured = f01_true * (1 + ageing_shift) * (1 + residual noise), cooled dies only

    Args:
        cfg: wafer configuration.
        seed: random seed; the same seed gives identical tables.

    Returns:
        (observed, truth) DataFrames sharing keys wafer_id, die_id, qubit_id.
    """
    rng = np.random.default_rng(seed)
    dies = die_layout(cfg)
    n_die, n_q = len(dies), cfg.qubits_per_die
    r_wafer = cfg.wafer_diameter_m / 2
    sigma = lognormal_sigma(cfg.rn_rsd)
    ln_median = np.log(cfg.rn_mean_ohm / np.sqrt(1 + cfg.rn_rsd**2))

    obs_parts, truth_parts = [], []
    for w in range(cfg.n_wafers):
        x, y = dies["x_m"].to_numpy(), dies["y_m"].to_numpy()
        # Smooth trend: radial bowl + linear tilt, then rescaled to its variance share.
        radial = _unit_std((x**2 + y**2) / r_wafer**2) * rng.choice([-1.0, 1.0])
        theta = rng.uniform(0, 2 * np.pi)
        linear = _unit_std(x * np.cos(theta) + y * np.sin(theta))
        a = cfg.frac_radial_in_trend
        trend_die = _unit_std(np.sqrt(a) * radial + np.sqrt(1 - a) * linear) * sigma * np.sqrt(cfg.frac_trend)
        offset_die = rng.normal(0.0, sigma * np.sqrt(cfg.frac_die), n_die)

        # Broadcast die-level terms to qubits.
        die_id = np.repeat(dies["die_id"].to_numpy(), n_q)
        qubit_id = np.tile(np.arange(n_q), n_die)
        trend = np.repeat(trend_die, n_q)
        offset = np.repeat(offset_die, n_q)
        noise = rng.normal(0.0, sigma * np.sqrt(cfg.frac_junction), n_die * n_q)

        if cfg.rn_design_per_qubit_ohm is not None:
            if len(cfg.rn_design_per_qubit_ohm) != n_q:
                raise ValueError("rn_design_per_qubit_ohm needs one value per qubit")
            ln_med = np.log(np.asarray(cfg.rn_design_per_qubit_ohm))[qubit_id]
        else:
            ln_med = ln_median
        rn_true = np.exp(ln_med + trend + offset + noise)
        rn_measured = rn_true * (1 + rng.normal(0.0, cfg.rn_probe_noise_rsd, rn_true.size))
        cq_true = cfg.cq_design_f * (1 + rng.normal(0.0, cfg.cq_rsd, rn_true.size))
        tc = cfg.tc_for_wafer(w)
        f01_true = f01_from_rn(rn_true, cq_true, gap_from_tc(tc))
        resid_die = np.zeros(rn_true.size)
        resid_qubit = np.zeros(rn_true.size)
        if cfg.f01_resid_die_hz > 0:
            resid_die = np.repeat(rng.normal(0.0, cfg.f01_resid_die_hz, n_die), n_q)
        if cfg.f01_resid_qubit_hz > 0:
            resid_qubit = rng.normal(0.0, cfg.f01_resid_qubit_hz, rn_true.size)
        f01_true = f01_true + resid_die + resid_qubit

        f01_meas_all = f01_true * (1 + cfg.ageing_shift_rel) * (
            1 + rng.normal(0.0, cfg.f01_residual_rsd, rn_true.size)
        )
        cooled_dies = rng.choice(n_die, size=min(cfg.n_cooled_dies_per_wafer, n_die), replace=False)
        is_cooled = np.isin(die_id, cooled_dies)
        f01_measured = np.where(is_cooled, f01_meas_all, np.nan)

        obs_parts.append(
            pd.DataFrame(
                {
                    "wafer_id": w,
                    "die_id": die_id,
                    "qubit_id": qubit_id,
                    "x_m": np.repeat(x, n_q),
                    "y_m": np.repeat(y, n_q),
                    "rn_measured_ohm": rn_measured,
                    "f01_measured_hz": f01_measured,
                    "is_cooled": is_cooled,
                }
            )
        )
        truth_parts.append(
            pd.DataFrame(
                {
                    "wafer_id": w,
                    "die_id": die_id,
                    "qubit_id": qubit_id,
                    "rn_true_ohm": rn_true,
                    "ln_rn_trend": trend,
                    "ln_rn_die_offset": offset,
                    "ln_rn_junction_noise": noise,
                    "cq_true_f": cq_true,
                    "tc_eff_true_k": tc,
                    "f01_resid_die_hz": resid_die,
                    "f01_resid_qubit_hz": resid_qubit,
                    "f01_true_hz": f01_true,
                }
            )
        )
    return pd.concat(obs_parts, ignore_index=True), pd.concat(truth_parts, ignore_index=True)


def eq1_prediction_hz(observed: pd.DataFrame, tc_k: float, cq_f: float) -> np.ndarray:
    """Analyst's prediction: Eq. 1 of arXiv:2505.08424 applied to measured Rn.

    Equation: f01_pred = f01_from_rn(Rn_measured, Cq_design, 1.764 kB Tc)
    """
    return f01_from_rn(observed["rn_measured_ohm"].to_numpy(), cq_f, gap_from_tc(tc_k))


def mean_abs_rel_error(observed: pd.DataFrame, tc_k: float, cq_f: float) -> float:
    """Mean |f_pred - f_meas| / f_meas over cooled qubits (our reading of the paper's "1.6% on average")."""
    cooled = observed[observed["is_cooled"]]
    pred = eq1_prediction_hz(cooled, tc_k, cq_f)
    meas = cooled["f01_measured_hz"].to_numpy()
    return float(np.mean(np.abs(pred - meas) / meas))


def tune_f01_residual(cfg: WaferConfig, target: float = 0.016, seed: int = 0, n_wafers: int = 20) -> float:
    """Find f01_residual_rsd so that the mean |prediction error| equals target.

    This is TUNING to match the 1.6% of arXiv:2505.08424 (Sec. III.1), not a derivation.
    Uses all dies cooled on n_wafers wafers (same seed for every trial) so the
    objective is smooth; the predictor uses Tc = first wafer's true Tc and Cq = design.
    """
    base = replace(cfg, n_wafers=n_wafers, tc_eff_true_k=(cfg.tc_for_wafer(0),), n_cooled_dies_per_wafer=10**6)

    def gap_to_target(resid: float) -> float:
        obs, _ = make_wafer(replace(base, f01_residual_rsd=resid), seed)
        return mean_abs_rel_error(obs, cfg.tc_for_wafer(0), cfg.cq_design_f) - target

    return float(brentq(gap_to_target, 0.0, 0.1, xtol=1e-6))
