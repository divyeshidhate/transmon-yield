"""Real data: imec 300 mm wafer, Josephson-junction test arrays and qubits.

Dataset: J. Van Damme, "Advanced CMOS manufacturing of superconducting qubits on 300 mm
wafers", Zenodo, doi:10.5281/zenodo.13143313 (2024), licence CC-BY 4.0. It is the data
repository of Van Damme et al., Nature 634, 74-79 (2024), doi:10.1038/s41586-024-07941-9.

The raw files are NOT stored in this repository. `load_imec` downloads them into a local
cache folder (git-ignored) and checks each file's md5 against the Zenodo record.

Files used (descriptions from the dataset README.txt):
* JJ_data_full_wafer.csv     room-temperature resistance + area estimate of 7872 junctions,
                             probed across the wafer (die x/y coordinates, no in-die position)
* JJ_data_three_diced_dies.csv  more junctions on three diced dies of the same wafer
                             (no position columns)
* qubit_data.csv             analysed qubit metrics (f_qb, anharmonicity, T1, ...) of the
                             393/400 functional devices

The dataset README does not define the identifier tokens (e.g. "SD05_00", "RES-JJ-080_1"),
so we do not decode them.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from yieldkit.physics import E_CHARGE, H_PLANCK, cq_for_anharmonicity, ej_ec_from_f01_alpha, f01_from_rn_exact, gap_from_tc

ZENODO_RECORD = "13143313"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{{name}}?download=1"
DATASET_CITATION = (
    "J. Van Damme, Advanced CMOS manufacturing of superconducting qubits on 300 mm wafers, "
    "Zenodo, doi:10.5281/zenodo.13143313 (2024), CC-BY 4.0; data for Van Damme et al., "
    "Nature 634, 74-79 (2024), doi:10.1038/s41586-024-07941-9"
)
# md5 checksums as listed in the Zenodo record's file metadata (zenodo.org/api/records/13143313).
FILES_MD5 = {
    "JJ_data_full_wafer.csv": "2cd4f12f1c3689a3a7aae538d2ef114c",
    "JJ_data_three_diced_dies.csv": "20ca92aa997202134a56a783994393ab",
    "qubit_data.csv": "f5e8f3a3461028c68f805279d897061e",
}
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "imec"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(cache: Path = DEFAULT_CACHE) -> dict[str, Path]:
    """Download the three CSVs once into `cache` and verify their md5; return their paths."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, md5 in FILES_MD5.items():
        p = cache / name
        if not p.exists() or _md5(p) != md5:
            with urllib.request.urlopen(ZENODO_URL.format(name=name), timeout=60) as r:
                p.write_bytes(r.read())
        got = _md5(p)
        if got != md5:
            raise RuntimeError(f"{name}: md5 {got} does not match the Zenodo record ({md5})")
        paths[name] = p
    return paths


def load_imec(cache: Path = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Load the three tables with short column names, SI units.

    Returns a dict with:
      "wafer":  die_x, die_y, cd_nm (integer label), cd_m, r_ohm, area_m2, ra_ohm_m2   (one row per junction)
      "diced":  die (0, 1, 2 = file order of the "SD05_0x" token; which physical die is which
                is not defined in the README), cd_m, r_ohm, area_m2, ra_ohm_m2
      "qubits": die_x, die_y, subdie, qtype, device, cd_m, f01_hz, alpha_hz (NaN where missing)

    ra_ohm_m2 = R * A (specific resistance) [ohm m^2].
    """
    p = fetch(cache)
    w = pd.read_csv(p["JJ_data_full_wafer.csv"], index_col=0)
    wafer = pd.DataFrame(
        {
            "die_x": w["x-coordinate"].astype(int),
            "die_y": w["y-coordinate"].astype(int),
            "cd_nm": w["critical dimension (nm)"].astype(float).round().astype(int),
            "cd_m": w["critical dimension (nm)"].astype(float).round() * 1e-9,
            "r_ohm": w["resistance (Ohm)"].astype(float),
            "area_m2": w["area (m^2)"].astype(float),
        }
    )
    d = pd.read_csv(p["JJ_data_three_diced_dies.csv"], index_col=0)
    token = d["unique identifier"].str.split("_").str[3]  # "00"/"01"/"02" (undefined in README)
    diced = pd.DataFrame(
        {
            "die": token.astype(int),
            "cd_nm": d["critical dimension (nm)"].astype(float).round().astype(int),
            "cd_m": d["critical dimension (nm)"].astype(float).round() * 1e-9,
            "r_ohm": d["resistance (Ohm)"].astype(float),
            "area_m2": d["area (m^2)"].astype(float),
        }
    )
    for t in (wafer, diced):
        t["ra_ohm_m2"] = t["r_ohm"] * t["area_m2"]
    q = pd.read_csv(p["qubit_data.csv"], index_col=0)
    qubits = pd.DataFrame(
        {
            "die_x": q["x-coordinate"].astype(int),
            "die_y": q["y-coordinate"].astype(int),
            "subdie": q["sub-die name"],
            "qtype": q["qubit type"],
            "device": q["device name"],
            "cd_nm": (q["JJ nominal cd (m)"].astype(float) * 1e9).round().astype(int),
            "cd_m": (q["JJ nominal cd (m)"].astype(float) * 1e9).round() * 1e-9,
            "f01_hz": q["f_qb (Hz)"].astype(float),
            "alpha_hz": pd.to_numeric(q["anharmonicity (Hz)"], errors="coerce"),
        }
    )
    return {"wafer": wafer, "diced": diced, "qubits": qubits}


# ---------------------------------------------------------------------------------------------
# Analysis (statistics; the physics chain is in yieldkit.physics)
# ---------------------------------------------------------------------------------------------

# Gross-failure cut: |ln(RA) - median| > Z_CUT robust sigmas (sigma = 1.4826 * MAD) within a CD group.
# OUR CHOICE (the dataset defines no exclusion rule). The data contain opens (R up to 1e30 ohm) and
# shorts; the notebook shows the result for a second cut (5) as a sensitivity check.
Z_CUT = 10.0


def good_junctions(t: pd.DataFrame, z_cut: float = Z_CUT) -> pd.Series:
    """Boolean mask: junction is not a gross failure (robust z of ln RA within its CD group <= z_cut)."""
    y = np.log(t["ra_ohm_m2"])
    med = y.groupby(t["cd_nm"]).transform("median")
    mad = (y - med).abs().groupby(t["cd_nm"]).transform("median") * 1.4826
    return (y - med).abs() <= z_cut * mad


def poly2d(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Design matrix of all monomials x^i y^j with i + j <= order (order 2 -> 6 columns)."""
    return np.column_stack([x**i * y**j for i in range(order + 1) for j in range(order + 1 - i)])


def decompose_wafer(wafer: pd.DataFrame, diced: pd.DataFrame, order: int = 2, z_cut: float = Z_CUT) -> dict:
    """Split the variance of ln(R*A) into wafer trend, die-to-die residual and within-die scatter.

    Model (one CD group, or all groups pooled after removing each group's mean):
        ln(RA)_{d,k} = T(x_d, y_d) + b_d + e_{d,k}
      T   smooth trend: least-squares 2D polynomial of `order` over the die coordinates,
          fitted to the die means; var_trend = variance of the fitted trend across dies;
      b_d die-to-die residual. Its variance is the residual mean square of the trend fit minus
          the part of the within-die noise left in a die mean (one-way random-effects ANOVA):
              var_b = SS_res / (D - p) - s_w^2 / n_h,  n_h = harmonic mean of copies per die;
      e   within-die scatter. Two estimates:
          (c)  pooled within-die variance on the 3 diced dies (junctions spread over the die);
          (c') pooled variance among the copies of the same CD on each wafer-map die.
    Shares are reported against var_trend + var_b + var_c.

    ln units are relative: sqrt(var) ~ relative spread (RSD) for small spreads.
    Returns a dict of variances, sigmas (relative), shares and counts.
    """
    w = wafer[good_junctions(wafer, z_cut)].copy()
    d = diced[good_junctions(diced, z_cut)].copy()
    w["y"], d["y"] = np.log(w["ra_ohm_m2"]), np.log(d["ra_ohm_m2"])
    die = [w["die_x"], w["die_y"]]
    dm = w.groupby(["die_x", "die_y"])["y"].agg(["mean", "count"]).reset_index()
    s_w2 = float(((w["y"] - w.groupby(die)["y"].transform("mean")) ** 2).sum() / (len(w) - len(dm)))
    s_c2 = float(((d["y"] - d.groupby("die")["y"].transform("mean")) ** 2).sum() / (len(d) - d["die"].nunique()))
    X = poly2d(dm["die_x"].to_numpy(float), dm["die_y"].to_numpy(float), order)
    coef, *_ = np.linalg.lstsq(X, dm["mean"].to_numpy(), rcond=None)
    fit = X @ coef
    res = dm["mean"].to_numpy() - fit
    n_h = len(dm) / float(np.sum(1.0 / dm["count"].to_numpy()))
    var_trend = float(np.var(fit, ddof=1))
    var_b = max(0.0, float(np.sum(res**2)) / (len(dm) - X.shape[1]) - s_w2 / n_h)
    total = var_trend + var_b + s_c2
    return {
        "n_wafer": len(w), "n_wafer_dropped": int(len(wafer) - len(w)),
        "n_diced": len(d), "n_diced_dropped": int(len(diced) - len(d)), "n_dies": len(dm),
        "var_trend": var_trend, "var_die": var_b, "var_within_diced": s_c2, "var_within_copies": s_w2,
        "sigma_trend": np.sqrt(var_trend), "sigma_die": np.sqrt(var_b),
        "sigma_within_diced": np.sqrt(s_c2), "sigma_within_copies": np.sqrt(s_w2),
        "share_trend": var_trend / total, "share_die": var_b / total, "share_within": s_c2 / total,
    }


def decompose_by_cd(wafer: pd.DataFrame, diced: pd.DataFrame, order: int = 2, z_cut: float = Z_CUT) -> pd.DataFrame:
    """decompose_wafer for each CD group, plus a 'pooled' row (each group's mean ln RA removed first)."""
    rows = []
    for cd in sorted(wafer["cd_nm"].unique()):
        r = decompose_wafer(wafer[wafer["cd_nm"] == cd], diced[diced["cd_nm"] == cd], order, z_cut)
        rows.append({"cd_nm": str(cd), **r})
    pw, pdc = wafer[good_junctions(wafer, z_cut)].copy(), diced[good_junctions(diced, z_cut)].copy()
    for t in (pw, pdc):
        ly = np.log(t["ra_ohm_m2"])
        t["ra_ohm_m2"] = np.exp(ly - ly.groupby(t["cd_nm"]).transform("mean"))
        t["cd_nm"] = 0  # one group
    rows.append({"cd_nm": "pooled", **decompose_wafer(pw, pdc, order, z_cut=np.inf)})
    return pd.DataFrame(rows).set_index("cd_nm")


def die_rn_for_qubits(wafer: pd.DataFrame, qubits: pd.DataFrame) -> np.ndarray:
    """Predicted qubit junction resistance from the die's test arrays [ohm] (NaN if the die has no test data).

    Rn_q = RA_die(CD_q) / A(CD_q), where
      RA_die(CD) = exp( linear interpolation over CD of the die-mean ln(RA) of each CD group ),
      A(CD)      = linear interpolation over CD of the dataset's area estimate per CD group.
    ASSUMPTION: a qubit junction of nominal CD behaves like a test junction of the same CD.
    """
    w = wafer[good_junctions(wafer)]
    m = w.assign(y=np.log(w["ra_ohm_m2"])).groupby(["die_x", "die_y", "cd_nm"])["y"].mean()
    area = wafer.groupby("cd_nm")["area_m2"].first()
    dies = set(m.index.droplevel(2))
    out = np.full(len(qubits), np.nan)
    for i, (x, y, cd) in enumerate(qubits[["die_x", "die_y", "cd_nm"]].itertuples(index=False)):
        if (x, y) not in dies:
            continue
        s = m.loc[(x, y)]
        ra = np.exp(np.interp(cd, s.index.to_numpy(), s.to_numpy()))
        out[i] = ra / np.interp(cd, area.index.to_numpy(), area.to_numpy())
    return out


def qubit_cq(qubits: pd.DataFrame) -> np.ndarray:
    """Capacitance of each qubit from its measured (f01, alpha) by exact diagonalisation [F]; NaN if alpha missing.

    Uses physics.cq_for_anharmonicity (transmon Hamiltonian of Koch et al. 2007), searched in
    [0.8, 1.3] x C0 where C0 = e^2 / (2 h |alpha|) is the alpha ~ -EC guess (the exact |alpha| is a
    little larger than EC, so the root lies above C0).
    """
    out = []
    for f, a in zip(qubits["f01_hz"], qubits["alpha_hz"]):
        if not np.isfinite(a):
            out.append(np.nan)
            continue
        c0 = E_CHARGE**2 / (2.0 * H_PLANCK * abs(a))
        out.append(cq_for_anharmonicity(f, a, (0.8 * c0, 1.3 * c0)))
    return np.array(out)


def fit_tc(rn: np.ndarray, cq: np.ndarray, f_meas: np.ndarray, bounds_k=(0.3, 1.5)) -> float:
    """Effective Tc by least squares: min over Tc of sum (f_meas - f01_exact(Rn; Tc, Cq))^2 [K].

    Same one-parameter calibration as Step 4 (Lang et al., arXiv:2504.18173, Sec. IV.2 fit Tc the same way).

    The fitted value is Tc_eff = Tc / k with k = R_cold / R_RT, plus the area estimate and the
    qubit-vs-test-junction geometry: all of these only scale EJ, so one fit absorbs them together
    (see yieldkit.physics.gap_from_tc). It is not a physical Tc.
    """

    def sse(tc: float) -> float:
        return float(np.sum((f01_from_rn_exact(rn, cq, gap_from_tc(tc)) - f_meas) ** 2))

    return float(minimize_scalar(sse, bounds=bounds_k, method="bounded", options={"xatol": 1e-5}).x)


def predict_leave_one_die_out(qubits: pd.DataFrame, rn: np.ndarray, cq_each: np.ndarray) -> pd.DataFrame:
    """Predict every qubit's f01 from its die's test arrays. All cryo-derived constants (Tc_eff, and the
    per-type Cq = median Cq of that qubit type) are fitted on the OTHER dies only.

    Returns the qubits that have test-array data, with columns
    rn_pred_ohm, cq_type_f, tc_fit_k, f01_pred_hz, resid_hz (= measured - predicted), die, design.
    """
    q = qubits.assign(rn_pred_ohm=rn, cq_each_f=cq_each).copy()
    q["die"] = [f"{x},{y}" for x, y in zip(q["die_x"], q["die_y"])]
    q["design"] = [f"{t}|{c}nm" for t, c in zip(q["qtype"], q["cd_nm"])]
    q = q[np.isfinite(q["rn_pred_ohm"])].copy()
    for col in ("cq_type_f", "tc_fit_k", "f01_pred_hz"):
        q[col] = np.nan
    for die in q["die"].unique():
        test, train = q["die"] == die, q["die"] != die
        cq_type = q[train].groupby("qtype")["cq_each_f"].median()
        cq_train = q.loc[train, "qtype"].map(cq_type).to_numpy()
        tc = fit_tc(q.loc[train, "rn_pred_ohm"].to_numpy(), cq_train, q.loc[train, "f01_hz"].to_numpy())
        cq_test = q.loc[test, "qtype"].map(cq_type).to_numpy()
        q.loc[test, "cq_type_f"] = cq_test
        q.loc[test, "tc_fit_k"] = tc
        q.loc[test, "f01_pred_hz"] = f01_from_rn_exact(q.loc[test, "rn_pred_ohm"].to_numpy(), cq_test, gap_from_tc(tc))
    q["resid_hz"] = q["f01_hz"] - q["f01_pred_hz"]
    return q


def split_die_within(resid: pd.Series, die: pd.Series, min_per_die: int = 2) -> dict:
    """Split residuals into a die-shared and a within-die part (one-way random-effects ANOVA, as ranking.calibrate).

        die-shared part on die d:   m_d = mean residual of the qubits on die d
        within-die part of qubit i: r_i - m_d
        sigma_within^2 = sum (r_i - m_d)^2 / sum (n_d - 1)
        sigma_die^2    = max(0, var(m_d) - sigma_within^2 / n_h),  n_h = harmonic mean qubits per die

    Only dies with at least `min_per_die` qubits are used. Units follow the input (Hz).
    """
    n = die.map(die.value_counts())
    r, g = resid[n >= min_per_die], die[n >= min_per_die]
    m = r.groupby(g).transform("mean")
    within = r - m
    sizes = g.value_counts().to_numpy()
    if len(sizes) < 2:  # need >= 2 qualifying dies for both parts
        return {"n_dies": len(sizes), "n_qubits": len(r), "sigma_within": np.nan, "sigma_die": np.nan,
                "within": within, "die_mean": m}
    s_q2 = float((within**2).sum() / np.sum(sizes - 1))
    means = r.groupby(g).mean().to_numpy()
    n_h = len(sizes) / float(np.sum(1.0 / sizes))
    s_d2 = max(0.0, float(np.var(means, ddof=1)) - s_q2 / n_h)
    return {"n_dies": len(sizes), "n_qubits": len(r), "sigma_within": np.sqrt(s_q2), "sigma_die": np.sqrt(s_d2),
            "within": within, "die_mean": m}


def same_design_across_dies(resid: pd.Series, die: pd.Series, design: pd.Series) -> dict:
    """Second, independent estimate of the die-shared part, from the same design on different dies.

    1. Spread of one design across dies (pooled within-design variance):
           s_design^2 = sum_k sum_i (r_ki - mean_k)^2 / sum_k (n_k - 1)   (= sigma_die^2 + sigma_within^2)
    2. Die part from pairs: after removing each design's mean residual (its model bias), two different
       qubits on the same die share only the die effect, so
           sigma_die^2 = mean over same-die pairs (i < j) of c_i * c_j.
       This uses no within-die differences, so it is independent of split_die_within's estimate.
    Only designs that appear on at least 2 dies are used. Units follow the input (Hz).
    """
    k = design.map(design.value_counts())
    r, g, dz = resid[k >= 2], die[k >= 2], design[k >= 2]
    c = r - r.groupby(dz).transform("mean")
    s_design2 = float((c**2).sum() / np.sum(dz.value_counts().to_numpy() - 1))
    num = den = 0.0
    for _, cc in c.groupby(g):
        v = cc.to_numpy()
        if len(v) >= 2:
            num += (np.sum(v) ** 2 - np.sum(v**2)) / 2.0  # sum over pairs i<j of v_i v_j
            den += len(v) * (len(v) - 1) / 2.0
    cov = num / den
    return {"n_designs": int(dz.nunique()), "n_qubits": len(r), "sigma_design_across_dies": np.sqrt(s_design2),
            "cov_same_die": cov, "sigma_die_pairs": np.sqrt(max(cov, 0.0)), "n_pairs": int(den)}


def add_design_offset_lodo(p: pd.DataFrame) -> pd.DataFrame:
    """Make each prediction design-specific: add the design's mean residual, taken from the OTHER dies.

        f_pred_design = f_pred + mean_{same design, other dies}(f_meas - f_pred)
        resid_design_hz = f_meas - f_pred_design

    Why: a nominal CD does not fix the junction geometry of every design (two designs sit ~1.5 GHz
    above the chain prediction), and Cq per type is only a median. A per-design offset absorbs that
    fixed model bias without using the qubit's own die. Qubits whose design appears on no other die
    get NaN.

    It also absorbs a per-design R_cold/R_RT ratio k (Step 8b): k only scales EJ, i.e. acts like a
    per-design Tc_eff = Tc/k. Strictly that is a multiplicative scale, not an additive offset; on the
    imec data (2026-11-27) fitting Tc_eff per design instead (leave-one-die-out) changed the within-die
    and die-shared parts by 4 MHz each (267 vs 271 MHz, 112 vs 116 MHz), so the offset is a good
    stand-in. No separate k parameter: it cannot be separated from Tc with room-temperature data.
    """
    p = p.copy()
    off = np.full(len(p), np.nan)
    for i, (die, des) in enumerate(zip(p["die"], p["design"])):
        other = (p["design"] == des) & (p["die"] != die)
        if other.any():
            off[i] = p.loc[other, "resid_hz"].mean()
    p["design_offset_hz"] = off
    p["resid_design_hz"] = p["resid_hz"] - off
    return p


def qubit_ec_eq411(qubits: pd.DataFrame) -> np.ndarray:
    """EC of each qubit [J] from its measured (f01, alpha) by the closed-form series; NaN if alpha missing.

    Solves Eq. 4.3 (f01) and Eq. 4.11 (alpha) of Krizan, PhD thesis, Chalmers (2026) jointly
    (physics.ej_ec_from_f01_alpha). A cross-check on the exact-diagonalisation EC of qubit_cq.
    """
    return np.array([ej_ec_from_f01_alpha(f, a)[1] if np.isfinite(a) else np.nan
                     for f, a in zip(qubits["f01_hz"], qubits["alpha_hz"])])


def ec_spread(p: pd.DataFrame, by: str = "qtype") -> pd.DataFrame:
    """Relative charging-energy spread per group (`by` = "qtype" or "design"), two ways [dimensionless].

    exact:   EC = e^2 / (2 Cq), Cq from (f01, alpha) by exact diagonalisation (qubit_cq) -> sd(EC) / mean(EC)
    eq411:   EC from (f01, alpha) by Eqs. 4.3 + 4.11 of Krizan, PhD thesis, Chalmers 2026 (qubit_ec_eq411);
             cross-check. Both remove the part of the alpha spread that comes from EJ. (Before Step 8c the
             second column was EC ~ -alpha, Koch et al. 2007, which does not.)
    Expects columns f01_hz, alpha_hz, cq_each_f and the grouping column.
    """
    rows = []
    g_all = p[np.isfinite(p["alpha_hz"])]
    ec411_all = pd.Series(qubit_ec_eq411(g_all), index=g_all.index) / H_PLANCK
    for key, g in g_all.groupby(by):
        ec = E_CHARGE**2 / (2.0 * g["cq_each_f"]) / H_PLANCK
        e4 = ec411_all.loc[g.index]
        rows.append({by: key, "n": len(g), "alpha_mean_mhz": g["alpha_hz"].mean() / 1e6, "ec_mean_mhz": ec.mean() / 1e6,
                     "rsd_exact": ec.std() / ec.mean(), "rsd_eq411": e4.std() / e4.mean(),
                     "ec_eq411_minus_exact_pct": ((e4 / ec - 1) * 100).mean()})
    return pd.DataFrame(rows).set_index(by)


def ec_die_split(p: pd.DataFrame) -> dict:
    """Die-shared vs within-die part of the EC scatter, all designs together.

    x_i = ln EC_i - mean over the same design of ln EC   (removes design-to-design differences)
    then split_die_within(x, die): die-shared and within-die relative spreads, dies with >= 2 qubits.
    Done for EC_exact and for EC from Eqs. 4.3 + 4.11 (Krizan thesis 2026); "approx" (EC ~ -alpha) is kept
    only to show what Step 8c replaced.
    """
    g = p[np.isfinite(p["alpha_hz"])]
    out = {}
    ec411 = pd.Series(qubit_ec_eq411(g), index=g.index)
    for name, ec in (("exact", E_CHARGE**2 / (2.0 * g["cq_each_f"])), ("eq411", ec411), ("approx", -g["alpha_hz"])):
        x = np.log(ec) - np.log(ec).groupby(g["design"]).transform("mean")
        s = split_die_within(x, g["die"])
        out[name] = {"n_dies": s["n_dies"], "n_qubits": s["n_qubits"], "rsd_within_die": s["sigma_within"],
                     "rsd_die": s["sigma_die"], "rsd_total_design_centred": float(x.std())}
    return out
