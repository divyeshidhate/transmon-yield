# transmon-yield

[![tests](https://github.com/divyeshidhate/transmon-yield/actions/workflows/tests.yml/badge.svg)](https://github.com/divyeshidhate/transmon-yield/actions/workflows/tests.yml)

**Ranking superconducting-qubit dies before cooldown from room-temperature junction data: a reproduction of published results and a simulation study.**

Every junction on a wafer can be probed at room temperature on an automated wafer prober; cooling a chip to millikelvin
temperatures and characterising it takes far longer. This repo asks:
*using only room-temperature resistance and a few cooled calibration dies, which dies should you cool first to find
collision-free chips, and how accurate must the prediction be for that to help?*

![Capture curve](docs/capture_curve.png)

*Cool dies in ranked order: what fraction of the truly collision-free dies have you found? One synthetic wafer
(360 uncooled dies, 14 collision-free) in the reference scenario: ~2% Rn spread, 14.5 MHz per-qubit prediction residual,
no capacitor scatter. With capacitor scatter added the curves move much closer to random (see below). Notebook 04.*

## Key results (simulation, synthetic data)

All results: heavy-hex d = 3 chip (23 qubits, 5.00 / 5.07 / 5.14 GHz), one chip per 10 × 7 mm die, 370 dies per 200 mm
wafer, 10 calibration dies, averaged over 10 synthetic wafers. "Residual" = the within-die, per-qubit frequency error that
room-temperature data cannot explain. None of this is validated on real fab data.

**Reference scenario** (14.5 MHz residual, IBM's measured value; no separate capacitor scatter, no ageing):

- **~25 vs ~129 cooldowns to find 5 collision-free chips** (ours vs random; naive ordering by |Rn − target|: ~68),
  for an untrimmed wafer with ~2% Rn spread. Only ~4% of dies are collision-free here.
- **After trimming to 0.5% Rn spread, ranking barely helps**: ~49% of dies are collision-free at a 14.5 MHz residual,
  but the ranking gain is only 0.08 (0 = random, 1 = perfect). It needs a residual of ≲ 5–10 MHz (gain 0.43 at 5 MHz,
  0.16 at 10 MHz). Trimming removes the within-die variation that ranking relies on.

**Realism sweep** (adds what a resistance probe cannot see: Osman et al.'s 0.8% capacitor (E_C) scatter, ~19 MHz per qubit;
4% shared Rn ageing between probe and cooldown; 0–1% per-junction ageing, an assumed value). Adding Osman's scatter on top of
IBM's measured residual is a pessimistic scenario, not a correction to IBM:

- **Untrimmed, the advantage shrinks from ~5× to ~2–3×, and collision-free dies become rare (~1–2%).**
  If half of the capacitor scatter is shared by the die: ~114 vs ~215 cooldowns to 5 good chips (all 10 wafers had at
  least 5), ~40 vs ~129 to 3. If all of it is per-qubit: ~96 vs ~178 to 3 good chips (1 of 10 wafers had fewer
  than 3); 4 of 10 wafers had fewer than 5.
  Osman et al. do not report how their 0.8% splits between dies, so these bracket the answer.
- **Trimmed, ranking gives nothing** (gain −0.09 to 0.09 in every realism run).
- **Ageing shared by the whole wafer does not hurt the ranking** once a few cooled dies recalibrate T_c (gain 0.44 without
  vs 0.48 with 4% drift). Per-junction ageing (assumed values, none published) acts like extra residual: at 1% of Rn it
  adds ~27 MHz per qubit.

**Physics behind both:**

- **Only within-die, per-qubit error matters for picking dies.** A frequency offset shared by the whole die cannot cause a
  collision; a shared change in Rn or T_c barely can (it squeezes all gaps by about half the fractional change, against a
  fixed anharmonicity). A shift in effective T_c from 0.71 to 0.68 K biases absolute predictions by +116 MHz but leaves the
  ranking unchanged (gain 0.61 vs 0.60). Recalibrating on 3 cooled dies removes the bias (−0.5 ± 1.7 MHz).
- **A 1.6% residual (~81 MHz), if entirely per-qubit, leaves no collision-free heavy-hex chips at all** (both Rn
  presets). What matters is how much of the error is shared by the whole die. With 10 cooled dies the calibration
  recovers that split (10.6 / 11.0 MHz estimated vs 10.3 / 10.3 MHz true).
- **Calibration is cheap**: effective T_c is fitted to ±0.0007 K from 2 cooled dies, and the ranking gain is flat
  (0.61–0.64) from 2 to 20 calibration dies.

## Real-data check: test arrays vs qubits (notebook 05)

**One wafer, one process (imec 300 mm R&D).** In processes where qubit junctions are not probed, only nearby test structures
are. Notebook 05 asks what those can tell you on this one wafer, using imec's public data: Van Damme et al., Nature 634, 74–79
(2024), and the accompanying Zenodo dataset (CC-BY 4.0; downloaded by the notebook, not stored here): 7,872 test junctions
on 82 dies and 393 measured qubits on 36 dies. None of this is a general claim about other fabs.

- **Visible to test arrays vs not.** Of the test-junction R×A variance, the part a test array can see (smooth wafer trend +
  die-to-die, combined) is **~38–55%** per junction size; the local, within-die part it cannot see for a neighbouring qubit
  is **~45–62%** (σ ≈ 8–10% for large junctions, 12–16% at 80–100 nm). We report trend and die-to-die together because
  how the variance splits between them depends on the order of the trend fit. Eight copies of a junction on one die
  disagree about as much as junctions spread over the whole die.
- **Predicting each qubit from its die's test arrays** (35 dies, leave-one-die-out): the die-shared error is ~120 MHz
  (an independent estimate gives ~125 MHz), the per-qubit error ~274 MHz (5.2%). That per-qubit error is what the
  test-junction scatter alone predicts (~272–279 MHz).
- **Against collision windows (a yardstick only).** These imec chips were not laid out against frequency collisions, so
  Hertzberg et al.'s Table 1 windows serve only as a scale: only ~5% of qubits fall within ±17 MHz of their prediction.
  On this wafer a test array tracks the die average, not the per-qubit part that would cause collisions in a collision-aware
  layout; per-qubit probing, post-fabrication tuning or lower junction scatter would be needed.
- **Capacitor (E_C) scatter**, from the measured anharmonicity: ~2%, mostly within-die, ~2.5× Osman et al.'s 0.8%. This is
  an **upper bound**: the dataset gives no uncertainty for the anharmonicity, so fitting noise is included.

## What we reproduced

**Room-temperature resistance → qubit frequency** (Eq. 1 of Mayer, Weber et al. 2025, T_c = 0.71 K, C_q = 86 fF): notebook 01

| | Paper | This repo |
|---|---|---|
| f01 at Rn = 7.1 kΩ | design 4.42 GHz | 4.397 GHz |
| Rn needed for 4.42 GHz | ≈ 7.1 kΩ | 7.03 kΩ |
| f01 error if the bulk-Al T_c (1.2 K) is used instead | – | +31.5% |

**Collision-free yield, Hertzberg et al. 2021, Table 2** (σ_f = 14 MHz; lattices transcribed from their Supplementary Figs 1–9): notebook 03

| Lattice (qubits) | Yield, paper | Yield, ours | Mean collisions, paper | Mean collisions, ours |
|---|---|---|---|---|
| Heavy-hex d=3 / 5 / 7 (23 / 65 / 127) | 70 / 33 / 8% | 70.0 / 29.7 / 10.5% | 0.4 / 1.2 / 2.7 | 0.38 / 1.19 / 2.35 |
| Heavy-square d=3 / 5 / 7 (25 / 73 / 145) | 67 / 27 / 6% | 66.6 / 26.9 / 6.6% | 0.4 / 1.5 / 3.5 | 0.45 / 1.46 / 3.00 |
| Square d=3 / 5 / 7 (17 / 49 / 97) | 6 / <0.1 / <0.1% | 5.6 / 0.0 / 0.0% | 3 / 10 / 23 | 2.29 / 8.10 / 19.35 |

Where we differ: our square lattices show ~20% fewer mean collisions than the paper. We could not pin down why. The exact
counting rules and the frequency-step sweep range are not published.

## What is ours vs assumed

| Item | Value | Status |
|---|---|---|
| Rn variance split: wafer trend / die offset / junction | 50 / 25 / 25% | Assumption (Kreikebaum et al.'s data suggest the local share is ~20–45%) |
| Residual split: per-qubit / die-shared | 100 / 0% (sensitivity run at 50 / 50%) | Assumption |
| Capacitance C_q in the ranking study | 68.1 fF | Derived: chosen so exact anharmonicity = −330 MHz (Hertzberg et al.) |
| Chip frequencies | 5.00 / 5.07 / 5.14 GHz, fixed | Example from Hertzberg et al.; not optimised |
| Edge exclusion | 4 mm | Assumption, chosen to land near Lang et al.'s ~375 dies |
| Rn probe noise | 0.1% | Hertzberg et al. ("better than 0.1%") |
| Rn spread presets | 2% / 0.5% | Križan et al. 2026 (citing their ref. [9]) / Wang et al. 2026 |
| Residual presets | 14.5 MHz / 1.6% | Hertzberg et al. 2021 / Mayer, Weber et al. 2025 |
| Frequency-step sweep (notebook 03) | 30–150 MHz | Our choice (range not published) |
| Capacitor (E_C) scatter, realism sweep | 0.8% (~19 MHz per qubit) | Osman et al. 2023, Fig. 2(b); measured across 4 dies, split not reported: run fully per-qubit (upper bound) and 50 / 50 |
| Per-qubit anharmonicity | α_q = −330 MHz × E_C,q / E_C,design | Approximation (truth only; the ranking uses the design α) |
| Shared Rn ageing, probe → cooldown | 4% | Osman et al. 2023 (+4% in 21 days); Stehli et al. 2020 report +1.6% over 6 months |
| Per-junction Rn ageing | 0 / 0.25 / 0.5 / 1% | **Assumed: no published per-junction value** |

Combining the papers: the ranking study uses an IBM-style chip and collision rules with Fraunhofer's
room-temperature → frequency model. It is not a model of any one fab's chip.

## Limitations

- **Ranking results use synthetic data only.** Wafers are generated from published statistics. Notebook 05 checks one real
  wafer (test arrays vs qubits), but the ranking itself is not validated on real fab data.
- **Per-qubit probing assumed.** The ranking assumes per-qubit room-temperature Rn. In many processes only nearby test
  structures are probed, which see die-level variation but not the per-qubit part that causes collisions.
- **Simplified frequency plan.** One fixed 3-frequency pattern; real chips also have resonators, couplers and other constraints.
- **Results depend on assumptions**, especially the variance split, which sets how much within-die variation the probe can see.
- **The headline depends on scatter the probe cannot see.** How much of the capacitor scatter is per-qubit (not reported
  in the source) moves the untrimmed advantage between ~1.5× and ~3×; per-junction ageing has no published value at all.
- **Small numbers of good dies.** In the realism sweep many wafers have only 2–8 collision-free dies, so the ranking gain
  scatters by ±0.1–0.3 between wafers; cooldowns-to-5 is undefined on wafers with fewer than 5 good dies (counted, not dropped).
- **No trimming model.** "Trimmed" means a narrower Rn spread (0.5%); the trimming process itself is not simulated.
- **Simple models**: Ambegaokar–Baratoff plus the second-order transmon formula (Križan, PhD thesis, Chalmers 2026, Eq. 4.3;
  Eq. 1 of Mayer, Weber et al. is kept only to reproduce their paper) with one fitted effective T_c; Gaussian residuals;
  Table 1 collision windows taken as hard cut-offs.
- **The fitted T_c is effective and already contains the cold/warm resistance change.** A qubit's E_J is set by its junction
  resistance at millikelvin, which is higher than the room-temperature value by a process-dependent factor k (k = 1.1375 for
  Chalmers Al/AlOx junctions, Križan 2026 thesis p. 49, "not to be considered universal"). Since E_J ∝ T_c / (k·R_RT),
  room-temperature data cannot tell T_c and k apart: the fitted value is T_c/k, together with any area or series-resistance
  error. k depends on junction and process, so this scale has to be fitted per design (one design in notebooks 01–04;
  per-design offsets in notebook 05), not copied from another lab. (Our reading, not a claim of either paper: Fraunhofer's
  fitted 0.71 K is 1.7× below bulk Al's 1.2 K, more than k ≈ 1.14 alone would explain.)

## How to run

Tested on Python 3.12 and 3.13.

```bash
pip install -e ".[all]"
```

Run the tests and execute all five notebooks (notebook 05 downloads ~1.3 MB of public data from Zenodo on first run):

| | macOS / Linux | Any OS (including Windows) |
|---|---|---|
| Tests + notebooks | `make all` | `python run_all.py` |
| Tests only | `make test` | `python -m pytest` |
| Optional data-download test (Zenodo; skipped by default and in CI) | `make test-network` | `python -m pytest -m network` |
| Also save notebook outputs and regenerate `docs/capture_curve.png` | `make notebooks` | `python run_all.py --inplace` |

| Notebook | Content |
|---|---|
| `01_rt_to_frequency` | Ambegaokar–Baratoff + transmon chain; Fraunhofer design point; exact diagonalisation |
| `02_synthetic_wafer` | 200 mm synthetic wafer; predicted vs measured f01 |
| `03_collision_yield` | Reproduction of Hertzberg et al. 2021; bridge from wafer spread to yield |
| `04_die_ranking` | Die ranking, residual sweep, calibration, drift, realism sweep with capacitor scatter and ageing (expected vs actual for each experiment) |
| `05_imec_test_arrays` | Real data (imec, 300 mm): variance of test-junction R×A, test array → qubit frequency, E_C scatter (predicted vs actual) |

## References

- T. Mayer, J. Weber et al., *CMOS-Compatible, Wafer-Scale Processed Superconducting Qubits Exceeding Energy Relaxation Times of 200 µs*, [arXiv:2505.08424](https://arxiv.org/abs/2505.08424) (2025).
- S. J. K. Lang et al., *CMOS-compatible processing and room-temperature characterization on wafer-level for scalable quantum computing*, Phys. Rev. Applied 24, 054052 (2025), [doi:10.1103/42nj-fjg8](https://journals.aps.org/prapplied/abstract/10.1103/42nj-fjg8), [arXiv:2504.18173](https://arxiv.org/abs/2504.18173).
- J. B. Hertzberg et al., *Laser-annealing Josephson junctions for yielding scaled-up superconducting quantum processors*, npj Quantum Information 7, 129 (2021), [doi:10.1038/s41534-021-00464-5](https://doi.org/10.1038/s41534-021-00464-5), [arXiv:2009.00781](https://arxiv.org/abs/2009.00781); and Hertzberg et al. 2021, Supplementary Information.
- C. Križan et al., *Electrical post-fabrication tuning of aluminum Josephson junctions at room temperature*, Materials for Quantum Technology (2026), [doi:10.1088/2633-4356/ae7fe1](https://iopscience.iop.org/article/10.1088/2633-4356/ae7fe1/meta), [arXiv:2602.20002](https://arxiv.org/abs/2602.20002).
- X. Wang et al., *Scaling Alternating-Bias-Assisted Annealing for Precision Transmon Frequency Targeting on Superconducting Quantum Processors*, [arXiv:2608.27789](https://arxiv.org/abs/2608.27789) (2026).
- J. M. Kreikebaum et al., *Improving wafer-scale Josephson junction resistance variation in superconducting quantum coherent circuits*, Supercond. Sci. Technol. 33, 06LT02 (2020), [doi:10.1088/1361-6668/ab8617](https://doi.org/10.1088/1361-6668/ab8617), [arXiv:1909.09165](https://arxiv.org/abs/1909.09165), including its supplementary material.
- A. Osman et al., *Mitigation of frequency collisions in superconducting quantum processors*, Phys. Rev. Research 5, 043001 (2023), [doi:10.1103/PhysRevResearch.5.043001](https://doi.org/10.1103/PhysRevResearch.5.043001).
- A. Stehli et al., *Coherent superconducting qubits from a subtractive junction fabrication process*, Appl. Phys. Lett. 117, 124005 (2020), [doi:10.1063/5.0023533](https://doi.org/10.1063/5.0023533), [arXiv:2006.16862](https://arxiv.org/abs/2006.16862).
- J. Van Damme et al., *Advanced CMOS manufacturing of superconducting qubits on 300 mm wafers*, Nature 634, 74–79 (2024), [doi:10.1038/s41586-024-07941-9](https://doi.org/10.1038/s41586-024-07941-9).
- J. Van Damme, dataset for the above, Zenodo, [doi:10.5281/zenodo.13143313](https://doi.org/10.5281/zenodo.13143313) (2024), licence [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used in notebook 05; not redistributed here.
- J. Koch et al., *Charge-insensitive qubit design derived from the Cooper pair box*, Phys. Rev. A 76, 042319 (2007).
- V. Ambegaokar and A. Baratoff, Phys. Rev. Lett. 10 (1963) (critical current from normal-state resistance).
- [scqubits](https://scqubits.readthedocs.io) (used only in tests, as an independent check of the transmon spectrum).

## License

MIT (see `LICENSE`).

---

Have real wafer data (RT resistance + cryo frequencies)? I'd like to test this on it.
Email: divyeshidhate2709@gmail.com · LinkedIn: https://www.linkedin.com/in/divyesh-idhate-bb1170218
