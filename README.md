# Battery digital twin & operando diagnostics (v5.0)

```bash
pip install -r requirements.txt
streamlit run app.py                      # "Load synthetic demo cohort" works with no data
pip install -r requirements-dev.txt && pytest
python benchmark.py --master data/master.parquet --imp data/impedance.parquet --out results/bench.parquet
python benchmark.py --synthetic --fracs 0.3 0.5 --paradigms ML Twin SemiEmp PF   # ground-truth sanity run
docker build -t battery-twin . && docker run -p 8501:8501 -v twin-data:/data battery-twin
```

Files: `twin_engine.py` (all computation, no UI), `app.py` (Streamlit views), `study.py` (offline cohort
study), `service.py` (streaming REST service), `benchmark.py`
(offline sweep → results file the app can load), `tests/` (55 tests incl. gradient checks and
synthetic-truth recovery). Set `TWIN_CACHE_DIR` to persist downloads and uploads; `GIT_COMMIT`
is recorded in run manifests.

## New in v5.0: study release

- **Study results** view (and `study.py` CLI): runs the live multi-model twin on every usable battery, scores
  20-cycle forecasts at 30 % and 50 % of life per operating-condition group (Reference, High current, Hot, Cold,
  Pulsed load, Corrupted logging), paired Wilcoxon tests (live ensemble vs twin), mechanism-physics checks
  (plating in the cold, SEI when hot, LAM under high current), cohort update-interval study, and states the
  answer to each Mission question with its evidence; HTML report and CSV export.
- NASA idiosyncrasies handled: square-wave cells detected (rest fraction) and run with a pulse-aware twin;
  crashed-logging cells reported as their own group.
- Mission 3 calibrated on real batteries: `calibrate_plant` sets the Operations plant from the battery's own
  capacity, resistances and fade rate plus the cohort stress law; DP and grid studies plan for that battery.
- Deployment path: `service.py` (REST, FastAPI optional) with a framework-free `TwinRegistry` around
  `StreamingTwin` (per-cycle ingest -> SOH, RUL, dominant mechanism, alarms).
- Clean-up: retired semi-empirical law, physics-mean GP, stacked ensemble, live PINN and four dominated ML
  models (MLP, RF, Ridge, ElasticNet); removed dead UI code. The power-law particle filter was kept after the
  tests showed it is the best live model on knee cells.

## New in v4.9: mechanism-aware live twin

- Mechanistic particle filter (`MechanisticStream`): SEI growth, lithium plating and loss of active material
  integrated cycle by cycle with the *measured* temperature and current, so operating conditions decide which
  mechanism grows (Arrhenius SEI when hot, cold-gated plating, C-rate LAM). Rate constants and latent losses
  are estimated jointly; the Live twin shows the mechanism shares live and the dominant mechanism.
  Synthetic harsh cells: 20-cycle RMSE 0.003 at 4 °C, 0.004 at 4 A, 0.008 at 43 °C + 4 A (twin: 0.005 /
  0.009 / 0.060).
- Optional live mechanistic PINN, retrained every 25 cycles (experimental, slow).
- Models view is machine learning only (workbench, early-life ΔQ(V), cross-cell benchmark). The PINN, the
  physics equations, the measurement ablation and the update-frequency study moved to the Live twin.

## New in v4.8: live multi-model twin

- The Live twin streams several models cycle by cycle: ECM twin (dual EKF), particle filter on the physics
  power law with the fleet prior, an adaptive trend Kalman filter (level-slope-curvature, bends quickly at
  knees) and hierarchical Bayes. A live ensemble weights them by their recent 5-cycle-ahead error
  (`live_multi_model`, `live_skill_table`). On synthetic knee cells (fade 2.5-3x faster after the knee)
  the ensemble's 20-cycle forecast error is ~2.5x lower than the twin alone, with honest bands.
- Twin: Sage-Husa adaptive process noise on the degradation rate (`adaptive_q`), so k re-converges quickly
  after a regime change.
- Synthetic generator: optional knee (`knee={cell_index: (n_knee, factor)}`) for harsh-regime tests.

## New in v4.7

- Twin accuracy: voltage-model errors treated as correlated (effective independent samples, default 3) and an
  adaptive EKF that re-opens to the data under persistent mismatch; optional periodic reference capacity checks
  (app default: every 10 cycles). On noisy synthetic data: SOH bias 0.019 -> 0.004, ±2σ coverage 25 % -> 86 %.
- Automatic hyperparameter tuning (random search). Forecast task: validated by forecast backtests on the
  nearest-condition *other* batteries; estimation task: grouped CV by battery inside the training split.
  Robust score mean + 0.5 SD; defaults always a candidate. MLP now has small-data defaults and early stopping.
- Cohort plot: one figure per ambient temperature with its own legend; data gaps shown as gaps.

## Diagnostics (first view)

One view, one selection. A scope switch chooses **Single battery**, **Selected batteries** (up to 8, or
"Select most critical" from the fleet ranking) or **Whole fleet**. The health status adapts: instrument
cards (SOH, quick RUL, resistance growth and peak-temperature gauges with alert banner) per battery, or the
fleet treemap and triage table. The risk matrix always shows the selection against the grey fleet. Every
section below (fade comparison, raw telemetry, cohort grid, health indicators, ICA/DVA, degradation modes,
half-cell fitting, stress exposure, event log, HTML report) follows the same selection; single-battery
analyses use a "focus battery" chosen from the selection.

## New in v4.5 (model clean-up)

Removed (dominated on every benchmark): Gradient Boosting (kept the faster Hist. GB), AdaBoost, Decision Tree,
k-NN, SVR, Kernel Ridge, the direct SOH(n) ML strategy, the joint per-sample EKF from comparisons, and the
double-exponential particle filter. Added: early-life ΔQ(V) features and a Severson-style lifetime model;
hierarchical Bayesian degradation model (fleet prior with temperature/current covariates); particle filter and
physics-mean GP on the same physics law; skill-weighted stacked ensemble; PINN mechanism states constrained by
half-cell LLI/LAM.

## New in v4.4

- **Live twin replay** (second view): streams a battery's life one discharge at a time with play / pause /
  step; the dual EKF assimilates each cycle, and the forecast band, predicted end of life and personal
  degradation rate k update live (`replay` uses `run_dual_twin` + `twin_forecast`).
- **Optimal operation + replacement by dynamic programming** (Operations): semi-Markov DP over
  (SOH, season phase) with Dinkelbach's method on the renewal-reward rate, sudden-failure hazard, energy cost
  and downtime (`solve_replacement_dp`, `DPPolicy`, `evaluate_dp_policy`).
- **Half-cell OCV fitting** (Data): LiCoO₂ (Ramadass 2004) and graphite (Doyle 1996) potentials fitted to the
  IR-compensated discharge curve for quantitative LLI, LAM_PE and LAM_NE (`fit_half_cell`,
  `half_cell_trajectory`).

## What answers which project question

| Mission question | Where in the app | Engine function |
|---|---|---|
| M1 · Which parameter best represents health? | Data › health indicators | `rank_health_indicators` |
| M1 · One parameter or several? | Data › PCA card | `hi_pca` |
| M1 · How do operating conditions influence degradation? | Data › cohort expander, stress & safety | `stress_factor_regression`, `stress_exposure` |
| M1 · Degradation mechanisms (LLI / LAM / CL) | Data › ICA, DVA, modes | `ica_evolution`, `dva_evolution`, `degradation_modes`, `detect_knee` |
| M2 · Can future degradation be predicted accurately? | Models › comparison, cross-cell benchmark | `compare_paradigms`, `run_benchmark` |
| M2 · Which variables are most informative? | Models › ablation (information gain) | `twin_ablation` |
| M2 · How often should the model update? | Models › update frequency | `update_frequency_study` |
| M3 · Integrated operation + maintenance | Operations › integrated optimisation | `integrated_om_study`, `renewal_evaluation` |

ML workbench: 14 scikit-learn models with editable hyperparameters, two tasks (forecast future SOH;
estimate current SOH from operando indicators) and train/test splits by fraction, by time or by battery.
Forecasting paradigms: ML fade-rate surrogates with conformal bands; dual-EKF ECM twin; semi-empirical power law in Ah; particle filter (double exponential);
hybrid PINN, lumped or mechanism-resolved (SEI growth, lithium plating, loss of active material,
resistance coupling, Butler–Volmer and mean-voltage electrochemistry).

Beginning-of-life capacity is estimated robustly: invalid low-start segments (B0049–B0056, crashed logging)
are detected and excluded, and such cells are flagged `baseline_suspect`.

Known limits: activation energy is not identifiable per cell (fixed or pooled by default);
split-conformal bands under-cover when the target cell is unlike the calibration cells; the PINN
ensemble band is epistemic only; LLI/LAM values are proxies from ~1C full-cell data; the
sudden-failure hazard in the O&M study is an assumed model, so read the optimum's location rather
than its absolute value; the one-step policy optimises per-cycle margin, not lifetime rate (DP is
future work).
