# Self-updating digital twin for Li-ion battery diagnostics (v5.6.0)

Streamlit platform and Python engine for the internship study on NASA Ames 18650 LiCoO₂/graphite ageing
data. A physical cell and its virtual copy are synchronised cycle by cycle to diagnose ageing, forecast
remaining life and optimise operation and maintenance.


## Level 3 · Deep learning and usability (v5.6.0)

- Level 3 · Deep learning, same approach as Level 2 (same question, data, training batteries, origins, anchoring,
  cross-validated bands), independent of Levels 1-2 (`train_deep_trajectory`, `DL_MODELS`): MLP on exactly the
  Level-2 inputs; GRU and Transformer reading the last 10 cycles (forecast: SOH, resistance, temperature rise, CV time
  up to the origin -> SOH change at 32 horizons; measured: mean voltage, resistance, charge times, temperatures,
  efficiency -> SOH of the cycle). In the accuracy assessment as dl: / dlm:.
- Each level keeps the results of both modes in separate tabs (training the other mode does not erase them).
- Training and test battery pickers are mutually exclusive.
- Baselines stop at the battery's last measured cycle.
- Accuracy assessment ~36 % faster with identical results: no permutation importance and 2-fold bands in
  assessment runs (synthetic, 20 forecasts: 120 s -> 77 s; the Gaussian Process is the slowest model, ~12 s per
  forecast).
Synthetic single-battery check (30 % origin, RMSE): forecast GP 0.042, MLP 0.068, GRU 0.036, Transformer 0.038;
measured GP 0.076, MLP 0.011, GRU 0.036, Transformer 0.026 (one battery; judge with the assessment).

## Fixes after the first real assessment (v5.5.5)

- Training pool (`v2_training_pool`): the battery's own condition group when it has >= 3 other batteries, otherwise all
  comparable batteries (fleet shape, Level 2 and the within scheme).
- Fade speed at the origin: Theil-Sen on the actual SOH (3-cycle median) of the transient-free cycles, instead of the
  running-minimum curve that stayed flat after a recovery jump; the target's prepared history uses the same values.
- Fleet shape: beyond the range covered by >= 2 batteries it continues with the median of the batteries' own
  late-life fade rates (it used to level off); its band is interpolated between horizon bins (no steps).
- Gaussian Process: one length scale per input plus a linear term (it collapsed to the mean on real data), at most
  400 training rows, bounded optimiser.
- Measured mode: trailing 5-cycle median of the predictions (causal).
- Scoreboard verdict: "better / worse on all N batteries", noting when N is too small for any test to reach 5 %
  (two-sided Wilcoxon minimum p = 2 / 2^N, i.e. 0.0625 with 5 batteries).
Synthetic Reference-like group (5 batteries, recovery jump before the 30 % origin), RMSE / bias / |EOL error|:
fleet shape 2.47 / +0.16 / 4.8 -> 2.57 / -0.72 / 3.8; Gaussian Process 5.97 -> 4.71; Random Forest, Bayesian
Ridge, SVM unchanged within 0.3; linear trend stays ~9 points optimistic (a straight line cannot anticipate a knee).

## Two independent levels (v5.5.4)

Models view = Level 1 · Baselines (persistence, linear trend, fleet shape) and Level 2 · Machine learning, plus the
leaderboard and the accuracy assessment, which compare the levels but do not connect them. No model uses another
level's output (the ML v2 workbench, built as baseline + correction, and the v1/v2 switch are no longer in the view).
Level 2 = the trajectory models, organised around Mission 2: SOH of every cycle learned from all training batteries.
Forecast-mode fixes: tree models (Decision Tree, Random Forest, XGBoost, LightGBM) learn the average fade per Ah since
the origin (rows >= 5 cycles ahead, clipped to [-2 %/Ah, 0]); linear and kernel models learn the SOH change and are
anchored (their own prediction at zero cycles ahead is subtracted), so every forecast starts at today's SOH; training
origins 5-60 % of life, with a warning when the chosen origin is outside. Synthetic (3 batteries, origins 8 / 30 %):
jump at the origin 0.019 -> 0.003; Random Forest 0.0148 -> 0.0115 at 30 %; SVM unchanged; Bayesian Ridge 0.020 ->
0.031 at 30 % (its earlier score partly came from an offset).

## Trajectory models (v5.5.3)

`train_trajectory_model`: SOH of every cycle learned from every cycle of the training batteries, predicted cycle by
cycle for the selected battery (Models › "Trajectory models"; also in the accuracy assessment as traj: / trajm:).
Forecast mode (a real forecast): cycle number, cycles since the origin, estimated throughput, ambient temperature,
discharge current, cut-off, and the battery's state at the origin (SOH, SOH trend per Ah, resistance and its trend,
CV-charge time, dQ(V) variance); target = SOH change since the origin; training rows from origins at 10-60 % of
each training battery's life. Measured mode (estimation): cycle number, conditions and each cycle's own
measurements (load-step resistance, temperature rise, mean temperature, CC/CV charge times, mean voltage,
efficiency, EIS Re/Rct). Bands from cross-validation by battery; permutation importance; stops at the last
measured cycle. Synthetic (10 batteries, 4-43 °C, 1-4 A, forecast from 30 %): results vary by battery (GP 0.0059
vs fleet shape 0.0147 on one, 0.0288 vs 0.0105 on another), so judge them with the accuracy assessment;
measured mode 0.0045-0.0073 with SVM / Bayesian Ridge.

## Forecast shape and accuracy assessment (v5.5.2)

- ML v2 predicts 12 horizons (+5 … +140 cycles) joined by a smooth, monotone (PCHIP) curve instead of 7 points
  joined by straight segments.
- Fleet-shape baseline (`fleet_shape`, `shape_drop`, `fleet_shape_forecast`): the typical fade curve of similar
  batteries, placed at this battery's state and speed. Third Level 1 baseline, and ML v2's default starting
  point (option: straight trend).
- Accuracy assessment (`run_protocol`, `protocol_scoreboard`): each battery of a condition group held out in turn,
  several origins, per-battery averaging, 95 % CI, paired verdict against the best baseline, CSV export.
Synthetic check (6 early-knee batteries with recovery jumps, ML v2 with Bayesian Ridge and Random Forest, 3 test
batteries, mean RMSE from 15 % / 40 % of life): v5.5.1 0.0619 / 0.0375; smooth horizons + straight trend
0.0573 / 0.0359; smooth horizons + fleet shape (default) 0.0301 / 0.0346. The fleet shape gives the main gain,
mostly at early origins; smoothing alone helps little.

## Models view in v5.5.1

Restored from v5.5 and reduced to Level 1 · Baselines and Levels 1–3 · the machine-learning workbench (plus their
leaderboard). Advanced levels (deep learning, hybrid, first principles), early-life prediction and the cross-cell
benchmark are hidden from the view (the engine keeps them) and will be added back step by step once the ML models
are developed and validated.

## Study objectives (Missions)

| Mission | Questions |
|---|---|
| M1 · Health | Which parameter best represents health? One or several? How do operating conditions and degradation mechanisms (LLI, LAM, CL) shape ageing? |
| M2 · Self-updating twin | Can future degradation be predicted accurately? Which variables are most informative? How often should the model update? |
| M3 · Operation & maintenance | Maximise Profit = Revenue − EnergyCost − MaintenanceCost (J_op = Revenue − EnergyCost; J_maint = MaintenanceCost + PerformanceLoss) by choosing the operating policy and the replacement time jointly. |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                     # "Load synthetic demo cohort" works without data
pytest                                   # 70 tests (pip install -r requirements-dev.txt)
python study.py --master data/battery_master_data.parquet --imp data/impedance.parquet --out results/study
python study.py --synthetic --out results/study_demo
pip install -r requirements-service.txt && uvicorn service:app --port 8000   # streaming REST service
```

Deploying on Streamlit Community Cloud: commit **`app.py` and `twin_engine.py` together** (the app checks
`REQUIRED_ENGINE`), Python 3.12, `requirements.txt` without comments, then *Reboot*. Cached results are keyed
on the engine version, so engine updates always recompute.

## Files

| File | Role |
|---|---|
| `twin_engine.py` | All computation, no UI (~6,000 lines, numbered sections) |
| `app.py` | Streamlit front end (5 views) |
| `study.py` | Offline cohort study → CSVs, `summary.md`, manifest |
| `service.py` | Streaming twin: framework-free `TwinRegistry` + optional FastAPI endpoints |
| `benchmark.py` | Offline forecast-origin sweep (ML, Twin, PINN, HB) |
| `tests/test_twin_engine.py` | 70 tests: gradient checks, synthetic-truth recovery, every model and study function |

## Views

1. **Diagnostics.** Scope switch: single battery, selected batteries (up to 8) or whole fleet. Contents:
   - health status: gauges, alerts, risk matrix, triage, event log, HTML report;
   - fade comparison and raw telemetry (5 cycle-selection modes);
   - cohort by ambient temperature;
   - health-indicator ranking and PCA (M1);
   - ICA/DVA and LLI/LAM/CL proxies;
   - half-cell OCV fitting (quantitative LLI, LAM_PE, LAM_NE);
   - stress and safety exposure.
2. **Live twin.** A battery is streamed one discharge at a time (play, pause, step) through:
   - the ECM twin (dual EKF);
   - the mechanistic particle filter (SEI · plating · LAM, driven by measured temperature and current);
   - the particle filter on the physics power law;
   - the adaptive trend Kalman filter;
   - hierarchical Bayes;
   - a live ensemble weighted by each model's recent 5-cycle-ahead error.

   The view also shows live accuracy per model, mechanism shares, the equations, the measurement ablation
   and the update-frequency study (M2).
3. **Models & forecasting: a learning ladder.** A training scheme at the top applies to every level:
   *within a battery* (train on its first part, test on the rest; optionally also learn from the other batteries)
   or *across batteries* (train on chosen batteries, forecast one or more test batteries after seeing the start
   of each). The leaderboard averages over the test batteries:
   - Level 1 · Baselines: persistence, linear trend, decision tree, Bayesian ridge.
   - Level 2 · Classical ML: random forest, extra trees, Gaussian process.
   - Level 3 · Boosting: histogram GB; XGBoost and LightGBM when installed (`requirements-ml.txt`).
   - Level 4 · Deep learning: GRU and single-head Transformer (numpy autodiff, no extra dependency).
   - Level 5 · Hybrid & physics-informed: mechanistic PINN, hierarchical Bayes.
   - Level 6 · First principles: single-particle electrochemical model with SEI growth (`spm_discharge`,
     `spm_forecast`).
   Earlier ML features kept below the ladder:
   - Four curated models: Gaussian Process, Extra Trees, Hist. Gradient Boosting and Bayesian Ridge.
   - Two tasks: forecasting (fade-rate model, conformal bands) and estimating SOH from indicators (random, chronological or by-battery splits).
   - Leakage-free auto-tuning.
   - Early-life ΔQ(V) lifetime model (Severson et al. 2019).
   - Cross-cell benchmark.
4. **Operations & control (M3).**
   - Plant calibrated on the selected battery (`calibrate_plant`).
   - Twin-aware policy, with energy cost included.
   - Integrated operation + replacement optimisation (renewal-reward with sudden-failure hazard).
   - Dynamic-programming policy over (SOH, season), solved with Dinkelbach's method.
   - What-if scenario planner.
5. **Study results.** Runs the live twin on every usable battery and reports per condition group:
   - forecast error, accuracy and band coverage;
   - paired Wilcoxon tests (ensemble vs twin);
   - mechanism-physics checks;
   - the update interval;
   - the plant calibration.

   It states the answer to each Mission question and exports HTML and CSV.

## Key engine API

| Area | Functions / classes |
|---|---|
| Data | `ParquetStore`, `build_cycle_table`, `robust_bol_capacity`, `condition_groups` |
| Diagnostics | `rank_health_indicators`, `hi_pca`, `stress_factor_regression`, `ica_evolution`, `dva_evolution`, `degradation_modes`, `half_cell_trajectory`, `detect_knee`, `fleet_status`, `fleet_events` |
| Twin | `run_dual_twin` (`DualTwinConfig`: `capacity_every`, `voltage_n_eff`, `adaptive`, `adaptive_q`), `twin_config_for` (pulse-aware), `twin_forecast`, `twin_ablation`, `update_frequency_study` |
| Live / Bayesian | `live_multi_model`, `MechanisticStream`, `hierarchical_bayes_forecast`, `live_skill_table`, `live_forecast_accuracy`, `StreamingTwin` |
| ML | `MODEL_SPECS`, `train_ml_forecast`, `train_soh_estimator`, `tune_ml_forecast`, `tune_soh_estimator`, `early_life_lifetime` |
| PINN (offline) | `train_pinn`, `MechanisticPINN` (optional `mode_targets` from half-cell LLI/LAM) |
| Study | `cohort_validation`, `cohort_summary`, `paired_model_test`, `mechanism_checks`, `calibrate_plant` |
| Operations | `CellPhysics`, `Economics`, `MaintenanceModel`, `integrated_om_study`, `solve_replacement_dp`, `evaluate_dp_policy`, `scenario_projection` |
| Testing | `make_synthetic_master(..., knee=...)` with known ground truth |

## Data

- **Master parquet** (one row per sample): `Cell_ID, Cycle_Index, Cycle_Type (charge/discharge/impedance), Time_s, Voltage_V, Current_A, Temp_C, Capacity_Ah[, Ambient_C]`.
- **Impedance parquet:** `Re_ohm, Rct_ohm` per EIS test.
- **End of life:** one definition, the control-bar EOL capacity (default 1.4 Ah, the NASA convention). The Operations replacement SOH is a separate, optimised decision.
- **Known dataset issues, handled:**
  - B0049–B0056: crashed logging, repaired baseline, separate group;
  - B0025–B0028: square-wave load, pulse-aware twin;
  - erratic 4 °C runs: outlier flags.

## ML v2 (v6.1)

`train_ml_v2`: forecast = the battery's own robust recent trend + a learned deviation. Features at the origin:
level and 10-cycle trend of cleaned SOH, load-step resistance, temperature rise, CV-charge time, mean voltage,
energy efficiency, cumulative Ah, Arrhenius factor, C-rate, cut-off. One model per horizon (+10 … +140 cycles)
predicts the deviation directly; grouped cross-validation by battery sets both the band and how much of the
correction is trusted (0 = trend only). Regeneration jumps removed, causal cleaning (no look-ahead),
comparable training pool (mixed, corrupted and pulsed cells excluded by default). Synthetic check with
recovery jumps and a knee: better than v1 in 21/36 forecasts; HistGB RMSE 0.046 -> 0.032 at 12 % of life.

## v5.1 fixes (from the first real-data study)
Condition groups (explicit crashed-logging ids, new "Mixed conditions" group with load/ambient levels),
no duplicate forecasts, best model chosen on common cells with coverage flags, per-cell paired tests,
power labels for small groups, EOL = k consecutive cycles below threshold with "EOL not reachable in
data" status, twin NIS consistency guard. Root cause of twin divergence on B0033-B0044: rate-dependent
capacity under mixed loads (open: rate-normalised SOH). LiCoO2 potential restricted to its valid domain
(y >= 0.48), which also fixes a latent half-cell-fit bound.

## Known limits

- **Synthetic evidence.** Most performance numbers so far come from synthetic cohorts. The study results on the real NASA data are the evidence to report.
- **Mechanism attribution.** It is model-based; validate it against half-cell LLI/LAM.
- **Hot knees.** Hot cells with a sudden knee remain hard for every model; the live ensemble helps most there.
- **Mission 3 simplifications.** Mission 3 uses a lumped plant and an assumed sudden-failure hazard, so read the location of the optimum rather than its absolute value.

## References

- Plett (EKF twins)
- Dubarry et al. 2012; Birkl et al. 2017 (degradation modes, half-cell fitting)
- Ramadass et al. 2004; Doyle et al. 1996 (electrode potentials)
- Severson et al. 2019 (early-life ΔQ)
- Saha & Goebel 2009 (particle filter on NASA cells)
- Saxena et al. 2010 (prognostic metrics)
- Coble & Hines 2009 (health-indicator criteria)
- Rufino Júnior et al. 2024; Menye et al. 2025 (degradation reviews)
