# Battery digital twin & operando diagnostics (v4.4)

```bash
pip install -r requirements.txt
streamlit run app.py                      # "Load synthetic demo cohort" works with no data
pip install -r requirements-dev.txt && pytest
python benchmark.py --master data/master.parquet --imp data/impedance.parquet --out results/bench.parquet
python benchmark.py --synthetic --fracs 0.3 0.5 --paradigms ML Twin SemiEmp PF   # ground-truth sanity run
docker build -t battery-twin . && docker run -p 8501:8501 -v twin-data:/data battery-twin
```

Files: `twin_engine.py` (all computation, no UI), `app.py` (Streamlit views), `benchmark.py`
(offline sweep → results file the app can load), `tests/` (46 tests incl. gradient checks and
synthetic-truth recovery). Set `TWIN_CACHE_DIR` to persist downloads and uploads; `GIT_COMMIT`
is recorded in run manifests.

## Operations centre (first view)

Live status bar, fleet KPIs, an instrument cluster (SOH, quick RUL, resistance growth, peak temperature
gauges) for the selected battery, a fleet health treemap, a risk matrix (remaining life vs degradation
speed), a triage table with risk levels and alerts, a filterable event log (knees, EOL crossings,
over-temperature, cold charging, regeneration, excluded cycles) and a one-click HTML report.
The Operations view adds a what-if scenario planner driven by the cohort stress-factor law.

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
