# STATUS · handover for a new conversation · engine/app v5.31.0 · 2026-10-27

Read this first. It replaces every earlier STATUS.md. Full change history: README.md.

## 1. Project in one paragraph
Internship (Yokogawa, UM6P collaboration): a self-updating digital twin for the NASA Ames 18650 Li-ion ageing data
(LiCoO2/graphite, 34 cells, 4-44 C, 1-4 A). Three missions from the research plan: M1 health indicators and degradation
mechanisms; M2 self-updating twin (can degradation be predicted, which variables are informative, how often to update);
M3 integrated operation and maintenance. Deliverable: a Streamlit app (app.py) + a tested engine (twin_engine.py).

## 2. Files and versions (always deliver both .py files from the same release)
- twin_engine.py  ENGINE_VERSION = "5.31.0"   (all computation; no Streamlit)
- app.py          REQUIRED_ENGINE = "5.31.0"  (exact version guard + required-function check)
- tests/test_twin_engine.py: 80 tests (passing at v5.11.0; not rerun since, file not in the Project) +
  tests/test_cv_selection.py (5) + tests/test_frozen_live.py (10), passing at v5.15.0;
  tests/test_integrated_twin.py (12, v5.31.0)
- requirements.txt: streamlit>=1.46,<1.66 (st.components.v1.html is deprecated; verified with 1.65), xgboost, lightgbm
- battery-twin.zip: complete local package (INSTALL.bat one-click Windows installer -> %USERPROFILE%\battery-twin-app,
  Desktop shortcut "Battery Digital Twin"; setup/run scripts for Windows and Mac; .streamlit/config.toml upload 1 GB)
- Runs on Streamlit Community Cloud (slow: shared 0.08-2 CPU) and locally (recommended for assessments).
  User's laptop: Windows, Python 3.13.12 via the "py" launcher (python not on PATH) -> scripts use "py -3".

## 3. App structure
Views: Home (animated cell <-> twin banner: KEEP IT), Diagnostics, Live twin, Models & forecasting, Operations &
control, Study results. Design: software look (tab strip, panels, full width) with the ORIGINAL colours (blue-green
gradient); no feature chips. Upload boxes on the start screen + sidebar. Control bar: target cell, end of life
(fixed Ah or % of initial capacity), "Reference-condition SOH" switch.

Models & forecasting (independent levels, compared only in leaderboard/assessment):
- Training scheme: within / same operating conditions / across; exclusive training and test pickers; share of life
  seen (origin); safe_origin moves an origin forward if < 10 valid cycles before it (stated).
- Level 1 baselines: persistence, linear trend, fleet shape.
- Level 2 ML (forecast + measured modes): Bayesian Ridge, DT, RF, SVM, GP (ARD + linear), XGBoost, LightGBM.
- Level 3 DL: MLP (Level-2 inputs), GRU, Transformer (last 10 cycles).
- Level 4 PINN (forecast only): LLI_SEI, LLI_plating, LAM with mechanism equations + ablation (no physics).
- Per level: equations (sigma icon), parameters, automatic selection, Self-optimise (remembered per battery set),
  self-weighted ensemble (1/CV^2), star = recommended (lowest CV error), indicator chart for a chosen model,
  "Which training batteries perturb the models?" (leave-one-battery-out influence).
- Accuracy assessment (v5.12 default: best 2 per level chosen by CV inside each fold, + baselines + PINN;
  star-agreement table; manual model choice still available): each battery held out in turn, cached per forecast (resumable), scoreboard with 95% CI
  (none below 3 batteries), verdict vs best baseline, per-battery winners, CSV.
Live twin: replay with ECM dual EKF, mechanistic PF, power-law PF, adaptive trend KF, hierarchical Bayes, live ensemble;
update-frequency study; measurement ablation.

## 4. Decisions (do not reopen without new evidence)
- MISSION 2 CLOSED (2026-10-21): update study complete for Hot, Reference, Cold (B0045-B0048), Pulsed; Cold B0041
  and the frozen-model vs filters scoreboards dropped by user decision.
- NEXT PHASE "full-signal physics": (1) dQ/dV degradation-mode indicators, (2) PINN v2 (all relevant NASA signals as
  inputs and targets, cross-group training, plating by charge, LAM by cut-off/DoD, priors), (3) all levels get
  dQ/dV + EIS inputs, (4) assessment capacity-based vs full-signal; synthetic benchmark with known mechanisms.
- Role of the PINN (2026-10-21): the twin's EXPLANATORY model (mechanism attribution: SEI, plating, LAM); ML/DL are
  its accuracy benchmark. No tuning of the test to make it win. Next focus after Mission 2: can the attribution be
  trusted? (identifiability, seed stability, validation against EIS Re/Rct, physics fixes: plating driven by charge
  current/temperature, cut-off voltage as input, low-capacity outliers).
- No frozen-model update study for Mixed conditions (user decision, 2026-10-21): conditions change during the test,
  so updating against measured capacity mixes ageing with load/temperature effects; at reference condition only
  15-29 gapped cycles remain. Mixed stays an exploratory assessment group only.
- FEATURE FREEZE from 2026-10-19 until Mission 2 is closed: no new app features, bug fixes only. v5.15.1 is the
  version of record for all Mission 2 results (v5.15.2 = bug fix: update study applies the assessment's data rule). Extra studies run as standalone scripts (scripts/).
- Small samples framed everywhere: "Results are from 4-6 batteries per condition group; differences under about
  0.1-0.2 SOH points are within battery-to-battery variation, and no claim rests on a single battery."
- Every assessment result uses each model's DEFAULT settings: the assessment never applies self-optimisation or
  automatic parameter selection (Models-view tuning does not reach it). Frozen models = default settings, v5.14.1.
- Frozen models (user decision): per group and mode, the learned model with the lowest mean test RMSE over all
  batteries, baselines and PINN ablation excluded (mission2_frozen_models.csv). Report next to the CV-chosen
  (honest) score from mission2_summary_by_group.csv, because choosing on the test batteries is optimistic.
- Assessment groups: all usable groups; Pulsed and Mixed labelled exploratory; Mixed always at reference
  condition; Corrupted logging always excluded; membership from recorded conditions, never from toggles (v5.14).
- CV holds out TRAINING batteries only; the predicted battery's own past always stays in the fit (v5.13).
- Levels independent (no model uses another's output). Two modes: forecast vs measured.
- Forecasts anchored at today's SOH; SOH now = median of last 3 valid transient-free cycles (v5.11).
- Trees learn fade rate per Ah; smooth models learn the change; inputs limited to training range +-50 %; forecast at
  most +2 SOH points above today (v5.11).
- CV (held-out) error = same task as the test: held-out training batteries cut at the same share (3 cuts) (v5.11).
- Training rows capped (6,000; SVM 4,000) sampled evenly across batteries; data-fingerprint caches (identical results).
- Keep data as they are; show which batteries perturb a model instead of removing campaigns.
- Report per condition group, never pooled.

## 5. Real NASA results so far (v5.10.x, BEFORE the v5.11 fixes; origin 30 % unless noted)
| Group | Forecast winner | vs best baseline | SOH-now (measured) best | Physics |
|---|---|---|---|---|
| Reference (B0005 only) | GP 1.25 SOH pts (fleet shape 3.24) | clearly better | MLP 1.8, GRU 2.3 | hurt (PINN 8.3 vs ablation 1.8) |
| Hot 43 C/4 A (4) | PINN 0.97 (ablation 1.20; GRU 1.53) | 14 models better on all 4 | Bayesian Ridge 0.87, GP 0.88 | HELPS (SEI-dominated) |
| Cold 4 C/1 A (4) | GP 3.15 (fleet shape 3.94) | no clear difference | weaker than forecast (4.7-5.2) | fails (8.3) |
| Pulsed load (2 of 4 scored) | inconclusive; B0025: GP 0.74 | - | GP 0.54 | - |
| Mixed conditions (B0044) | RF 1.19, GP 1.26 (persistence 9.0) | - | SVM 2.26 | fails (capacity rises with load change) |
Caveats: Hot mixes 43 C with 4 A; mixed group shares one test schedule (models learn the protocol); cold cells start
near 1.4 Ah (use EOL 80 % of initial); B0029 low first cycle biased baselines/Level 2 before v5.11.

## 5a. Reference group with v5.11 (5 batteries: B0005, B0006, B0007, B0018, B0036; origin 30 %; all 25 models)
- Forecast: NO model beats fleet shape (4.34 +- 1.80 SOH pts). DL MLP 4.47, GP 4.99 (GP 1.25 on B0005 but 7.8 on
  B0006, 9.1 on B0036: the old "GP wins" was the one-battery anecdote), PINN 6.43 (4 batteries, B0036 missing).
- Measured (SOH now; compare only with measured models): GRU 2.05 +- 0.98, most stable (worst 3.17); MLP 3.16,
  Transformer 3.22, Bayesian Ridge 3.49. Trees fail on B0036 (20-29 pts, bias -20 to -28).
- B0036 (other campaign, same nominal conditions) fades more slowly: all forecasts predict too much fade on it.
  Without it (sensitivity only): MLP 3.73, GP 3.96 vs fleet shape 4.22, still not clearly better.
- EOL scored on 3 batteries only (B0007, B0036 never reach EOL). Trend/persistence bands far too narrow (9 %, 5 %).
- First v5.12 Reference run was interrupted after 6 forecasts (B0005, 3 of 7 ML models): not usable; v5.12.2 now
  refuses to pick from an incomplete level and says so.

## 5b. Hot group, run before v5.12 (no CV column; B0029-B0032, 43 C; origin 30 %; all 25 models, complete)
- Each battery has only ~40 cycles: forecast from cycle 12, 28 test cycles, EOL never reached (no EOL scoring).
- Nearly every model beats every baseline on all 4 batteries (fleet shape 3.21, trend 4.50, persistence 5.09).
  Forecast: Decision Tree 0.95, PINN 0.97, Bayesian Ridge 1.09, GP 1.16, RF 1.20, MLP 1.22 (top ones overlap within
  their CIs: no single winner). Measured: Bayesian Ridge 0.87, GP 0.88, MLP 1.09.
- Physics vs ablation NOT robust (corrected after the v5.13 rerun, see 5d): first run PINN 0.97 vs ablation 1.50
  (4/4), rerun ablation 1.20 (PINN better on 2/4 only); ablation B0032 1.72 -> 0.72 between identical runs.
- Not comparable to Reference: horizon 28 cycles vs ~120, so the Hot task is much easier. Forecast bands too
  narrow (coverage 65-84 %). Fleet shape predicts too much fade on all 4, trend too little.

## 5c. Reference with v5.12 (complete: 5 batteries, 25 models, CV picks) and the CV leak found in it
- Per-model test RMSEs reproduce the v5.11 run (pipeline deterministic) and stay VALID.
- CV picks / star table INVALID: Level 2 (ML) CV could hold out the predicted battery itself and score it on its
  test cycles (fixed in v5.13.0, test proves CV no longer changes when the target's future is corrupted). Also
  affected before v5.13: the Models-view star and self-optimisation for Level 2, and Level 2 forecast bands
  (coverage). DL and PINN CV were clean. Measured-mode CV was also a harder task than the test (held-out battery
  had no own cycles in the fit; CV 12-22 vs test ~3); v5.13 keeps its cycles up to the same share, like the test.
- Even with the leak, the star was poor at Level 2 (best on test in 1 of 5 folds, mean rank 3.2/7 forecast and
  4.4/7 measured); Level 3 better (mean rank 1.6/3 and 1.8/3). Must be re-measured with v5.13.

## 5d. Hot with v5.13 (complete: 4 batteries, 25 models, CV picks)
- 81/100 forecasts identical to the earlier Hot run. Level 2 forecast mode unchanged: at origin cycle 12 the
  target has no own rows, so the CV leak could not occur in Hot forecast mode. Measured-mode CV now on the test
  scale (Bayesian Ridge measured CV 0.81 vs test 0.87); measured bands changed as expected.
- Honest scoreboard (best by CV): ML measured 0.87, PINN 0.97, ML forecast 1.16, DL measured 1.23, DL forecast 1.29;
  all better than fleet shape (3.21) on all 4 batteries.
- Star: ML measured best on test 3/4 (mean rank 1.5/7, fixed by v5.13); DL forecast 3/4 (1.5/3); DL measured 2/4
  (1.75/3); ML forecast 0/4 (3.5/7) but the 7 ML forecast models are within ~0.3 pts, so ranks are mostly noise.
- Run-to-run variation: PINN, its ablation, XGBoost and GP differ slightly between identical runs (fixed seeds;
  most likely multithreaded floating point amplified by long training); ablation up to 1.0 pt. Any PINN vs
  ablation claim needs several seeds.

## 5e. Cold with v5.13 (5 batteries: B0041, B0045-B0048; 4 C; origin 30 %; 25 models, CV picks)
- B0041 belongs to Cold: confirmed with v5.14.1 (data as recorded, membership from recorded conditions), so its
  recorded discharges are at a single load level (the NASA README describes 4 A and 1 A across the B0041-B0044
  set). The earlier guess that the Reference-condition toggle put it there was wrong.
- Group is heterogeneous: B0045-48 have 4 different cut-off voltages (2.0/2.2/2.5/2.7 V); B0041 is another campaign
  with mixed 4 A / 1 A discharges. Biases alternate in sign across batteries (e.g. B0047 too high, B0048 too low).
- Honest scoreboard: no CV-picked model beats fleet shape (3.20): ML forecast 2nd by CV 3.19, best by CV 3.75,
  DL forecast 3.89; measured picks (4.26, 4.29) worse than fleet shape on all 5. Persistence 10.7, trend 11.7.
- Star fails: ML forecast 0/5 best on test, mean rank 4.8/7 (worse than random); others ~ random. GP forecast is best
  on test (2.77; without B0041 better than fleet shape on 4/4) but CV never picks it: an after-the-fact finding.
- PINN 8.39 vs ablation 5.09: physics hurts at 4 C (B0045 16.1, bias -13.7: predicts far too much fade).
- DL GRU / Transformer forecast failed on B0041 (not enough training windows): scored on 4 batteries.
- EOL crossed before the origin on every battery (cycles 2-44): no EOL scoring. Check which EOL setting was used.
- Across groups so far: learned models beat fleet shape only in Hot (short 28-cycle horizon, homogeneous group).

- Cold rerun with v5.14.1 (stamped: data as recorded, EOL 1.40 Ah): identical to the v5.13 run, so the Cold SOH
  results in 5e are confirmed. EOL not scored for Cold: with the NASA 1.40 Ah threshold, cold cells cross EOL by
  cycles 2-44, before the forecast origin. Decision (user): keep 1.40 Ah, no Cold EOL rerun; report it as a limit.

## 5f. Mixed conditions with v5.14.1 (exploratory; reference condition; B0038-B0040, B0042-B0044; 30 %)
- Two campaigns: B0038-40 (horizon 15 cycles, almost no fade at reference condition: persistence 0.2-0.9, so
  differences there are at noise level) and B0042-44 (4 C; horizon 29 cycles, 4-7 pts fade: the informative part).
  EOL never reached. DL GRU / Transformer forecast failed on all 6 (too few windows): DL forecast = MLP only.
- Best baseline = linear trend (0.79); fleet shape fails (3.08, worse than trend on all 6): the group is not a fleet.
- PINN best: 0.44 (better than trend on 5/6, not significant); ablation 1.62 (PINN better on 5/6; large ablation
  errors on B0042/B0043). Seed caveat from Hot applies. Contrast: PINN failed in Cold (B0045-48, also 4 C).
- B0042-44 only: PINN 0.70, ML Bayesian Ridge 0.77, DL GRU measured 0.80, trend 1.18, fleet shape 4.83.
- Star fails: ML forecast mean rank 6.33/7 (CV picked the worst). CV errors 3-10x the test errors and nearly equal
  across models (fold fits use only 2-3 batteries, often from the other campaign), so CV cannot rank here.

## 5g. Pulsed load with v5.14.1 (exploratory; as recorded; B0025-B0028; origin 30 %, not 40-50 %)
- Short records: ~28 cycles per battery, forecast from cycle 10-11, horizon 17-18 cycles, little fade (persistence
  1.85); EOL never reached. DL GRU / Transformer forecast failed on all 4 (too few windows): DL forecast = MLP only.
- Nothing beats fleet shape (0.98): ML measured best by CV 0.94 and ML forecast best by CV 1.03 (3/4 each, no clear
  difference); PINN 1.11 vs ablation 1.15 (no physics effect). Every model within ~0.3 pts: noise level.
- Star: ML measured 3/4 best on test (rank 2.25/7), ML forecast 2/4 (3.5/7). DL forecast 1/1 is trivial (one model).

## 5h. Frozen models (best mean test RMSE, all batteries scored)
| Group | Forecast | Measured (SOH now) |
|---|---|---|
| Reference | DL MLP 4.47 (GP 4.99) | DL GRU 2.04 (DL MLP 3.16) |
| Hot | ML Decision Tree 0.95 (PINN 0.97) | ML Bayesian Ridge 0.87 (ML GP 0.88) |
| Cold | ML Gaussian Process 2.77 (DT 3.17) | DL Transformer 3.86 (DL GRU 4.06) |
| Mixed* | PINN 0.44 (ML BR 0.92) | DL Transformer 0.79 (DL GRU 0.80) |
| Pulsed* | ML Gaussian Process 0.95 (XGBoost 0.99) | ML GP 0.94 (DL Transformer 1.00) |
Runner-up in brackets. Near-ties (<0.05 pts): Hot forecast, Hot measured, Mixed measured, Pulsed forecast.

## 5i. Frozen-model update study, Hot (v5.15.1; DT forecast, BR SOH-now; 4 batteries; next 20 cycles; from cycle 12)
- Forecast RMSE (mean of 4): no anchor 0.79-0.80 (retrain 5-20) vs 0.92 (never); measured-capacity anchor 0.84-0.89
  (no better than none); field mode (SOH-now anchor) 0.93-1.06, SOH-now error 0.87. Spread across batteries
  0.1-0.7, so differences of <0.15 pts are noise. Retraining helps mainly B0031 (1.96 -> 1.12, no anchor).
- Short records (~40 cycles, ~27 forecast cycles): retrain every 50 = never (one fit). EOL never reached.
- Anchor uses the single latest measured cycle, while SOH-now elsewhere is the median of the last 3 clean cycles
  (v5.11 decision): a noisy cycle shifts the whole forecast. Candidate fix before the other groups.

## 5j. Frozen-model update study, Reference COMPLETE (v5.15.1; MLP forecast, GRU SOH-now; next 20 cycles)
- 5/5 batteries (B0005-B0007 group run, B0018 and B0036 run one at a time; no failures).
- Mean of 5 (forecast RMSE SOH pts; EOL |error| cycles on B0005, B0006, B0018):
  never updated 4.35 (EOL 20); retrain only every 5-10: 1.54-1.56 (12), better than never on 5/5; every 50: 2.02;
  measured-capacity anchor, never retrained: 1.45 (9), better than retrain-only every 10 on 3/5; retraining on
  top adds nothing (1.38-1.44); field mode (GRU anchor, SOH-now error 2.04): 2.35-2.50 (14-19), better than never
  updated on 5/5.
- B0036 (other campaign, slower fade): never updated 6.67, any update 1.0-1.5; GRU SOH-now error 0.98 there.
- Answer for Reference: the twin must update; cheapest effective update = anchor to measured capacity, no
  retraining; without anchor retrain every 10 cycles; field mode: train once, anchor to the GRU estimate.

## 5k. Physics notes for the write-up (started 2026-10-19)
- Level 4 PINN inputs per battery are constants: x_T (Arrhenius of median discharge cell temperature), x_C (degrees
  below 15 C / 10), x_I (log DISCHARGE current). Cut-off voltage is not an input. Within one condition group all
  batteries share x_T, x_C, x_I, so the condition sensitivities (b_T, b_C, a_I, a_L) are not identifiable from
  the group: physics acts as a fixed-shape prior (SEI self-limiting, decelerating; plating with LAM coupling,
  accelerating), scaled per battery by multipliers learned from its first 30 %.
- Plating is driven by charge (NASA: CC 1.5 A at 4 C), but the PINN's current term uses discharge current.
- Cold per battery (PINN / ablation RMSE, PINN bias): B0045 (2.0 V) 16.09 / 2.46, -13.65; B0046 (2.2 V) 3.55 / 5.99,
  +1.32; B0047 (2.5 V) 9.02 / 3.68, -6.01; B0048 (2.7 V) 4.51 / 4.46, -1.25; B0041 (2.0 V, other campaign) 8.81 /
  8.86. No monotonic trend with cut-off voltage; physics harms through too much fade (B0045, B0047); B0041's error
  is not physics (ablation equal). Hypothesis to check: early low-capacity runs (NASA README) read as plating, then
  extrapolated by the accelerating plating-LAM term. Check in the Models view: PINN mechanism shares for B0045.
- Mixed B0042-44 are also 4 C (same campaign as B0041) yet the PINN wins there (reference condition, short horizon).

## 5l. Frozen-model update study, Cold: B0045-B0048 (B0041 missing) (v5.15.1; GP forecast, Transformer SOH-now; next 20 cycles)
- 72 cycles, forecasts from cycle ~22, 48 forecast cycles; EOL at cycle 2 (1.40 Ah): no EOL scoring; every 50 = never.
- No anchor: never 1.33 (best), retraining 1.42-1.50 (slightly worse). Measured-capacity anchor 1.78-1.88 (worse
  than no anchor). Field mode 3.93-4.52: Transformer SOH-now error 4.96 on B0045.
- B0046 (47 forecast cycles; EOL at 17, before the start): no anchor never 2.97, retrain every 5: 2.62;
  measured-capacity anchor 2.73-3.10 (no better than none); field 3.90-4.36 (SOH-now error 3.64).
- B0047 (run with 10 and never only; 47 forecast cycles; EOL at 10): no anchor 3.53 never / 3.16 every 10;
  measured-capacity anchor 3.22 / 2.59 (best); field 3.11 / 3.11 (SOH-now error 3.71; field retraining
  changes almost nothing, the anchor dominates).
- B0048 (10 and never; EOL at 16): no anchor 3.78 / 2.85; capacity 3.11 / 2.88; field 4.82 / 4.27 (SOH-now 3.35).
- Mean of B0045-B0048 (settings 10, never): none 2.81 / 2.67; capacity 2.57 / 2.75; field 4.04 / 4.07.
  Differences of 0.1-0.3 against a battery spread of ~1 pt: in Cold the update strategy barely matters; field
  mode is clearly worse (Transformer SOH-now error 3.6-5.0).
- Single-cycle capacity anchor hurts on a noisy battery (NASA README: low-capacity runs at 4 C): first direct
  evidence for the 3-cycle-median anchor (open question; feature freeze).

## 5m. Frozen-model update study, Pulsed load COMPLETE (v5.15.1; GP forecast, GP SOH-now; 10 and never)
- 4/4 batteries (B0025-B0028), 16-17 forecast cycles each, EOL never reached, no failures.
- Mean of 4 (every 10 / never): none 0.79 / 0.89; measured-capacity anchor 0.81 / 0.82; field 0.85 / 0.85
  (GP SOH-now error 0.54-1.51). All within ~0.1 pts: update strategy irrelevant at this horizon.
- Per battery the best anchor differs (capacity worse on B0026, field worse on B0028): no single best setting.

## 5n. Frozen-model update study, Mixed conditions: not done (decision in section 4)
- Two runs with the toggle off were invalid (scored capacity jumps, not ageing); v5.15.2 fixes the data rule, but
  the study is dropped for Mixed by decision.

## 5o. Attribution check on synthetic data (scripts/pinn_seeds.py, 2 seeds, demo Reference)
- Synthetic fade is a lumped linear throughput law (no mechanisms), yet the PINN attributes ~77 % to LAM, stable
  across seeds (SD 0-10 %): the attribution follows the curve shape. Stable is not correct: EIS validation needed.

## 5p. PINN seeds study (scripts/pinn_seeds.py, v5.15.1, seeds 0-4, Hot/Cold/Mixed; two runs identical)
- Accuracy (mean over seeds, PINN vs ablation): Hot 0.95 vs 1.43, PINN better on 4/4 batteries (16/20 runs) and
  far more stable (seed SD 0.01-0.06 vs 0.26-0.86): the Hot physics benefit is now REAL. Mixed 0.43 vs 1.56 (5/6;
  B0042-44 large gains). Cold 7.92 vs 4.94: physics harms B0045 (15.84 vs 2.31) and B0047 (8.61 vs 3.61).
- Attribution (share at last cycle, mean +- SD over seeds): plating and LAM trade off between seeds (e.g. B0029
  plating 26+-39 / LAM 70+-39; B0042-44 plating ~80+-40): plating vs LAM NOT identifiable. Stable cases are LAM-
  dominated (B0031 86 %, B0045 91 %, B0047 83 %) - the shape LAM gets on synthetic linear fade too. SEI share small
  (Hot 4-8 %) though 43 C should favour SEI; Cold cells mostly LAM (plating 2-4 %) though 4 C should favour plating.
- Parameters: SEI activation energy 22.6-22.7 kJ/mol with SD ~0.1 in Hot and Mixed, cold sensitivity 1.000 +- 0
  in Hot and Mixed: they stay at their starting values (inputs constant within a group -> not learnable);
  LAM current sensitivity in Cold -0.53 +- 0.59 (sign undetermined).
- Conclusion: the PINN is a good, stable forecaster in Hot and Mixed, but its mechanism attribution cannot be
  trusted yet. Next (physics chapter): train across groups so temperature/current vary; add independent signals
  (EIS Re/Rct, incremental-capacity dQ/dV peak shifts = LLI vs peak area loss = LAM); plating driven by charge.

## 5q. dQ/dV method (scripts/dqdv_features.py, 2026-10-21)
- Per clean cycle, discharge and CC charge: Q(V) from current integration, IR-corrected (V +- I*R_dc), smoothed,
  dQ/dV on a 5 mV grid; main peaks (position, height, area); fit Q_n(V) = a * Q_ref(V - delta) vs first 3 cycles.
- LLI-like shift = (charge + discharge shift)/2, polarisation = (charge - discharge shift)/2 (polarisation moves the
  curves in opposite directions, lithium loss in the same direction).
- Synthetic check (pure capacity scaling + resistance growth): discharge a = SOH within 0.01, LLI-like shift 0-9 mV,
  polarisation 54-101 mV. tests/test_dqdv.py. Limit: no half-cell references -> indicators, not exact LLI/LAM_PE/NE.

## 5r. PINN v2 in the app (v5.16.0)
- Section "Level 4 · PINN v2 · Which degradation mechanisms explain this battery's capacity loss?" (Models view).
- Synthetic check (pure capacity scaling): v2 LAM 94-95 % +- 0.4 (v1 75 % +- 6, plating 16 % +- 5); reproduces the
  dQ/dV scale (0.858 vs 0.854); SEI Ea learned 25.6-26.0 kJ/mol across seeds (v1: stuck at its start).

## 5s. PINN v2 on NASA (v5.16.0; all groups, 3 seeds): attribution INVALID (method error)
- B0005, B0029, B0045, B0047, B0025: v2 gives 98-100 % LAM everywhere (v1 74-82 % LAM, plating 12-21 % +- 4-10).
- Cause: the global dQ/dV scale a tracks capacity (SOH 0.701 vs a 0.639; 0.892/0.884; 0.697/0.707; 0.749/0.729;
  0.969/0.961) whatever the mechanism; the rule "a = 1 - LAM" forced all loss into LAM. Synthetic check could not
  catch it (uniformly scaled OCV, no electrodes). Also: Re link does not bound SEI (g_e free; B0005/B0047 Re +31 %
  fitted with 2 % SEI). Observed LLI-like shift small and of mixed sign (-21 to +22 mV).
- Valid: dQ/dV measurements, v2 physics/constraints/cross-group design, PINN v1 and Mission 2.
- Fix: remove "scale = 1 - LAM", label v2 attribution not validated; then peak-resolved dQ/dV features (LCO/graphite
  staging peaks), validated on a PyBaMM LCO/graphite benchmark with known LLI, LAM_NE, LAM_PE before use; check
  peak resolution of real NASA curves at 1-2 A (B0005, B0045 images).

## 5t. Fix and benchmark start (v5.16.1, 2026-10-21)
- dQ/dV rules off in PINN v2 (dq_weight = 0); section experimental, "not yet validated" notice; test that dQ/dV
  values no longer change the attribution.
- NASA dQ/dV resolution (Diagnostics chart, raw voltage): B0005 (24 C, 2 A) one main peak ~3.69 V that halves and
  moves ~60 mV lower (mostly polarisation, Re +31 %) plus a shoulder ~3.95-4.0 V; B0045 (4 C, 1 A) weak broad noisy
  peak (~2 Ah/V) halving in 14 cycles: dQ/dV barely usable in the cold.
- PyBaMM 26.10 with Ramadass2004 (LiCoO2/graphite) works, ~1 s per discharge. First test (SPMe, 1C, 10 % imposed):
  LLI and graphite LAM give the same capacity loss (0.936/0.937) and nearly the same main peak -> not separable
  at 1C from the main peak. LiCoO2 LAM gave capacity 1.044: artefact of not re-balancing the electrodes.

## 5u. PyBaMM benchmark, work in progress (scripts/pybamm_benchmark.py, 2026-10-21)
- Ramadass2004 (LiCoO2/graphite), SPMe, NASA protocol (rest, CC 0.75C to 4.2 V, CV to C/100, rest, discharge at
  0.5/1/2C to 2.7 V), cells balanced by PyBaMM at 0 % SOC after imposing modes on electrode size (LAM, lithium kept)
  and cyclable lithium (LLI); Arrhenius factors added (Ramadass has none). ~1 s per cell.
- Limits found: LLI only up to ~6 % (LiCoO2 OCP table); 4 C far too pessimistic (fresh 1C 0.12 Ah) -> calibrate
  activation energies to the fresh NASA cold-cell capacity fraction; one mixed cell ended at 0.32 Ah (check);
  24 C missing from the quick identifiability table (bug); LAM of lithiated material not yet modelled.
- Preliminary single cells at 1C, 24 C (fresh 1.797 Ah): LLI 4 % -> capacity -5.5 %, main peak height -7 %,
  +15 mV, LLI-like shift +5 mV; graphite LAM 10 % -> capacity -3 %, peak unchanged (nearly invisible); LiCoO2 LAM
  10 % -> capacity -11 %, peak height -7 %, -5 mV, LLI-like shift +11 mV. NOT yet results.

## 5v. Mechanism-recovery test, step 1 (scripts/pinn_recovery.py, quick: H1, C1; 2 seeds; 2026-10-21)
- Simulated batteries with KNOWN split, generated with PINN v2's own equation forms (well-specified case), 11 cells
  at 4/24/43 C, 1-4 A, charge 0.75-1.5 A, cut-off 2.0-2.7 V; true splits diverse (cold: plating 47-83 %, hot: SEI
  82-94 %, deep discharge: LAM up to 52 %); losses 10-20 %. PINN sees capacity, R_dc, Re and conditions only.
- Result: H1 truth SEI/pl/LAM 89/6/6 -> v1 41/4/55, v2 21/5/73; C1 truth 15/83/2 -> v1 11/10/80, v2 6/29/65.
  Mean share error v1 42, v2 45 points; dominant mechanism wrong in all cases; seed SD small (stable but wrong).
- PINN (v1 and v2) cannot recover the mechanism split even in the favourable case. Next: diagnostic, training
  loss of the TRUE split vs the found split -> optimisation failure (fixable) or non-identifiability from capacity,
  resistance and Re (a data limit). Decide before PyBaMM step 2.

## 5w. Diagnostic: pure physics fit recovers the mechanisms (pinn_recovery.py --physics, 2026-10-21)
- Same simulated batteries; ONE shared parameter set (11 params), rate laws solved exactly in throughput, fitted to
  capacity, R_dc and Re of all 11 batteries from 30 random starts (least squares).
- Result: mean share error 1.1 pts, dominant mechanism correct 11/11, all 30 starts identical (spread 0.0),
  Ea 47.9 kJ/mol (truth 50), g 2.98 (3.0), g_e 4.54 (5.0). With links fixed: 0.5 pts. Cost at the true
  parameters 2079 vs best fit 1904 (noise-level difference).
- Conclusion: mechanisms ARE identifiable from capacity + resistance + Re across conditions, with shared physics.
  PINN v1/v2 failure (42-45 pts) is a design problem: network-drawn states and per-battery multipliers let each
  battery choose its split and ignore the condition patterns. Fixable.
- In the app since v5.17.0: Models & forecasting -> "Recovery test" section (engine section 47).
- DECISION/NEXT: redesign PINN v2 -> shared physics parameters, mechanisms from integrated rate laws (hard), small
  bounded network correction; rerun recovery test on the PINN; then PyBaMM step 2 (misspecified physics).

## 5x. Shared physics on NASA (v5.18.0; Re held out; 30 starts; 20 batteries)
- Unique (spread 0.01 pts). Group means SEI/plating/LAM (fit error): Hot 55/39/6 (1.6), Pulsed 52/42/6 (2.7),
  Reference 34/60/7 (6.8), High current 34/56/10 (7.2), Cold 13/85/2 (7.6). Ordering right (SEI hot, plating cold).
- NOT credible yet: plating dominates at 30-40 C (catch-all for steady fade); charge current is 1.5 A on EVERY NASA
  battery -> charge-current sensitivity unlearnable (ended at bound 5.0); Ea at 20 (bound), a_I 0, c 0; fit errors
  9.5-13 pts on B0045, B0041, B0006, B0036, B0033 (knees, regeneration not modelled); held-out Re: Hot Re flat
  (-1.8 to +2.1 %) while SEI 55 % (contradiction); positive rank correlations elsewhere are weak evidence.
- Shared physics v2 built (v5.19.0): plating zero above 15 C, no charge-current term, shared knee term, fit SEI to
  Re (check: load-step resistance) or the reverse; card lists parameters at a limit. To be run on NASA.

## 5y. Shared physics v2 on NASA (v5.19.0; Re in fit, R_dc held out; 30 starts; 20 batteries)
- Group means SEI/plating/LAM (fit error): Hot 56/0/44 (1.6), Pulsed 53/0/47 (2.8; only 2-4 % lost), Reference
  36/0/65 (7.1), High current 34/0/66 (7.2), Cold 12/71/17 (7.0). Plating only in Cold; unique (spread <= 0.09).
- Fit as good as v1 on almost every battery (v1 freedom bought nothing). Held-out R_dc check: rank correlation
  0.91-0.99 Hot, 0.57-0.97 Reference, 0.72-0.94 High current, 0.83-0.97 Cold; Pulsed negative (no signal).
- At a limit: Ea (20 floor), b_C (0: all cold cells at ~9 C, one temperature), a_I (0: discharge current tied to
  groups), beta0/beta_V (shared knee unused). Knee batteries still 9.6-12.1 pts (B0006, B0041, B0045, B0033, B0036).
- Contradiction: Hot Re flat (-1.8 to +2.1 %) while SEI 56 % and R_dc rises (R_dc agrees with SEI, Re does not).
- Thesis: physically consistent, unique, held-out-supported split; NASA design limits identify which sensitivities
  cannot be learned (one cold temperature, one charge current, current confounded with temperature).
- NEXT: knee by LAM-triggered plating (graphite margin) instead of a shared knee; SEI with ambient instead of
  discharge temperature (does Ea leave the floor?); score as forecaster in the accuracy assessment.

## 5z. Hybrid (v5.20.0, 2026-10-22)
- Hybrid = shared physics v2 (frozen; the split) + bounded network on the residual ("not explained by the physics").
- Validation on a known split with unmodelled knees: letting the network shape the physics always worsened the split
  (3.9-6.8 vs 3.4 share points); frozen physics keeps 3.43 and improves fit (knee batteries 0.73 -> 0.33).
  Lesson: the network improves the fit, not the explanation. Its correction localises knees only roughly (+1.3 to
  +1.7 on knee batteries, but -1.3 on R4): a hint, not a proof.
- App now (Models view, Level 4): original PINN, Shared physics (no neural network), Hybrid. PINN v2 and Recovery
  test sections removed from the app (engine/tests/scripts keep them).

## 5aa. Hybrid on NASA (v5.20.0; Re; 30 starts; penalty 0.3) and fixes (v5.20.1)
- Mean fit error physics alone 5.12 -> hybrid 2.27 SOH points. Hot (network <= 1.4 pts RMS) and Pulsed: physics
  explains almost all.
- Gap 1, room temperature late in life: B0005, B0006, B0018 need 20-25 % of their loss from the network (physics
  flattens: SEI decelerates, LAM steady); B0007 fits (1.5 %).
- Gap 2, cold early in life: B0041, B0045 lose ~10 % in the first ~15 cycles; network supplies 29-46 %.
- Compromise: physics over-predicts B0033, B0036, B0046-48 by 6-8 points (shared parameters pulled by the gaps).
- Flaws fixed in v5.20.1: correction now zero at the start of life (B0045 started at ~8 %); bound 8 -> 25 points (the
  cap hid ~half of B0006's ~20-point gap). Rerun the hybrid on NASA with v5.20.1.
- These gaps define physics version 3: an accelerating late-life mechanism at room temperature, and a fast early loss
  in the cold.

## 5ab. Joint training test and physics forecasters (v5.21.0, 2026-10-22)
- Joint hybrid with guardrails never beats frozen on the split (frozen 3.43; joint 4.17-5.09 share points; fit worse
  as guardrails tighten). DECISION: frozen hybrid = explanation model.
- Physics forecasters in the assessment (section 50): causal (tested), start at SOH now. Demo data: 5.9-6.4 SOH
  points vs fleet shape 1.04 (demo fade is linear, the physics' SEI decelerates: forecasts too slow; mechanics
  verified). NASA run needed.

## 5ac. Physics-based forecasters on NASA (v5.21.0; origin 30 %; Reference 5, Hot 4, Cold 5 batteries)
- RMSE (SOH pts) Reference / Hot / Cold: fleet shape 4.34 / 3.21 / 3.20; Mission 2 frozen 4.47 (MLP) / 0.95 (DT) /
  2.77 (GP); shared physics v2 8.34 / 2.99 / 4.40; frozen hybrid 6.43 / 3.35 / 4.12; joint hybrid 7.38 / 3.02 / 3.92.
  None clearly beats the baseline; all clearly behind the Mission 2 models.
- Systematic positive bias (too optimistic): B0005 +11, B0006 +6 to +12, Hot ~+3, Cold +2 to +5. Same gap as the
  hybrid's explanation: SEI decelerates, nothing accelerates; the network cannot extrapolate the future acceleration.
- Frozen hybrid EOL error 12.5 cycles vs fleet shape 32 where scorable (Reference, 2 batteries): a hint only.
- DECISION PENDING (proposed): Option A physics v3 (non-decelerating SEI on cracking particles = learned growth
  exponent; and/or plating triggered by lost graphite at room temperature), retest explanation + forecasting;
  Option B safety net: one twin, forecasting by Mission 2 frozen models / ensemble, explanation + anomaly + what-if
  by the hybrid.

## 5ad. Physics version 3 built (v5.22.0, 2026-10-22; DECISION: option A)
- SEI exponent z learned [0.3, 1.5]; plating triggered by lost graphite at any temperature (+ cold-driven plating);
  LAM linear. Identifiable on its own laws (split 0.10 pts; z 0.86 vs 0.9; m_c 0.030; Ea 50.2). Reads any knee as
  triggered plating (7.5 vs 3.4 pts on the knee cohort whose knee has no mechanism).
- Demo: card flags v3's split as NOT unique (near-best starts differ by 10.9 pts): check uniqueness on NASA.

## 5ae. Physics v3 on NASA (v5.22.0)
- Explanation (Re, 30 starts): learned z = 1.33 (SEI accelerates); triggered plating unused (k_tr ~0, m_c at 0.3);
  at limits: Ea, b_C, k_lam, a_I, m_c. Group means SEI/plating/LAM: Hot 23/0/77, Reference 23/0/77, High current
  25/0/75, Pulsed 14/0/86, Cold 2/77/21. Unique within the version; mean fit 5.21 (v2 5.12).
- MODEL-FORM UNCERTAINTY: Hot SEI 56 % (v2) vs 23 % (v3) with similar fit -> warm SEI vs LAM split NOT determined by
  capacity + resistance + Re. Robust across versions: plating only and dominant in the cold (71-77 %); warm = SEI +
  LAM mix; NASA design limits.
- Forecasting (30 %): physics v3 8.14 / 3.37 / 4.37, hybrid v3 6.89 / 3.53 / 4.33 (Reference / Hot / Cold) vs fleet
  shape 4.34 / 3.21 / 3.20: no gain over v2; still optimistic (B0005 +11). Likely cause: no battery-specific
  information in the shared physics (B0005 ages faster than its group; fleet shape adapts per battery).
- PROPOSED: per-battery ageing-speed factor (same for all mechanisms, fitted on the battery's own history: cannot
  shift the split) -> rerun forecasting. If still not competitive: option B for Mission 3.

## 5af. Ageing-speed factor (v5.23.0, 2026-10-22) - last physics forecasting attempt (user: if not working -> B)
- kappa per forecast battery, fitted on its own history, same for all mechanisms. Demo: worse (changes speed, not
  shape). NASA run decides.

## 5ag. Ageing-speed factor on NASA (v5.23.0) -> DECISION: OPTION B
- RMSE Reference / Hot (run interrupted: 3 batteries, B0032 missing) / Cold: fleet shape 4.34 / 3.62 / 3.20; best
  physics with speed: hybrid v2 6.27 / physics v3 2.98 / physics v3 3.13 (without speed 6.43 / 3.37 / 4.37);
  Mission 2 frozen 4.47 / 0.95 / 2.77. Speed helps Cold (ties baseline), slightly Hot, not Reference (B0005 +11-13).
- Physics never beats the Mission 2 models -> per the agreed rule, OPTION B.
- Phase conclusions: (1) data-driven Mission 2 models are the best forecasters; physics forecasts systematically
  optimistic (SEI laws miss late-life fade; learned exponent, triggered plating, speed factor did not fix the shape);
  (2) shared physics explains (recovery test), robust: plating only and dominant in the cold, warm = SEI + LAM mix,
  SEI/LAM proportion model-form dependent; (3) network improves fit not explanation; joint training worsens split.

## MISSION 3 (option B): one twin, each capability from the model proven best at it
- Forecast SOH/EOL: Mission 2 frozen model per group; accuracy over life: re-anchoring to measured capacity;
  explanation: hybrid v2 split + model-form uncertainty; anomaly detection: hybrid unexplained part + deviation from
  the forecast; what-if: shared physics sensitivities only for conditions NASA varied.
- FIRST STEP proposed: anomaly detector, replayed cycle by cycle; test = flags knee batteries (B0006, B0045, B0041)
  before their accelerated fade, few false alarms elsewhere.

## 5ah. Mission 3 · anomaly detector (v5.24.0, 2026-10-22)
- Own-trend CUSUM, standardised by the trend's prediction error, h = 8 (validated: 3/3 knees, 0/8 false alarms,
  alarm at 0.9-1.5 SOH points of knee). Lessons: fleet physics as expectation -> false alarms (it is slightly wrong
  for healthy batteries); standardising by cycle noise instead of prediction error -> false alarms.
- In the app: Operations & control, top section. Not yet run on NASA.

## 5ai. Anomaly scan on NASA (v5.24.0, threshold 8)
- 8/20 batteries alarm: B0005 (cycle 44), B0006 (45, 4 episodes), B0007 (41), B0018 (23), B0033 (56), B0034 (71),
  B0036 (65, 4 episodes), B0046 (32). Cause: NASA's rest periods -> capacity regeneration then fast relapse
  (sawtooth); the own-trend CUSUM reacts to each relapse (B0007, B0036 alarm though explained by the physics). The
  validation cohort had no regeneration.
- B0041, B0045 not flagged: their fast loss is in the first ~15 cycles, before the detector has its own history.
- KEY FINDING: kappa (ageing speed vs the fleet physics, leave-one-out) is highest for exactly the batteries the
  hybrid could not explain: B0041 3.00 (limit), B0045 2.65, B0006 2.61, B0018 1.75, B0005 1.46; all others
  0.30-1.22. Two independent analyses agree.
- PROPOSED: two indicators: (1) faster than the fleet = kappa with an alarm rule (e.g. > 1.8), validated on
  simulated batteries with a known faster speed; (2) faster than its own past = robust slope change (last ~20 vs
  previous ~20 cycles, Theil-Sen), robust to regeneration; validate both on simulated batteries WITH regeneration.

## 5aj. Two-indicator anomaly detector (v5.25.0, 2026-10-22)
- FLEET (kappa > 1.8 for 5 cycles) + OWN (Theil-Sen slope, 30 vs previous 30 cycles, z > 3 for 3 cycles).
- Validation WITH regeneration: 11/11 correct (knees by OWN after 27-35 cycles; 2.5x-fast batteries by FLEET at
  cycle 12; healthy quiet). v5.24 detector: 8/8 false alarms on the same cohort. Not yet run on NASA.

## 5ak. Two-indicator scan on NASA (v5.25.0; kappa 1.8, z 3.0)
- FLEET: B0045 cycle 13, B0041 cycle 55 (record starts at 44: first possible), B0006 cycle 54 = the hybrid's
  unexplained batteries; B0018 (kappa 1.70) and B0005 (1.51) just below; FALSE: B0034 at cycle 13 (early kappa swing,
  ends 0.96). Pulsed batteries kappa 0.30-0.45 (age slower than the fleet physics).
- OWN: B0005, B0006, B0007 all at cycles 73-74 = one COMMON test event (tested together), not three faults; B0033
  130, B0036 90 (3 episodes). OWN needs ~70 cycles (2 x 30 + cleaning): never evaluated on Hot, Pulsed, Cold (<= 72).
- PROPOSED: FLEET min history 8 -> 12 cycles; OWN common-cause classification (>= 2 batteries of the same campaign
  within +-5 cycles -> common event), with a validation case. Then the integrated twin view.

## 5al. Live anomaly detection and common causes (v5.26.0, 2026-10-23)
- FLEET min history 12; common-cause classification (validated 11/11); plain-language event log; live panel in the
  Live twin; page order: Models -> Model updating (new page) -> Live twin -> Operations -> Study results.

## 5am. Speed and live detection (v5.27.0, 2026-10-23)
- Live twin retrain default never; result backup on disk (results/cache/, content-keyed, engine-versioned; zip export
  in the sidebar; commit to GitHub for a warm start). Live twin 26 s -> 2 s in a new session.
- Operations: live detection replay with per-cycle verdict colours and the growing "What happened" log.

## 5an. End of life and removed cycles in the detection (v5.28.0, 2026-10-23)
- NASA v5.27 scan received. B0005 missing cycles 20-23, 31-34, 48-51, 90-93, 120-123, 151-153 = clean_capacity drops
  each regeneration cycle after a rest + the 2 transient cycles after it (and invalid ones): now shown hollow.
- EOL event and "Past end of life" status; name collision with the existing eol_crossing caught by tests, renamed.

## 5ao. PINN with all inputs (v5.29.0, 2026-10-23)
- User decision: the PINN must get all battery parameters; physics-forecaster levels removed from the assessment.
- pinn:PINN-full = PINN v1 physics + conditions + measured state at the origin (19 features incl. discharge time);
  pinn:PINN-full-ablation = same without physics. Demo: 0.68 vs fleet 1.04 vs no-physics 1.26. Caveat stated to the
  user: physics is not guaranteed to win (helped Hot/Mixed, hurt Cold); more inputs with 4-6 batteries risk overfit.

## 5ap. PINN with all inputs on NASA (v5.29.0; origin 30 %; assessment CSVs Reference 5, Hot 4, Cold 5)
- RMSE (SOH pts) Reference / Hot / Cold: PINN all inputs 5.15 / 0.97 / 9.09; its no-physics twin 6.04 / 1.45 / 7.81;
  PINN v1 5.61 / 0.98 / 8.44; v1 no-physics 6.32 / 0.93 / 4.79; fleet shape 4.34 / 3.21 / 3.20; Mission 2 frozen
  4.47 (MLP) / 0.95 (DT) / 2.77 (GP).
- vs PINN v1: Reference 3 better, 1 worse (B0006 9.2 vs 2.9), 1 tie; Hot identical on every battery; Cold worse
  (B0045 unchanged 16.0, bias -13.6). Physics vs own twin: Hot helps (3/4); Reference 2-2; Cold hurts.
- The 19 extra inputs make the no-physics network WORSE (Hot 0.93 -> 1.45, Cold 4.79 -> 7.81; B0045 2.3 -> 11.7):
  constant per battery, so with 3-4 training batteries they act as a battery fingerprint (overfitting).
- Single-run caveat: v1 ablation 0.93 in Hot vs seed study 1.43 (5p); single ablation runs scatter up to ~0.9.
- PINN v1 reproduces within run-to-run noise (Hot 0.98 vs 0.97; Cold 8.44 vs 8.39; B0047 9.27 vs 9.02).
- DECISION: frozen forecasters unchanged (Reference MLP, Hot DT, Cold GP). Reportable negative result. Mixed/Pulsed
  runs not needed (optional: Mixed, the one group where PINN v1 won).

## 5aq. Integrated twin page (v5.30.0, 2026-10-27)
- New page "Integrated twin" (after Live twin): per battery, a "Today" cycle; status (Normal / Watch / Alarm / End of
  life reached, with reason); measured SOH coloured by the detector (removed cycles hollow); frozen-model forecast
  from today with band and end-of-life estimate; where the capacity went (hybrid v2, unexplained part grey); event log;
  explanation table v2 (main) vs v3 (check) with the model-form gap; what-if; CSV timeline (one snapshot per cycle).
- Settings shared with other pages (results reused): frozen model trained once, re-anchored to measured capacity;
  hybrid Re, 30 starts, penalty 0.3 (Models defaults); anomaly kappa 1.8, z 3.0. Fleet parts built once per session by
  a button and saved in the result backup.
- Watch = common test event, or early edge of the forecast band reaches end of life within the watch window (20).
- What-if guardrails (engine-enforced): only conditions that vary in the data, only tested values (T and I = group
  medians, cut-off per battery, rounded); sensitivity at a bound -> "limited" (shown, flagged); untested combination
  flagged; charge current never. Output as differences (physics alone optimistic on NASA).
- BUG FIXED: since results went through st.cache_data/backed(), "Fit the hybrid to the NASA batteries" (Models) failed
  ("Can't pickle local object _hybrid_net_fit.<locals>.<lambda>"); reproduced on v5.29.0. hybrid_portable() drops the
  network closure (unused by tables, trajectories, what-if); the hybrid is now cached and backed up.
- Engine change purely additive (section 55 + version line): PINN v1 and Mission 2 code byte-identical.
- Checks: 9 new tests; AppTest (Streamlit 1.66.1) on the page: build, cursor, reveal, what-if, re-render, batteries
  without frozen model / explanation; Models hybrid button; all pages; harness md_check 0 leaks.
- Expect on NASA: first build a few minutes (hybrid v2 + v3, 30 starts each, + anomaly scan); then seconds.

## 5ar. Integrated twin on NASA (v5.30.0; B0005, B0029, B0045; EOL 1.40 Ah) and fixes (v5.30.1)
- B0005 (Reference, frozen MLP): measured EOL at cycle 126. From cycle 65 on, the median EOL forecast stayed within
  ~5 cycles: mean |error| 4.9 cycles over cycles 65-125 (65-84: 4.3, bias -2.6; 85-104: 6.8; 105-125: 3.7). One
  battery: a hint, not a result (no claim on one battery); score the group before reporting.
- B0005 detector: regeneration at 24; common test event 74-100 (onset 74, as expected with B0006/B0007); past EOL
  from 126. Explanation at the end: SEI + LAM with ~19 % not explained (gap 1, room temperature late in life);
  v2 vs v3 split gap 11 points.
- B0029 (Hot, frozen DT): Normal throughout; EOL beyond the forecast horizon (cycle 56). v2 vs v3 split gap 34 points
  (model-form uncertainty, as in 5ae).
- B0045 (Cold, frozen GP): with 1.40 Ah the threshold is 158 % of its 0.89 Ah initial capacity -> "end of life" at
  cycle 2 and a meaningless EOL forecast (n+1). Explanation: plating 46 % + 37 % not explained (gap 2, early cold
  loss) as expected. FLEET alarm confirmed at cycle 21 here; STATUS 5ak lists 13 -> CHECK against the Operations
  event log (same scan settings; possibly 13 = first kappa exceedance, 21 = after 5 consecutive cycles with the
  invalid cycle 20 removed).
- Fixes v5.30.1: (1) Watch uses the MEDIAN forecast EOL (the MLP band is ~+-15 points one cycle ahead, so its early
  edge kept B0005 on Watch for 60 cycles); the early edge is reported in the reason; wide-band note shown. (2) EOL
  threshold >= 98 % of initial capacity -> end of life undefined for this battery, warning + button "Use 80 % of
  initial capacity". (3) No forecast-EOL marker after EOL is reached. (4) SOH axis from valid cycles (invalid
  0 % / 120 % points no longer squash the chart). (5) Forecast shown ~35 % of the record ahead of today (B0029 showed
  3 cycles). (6) Forecast line and band drawn as lines (Plotly added markers to short traces).
- The white vertical line at ~cycle 24 in the B0045 screenshot is most likely the hover cursor, not data.

## 5as. Integrated twin scored per group (v5.30.1; Reference 1.40 Ah, Hot/Cold 80 % of initial; 13 timelines)
- End-of-life forecast error (cycles, median forecast vs measured EOL, every cycle from forecast start to EOL):
  B0005 7.7 (bias +2.7) · B0006 11.5 (+9.5) · B0018 7.2 (+4.1) · B0041 2.9 (+2.0) · B0045 1.6 (-0.4) · B0046 5.5 (+5.2)
  · B0047 4.8 (+4.8) · B0048 8.3 (+7.2). Mean 6.2 cycles; 7 of 8 biased late (optimistic). Error shrinks toward EOL
  (Reference: >30 cycles before EOL 10.5-15.5; 10-30 before 3.2-10.3; last 10 2.1-4.3).
- Not scorable: B0007 (ends at 74.8-75.6 %, just above its threshold; forecast ~167 vs record end 168 -> Watch
  from 149 was right), Hot (40 cycles, SOH 87-92 %, EOL beyond horizon 56).
- First warning (Watch or Alarm) before EOL: B0005 74 (common event) / 101 (EOL forecast) -> 52 / 25 cycles ahead;
  B0006 Alarm 54 -> 56 ahead (its forecast was 29 cycles late at 56: the FLEET alarm caught what the forecast
  missed); B0018 21; Cold 7-22 (EOL at 80 % comes only 7-24 cycles after the forecast starts at 22); B0045 FLEET at
  21 before its forecast existed.
- Detector: common test event B0005/6/7 at 73-74 (as expected); regenerations labelled normal (B0005/6/7/18/31);
  no alarm in Hot; FLEET B0006 54 (= 5ak), B0041 59 (5ak: 55), B0045 21 (5ak: 13) -> CHECK in Operations whether 5ak
  lists the onset of kappa > threshold and the twin the confirmed alarm.
- Explanation at the last cycle (v2 / v3): Cold plating 72-77 % (B0046-48), B0045 42 % + 37 % not explained,
  B0041 29 % + 56 % not explained; Reference LAM dominant in both versions (v2 40-67 %, v3 49-79 %; gap 9-13 pts ->
  robust); Hot SEI vs LAM flips with the version (v2 SEI 46-57 %, v3 14-20 %; gap 32-37 pts -> not determined).

## 5at. Conservative end of life for Watch (v5.31.0)
- Why not the forecast band: each frozen model has its own band method with no common meaning (B0005 MLP band
  +-15 points one cycle ahead), so a band quantile would be noise.
- Method (causal, per battery, "self-updating"): at today n, compare every forecast made in the last 40 cycles with
  the SOH measured since (horizon >= 5 cycles): optimism rate = (forecast - measured) / cycles ahead. Conservative
  forecast = today's median minus the q-quantile of that rate x cycles ahead (rate floored at 0: never more
  optimistic than the median). Needs >= 10 comparisons, else median (stated in the status). Changes Watch and the
  reported conservative EOL only; the Mission 2 forecast is untouched.
- App: "Watch uses" = Median (default) / Conservative 50, 75, 90 %; conservative curve and end of life on the chart.
  Timeline CSV always carries EOL conservative q50/q75/q90 and the optimism rates, whatever the choice.
- Demo (Reference, EOL 80 %; synthetic, not evidence): EOL error median / q50 / q75 / q90: S002 6.8 / 4.2 / 4.5 / 5.0;
  S005 17.8 / 8.1 / 9.0 / 9.7; S008 5.8 / 3.7 / 5.1 / 6.4; bias falls from +6..+18 to about 0..+3.

## 6. Next steps (Mission 3, option B)
1. USER: upload v5.31.0 (twin_engine.py, app.py, tests/test_integrated_twin.py); CI green; reboot.
2. USER: re-download the 13 timelines (same EOL settings as 5as; "Watch uses" = Conservative 75 %; the CSV holds all
   three quantiles anyway). Judge: EOL error and bias per quantile vs median (5as), warning lead time, Watch never
   firing too early. Also: Operations event log for B0045 / B0041 first FLEET alarm.
3. Writing (after the twin is finished, user decision): Mission 2 chapter, physics-phase chapter, Mission 3 chapter.
4. Optional: Mixed assessment run for PINN all inputs; PyBaMM benchmark.

## 7. Open data checks
B0042-B0044 ambient (~22 C in data vs 4 C in README); only B0049-B0052 have crashed logging (B0053-B0056 belong to
cold); capacity is defined to 2.7 V even for cells discharged lower.

## 8. How to work (user preferences and release routine)
- Sending charts: camera icon on any chart -> PNG named chart-id__battery-or-group__title__date__version
  (v5.12.1); send the PNG with its file name unchanged.
- Plain explanations, honest negative results, no invented numbers; explain why results look as they do before
  changing anything. Ask before changing the visual identity (banner, colours).
- Release routine: edit -> engine tests (two halves, ~5 min each) -> headless app run of every view + two-render
  check + HTML/style parsing check -> bump ENGINE_VERSION/REQUIRED_ENGINE together -> README entry -> deliver
  twin_engine.py + app.py (+ tests, README).
- The sandbox resets between sessions: restore /home/claude/proj from the latest files; the headless Streamlit/Plotly
  stand-in harness is in the streamlit-science-app skill (scripts/), real Streamlit can be pip-installed (PyPI allowed)
  and tested with streamlit.testing.v1.AppTest.
