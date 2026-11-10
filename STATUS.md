# STATUS · handover for a new conversation · engine/app v5.37.0 · 2026-10-29

Read this first. It replaces every earlier STATUS.md. Full change history: README.md.

## 1. Project in one paragraph
Internship (Yokogawa, UM6P collaboration): a self-updating digital twin for the NASA Ames 18650 Li-ion ageing data
(LiCoO2/graphite, 34 cells, 4-44 C, 1-4 A). Three missions from the research plan: M1 health indicators and degradation
mechanisms; M2 self-updating twin (can degradation be predicted, which variables are informative, how often to update);
M3 integrated operation and maintenance. Deliverable: a Streamlit app (app.py) + a tested engine (twin_engine.py).

## 2. Files and versions (always deliver both .py files from the same release)
- twin_engine.py  ENGINE_VERSION = "5.42.0"   (all computation; no Streamlit)
- app.py          REQUIRED_ENGINE = "5.42.0"  (exact version guard + required-function check)
- tests/test_twin_engine.py: 80 tests (passing at v5.11.0; not rerun since, file not in the Project) +
  tests/test_cv_selection.py (5) + tests/test_frozen_live.py (10), passing at v5.15.0;
  tests/test_integrated_twin.py (12) + test_m3_maintenance.py (5) + test_m3_operation.py (6) + test_m3_lifecycle.py (3), v5.37.0 (27 tests)
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

## 5au. Conservative end of life on NASA (v5.31.0; 13 timelines; same EOL settings as 5as) -> NOT ADOPTED
- EOL error / bias (cycles; median | q50 | q75 | q90): B0005 7.7/+2.7 | 14.9/-9.1 | 16.3/-10.9 | 17.2/-12.0 ·
  B0006 11.5/+9.5 | 17.2/-6.4 | 18.1/-8.2 | 18.9/-9.2 · B0018 7.2/+4.1 | 8.7/+1.3 | 11.4/-2.9 | 14.2/-8.4 ·
  B0046 5.5/+5.2 | 5.0/+2.9 | 3.8/+1.6 | 3.7/+1.3 · B0048 8.3/+7.2 | 7.5/+5.5 | 5.2/+1.7 | 3.6/-1.1 ·
  B0041, B0045, B0047: unchanged (EOL 7-8 cycles after the forecast starts: fewer than 10 comparisons before EOL).
  Mean of 8: median 6.2, q50 7.8, q75 8.0, q90 8.4.
- Reference: overcorrects (bias flips to -6..-12, error doubles). Cause: the optimism rate is learned on short
  horizons (5-20 cycles; the forecast starts at 50) and applied linearly 60-90 cycles ahead. B0005 at 58: median 145,
  conservative 101, true 126; rate 0.17 points/cycle, decays to ~0 by cycle 106, where both agree. The common test event
  (73-100) and regenerations also inflate it.
- Cold: q75/q90 help where there is history (B0046, B0048: error -30 to -55 %), but warning timing is unchanged.
- Hot: no EOL to score; q90 would put B0030 and B0032 on Watch at 22 and 34 with SOH 87-92 % (likely too early); q75 never.
- Warnings: conservative moves Reference Watch 9-24 cycles earlier (B0006 61 vs 85; true EOL 110), i.e. 36-49 cycles
  ahead for a 20-cycle window: earlier than useful.
- DECISION: default stays "Median forecast"; the option stays in the app as experimental, documented here. Report the
  late bias of the median (5as) together with this negative result. Possible fix not pursued: apply the correction
  only up to the horizons where it was measured (no linear extrapolation); would need a new NASA run.

## 5av. Review against the research plan + Mission 3 optimisation check (2026-10-28)
- Plan (Internship_Research_Plan, project_master_context): M3 = optimisation (J_op = revenue - energy cost,
  J_maint = maintenance cost + performance loss, profit; trade-off aggressive operation / faster degradation /
  delayed maintenance; deliverables: integrated optimisation framework, demonstration case study, "what benefit?").
  Option B work (5ah-5au) covers monitoring (forecast, anomalies, explanation, what-if), not the optimisation.
- The optimiser exists (Operations: twin-aware policy, integrated O&M grid, DP, mismatch study; built before Mission 2
  closed; no NASA results in STATUS). Its plant takes Ea, current exponent and cold multiplier from
  stress_factor_regression; maintenance hazard, prices and downtimes are ASSUMED values.
- Regression issues (code review): pools all 34 cells incl. Mixed and NASA-flagged B0049-56; 24 C current effect rests
  on B0033/B0034 (4 A, different cut-offs); cell-level bootstrap ignores campaign clustering (CI too narrow);
  Arrhenius on cell T + cold flag on ambient partly collinear; "early fade" = first half of record (20 cycles Hot,
  84 Reference).
- NASA does vary current at fixed temperature: 24 C 2 A (B0005-7, 18, 36) vs 4 A (B0033, 34); 4 C 1 A (B0045-48) vs
  2 A (B0049-56, flagged). So the shared physics' a_I = 0 (lower bound) likely means "no extra fade per Ah at higher
  current", not "cannot tell": the what-if wording "limited" must be revised once confirmed.
- Optimiser sensitivity (generic cell, demo engine): best operation flips with the current exponent (0 -> fixed 4 A,
  0.5 -> 4 A, 1 -> 2 A); replacement SOH 0.72 in all cases (driven by the ASSUMED hazard onset 0.68). Twin-aware
  greedy policy below best fixed current by 2-12 % (generic) but +52 % ... 0 % on a calibrated demo plant: the benefit
  depends on the law -> must be decided on NASA. DP (lifetime-optimal by construction) is the real optimiser.
- scripts/m3_condition_check.py (new): per-cell fade (half record and first 40 cycles), regressions on 3 subsets x
  2 windows x cell/group bootstrap, leave-one-group-out, current contrasts at fixed temperature, calibrated plant,
  optimiser sensitivity over the current exponent. Demo dry run OK (68 s).

## 5aw. MISSION 3 REBUILT on the validated twin (user decision 2026-10-28)
- DECISION: the pre-Mission-2 optimiser (Operations: plant, twin-aware greedy policy, O&M grid, DP, mismatch) and
  scripts/m3_condition_check.py are SET ASIDE. Mission 3 is rebuilt from the plan's objectives in three stages:
  1 maintenance (when to replace) on the real NASA lives with the twin's own forecasts; 2 operation (current, cut-off)
  with NASA-measured revenue/energy per condition and the shared physics' fade differences, tested conditions only;
  3 integrated (joint choice, lifecycle profit, "what benefit?") + app page.
- Stage 1 (engine section 56, v5.32.0): M3Economics (revenue per cycle x SOH = performance loss; planned replacement
  = battery_cost_cycles of new-battery revenue; unplanned x unplanned_factor + lead-time downtime), m3_life_profit,
  m3_twin_policy (order when forecast EOL within lead + margin; optional alarm and age-limit fallback), m3_oracle
  (best replacement in hindsight), m3_fixed_age, m3_maintenance_study (leave-one-out: twin margin on the other
  batteries, fixed age on the other batteries of the same group). Input = integrated timelines.
- NASA (8 batteries that reached EOL: B0005, B0006, B0018, B0041, B0045-48; v5.31.0 timelines; default: battery
  = 60 cycles of revenue, unplanned x3, lead 10): share of the perfect-knowledge gain over run-to-EOL: twin 0.42,
  twin + age limit 0.40, best fixed age 0.11, run to EOL 0. Unplanned: twin 3/8, fixed 5/8, run to EOL 8/8.
  Twin replaces 9-21 cycles before EOL in Reference (B0005 117 vs 126, B0006 99 vs 110, B0018 89 vs 98).
- Sweep (battery 30/60/120 cycles x unplanned 1.5/3/5 x lead 5/10/20): twin above best fixed age in 26/26 settings
  with room to gain; twin share 0.33-0.86 (lead 5: 0.64-0.86, lead 20: 0.33-0.50 apart from cheap unplanned);
  fixed age 0.09-0.14 when unplanned >= 3x, negative when unplanned is cheap. Margin chosen: 5 cycles (matches
  the late bias, 5as).
- Limits: 3 Cold batteries (B0041, B0045, B0047) are always unplanned for lead >= 10: EOL comes within a few cycles
  of the forecast start (30 % of the record), so no forecast can warn in time; the age fallback does not help (Cold
  lives 29-60 cycles, too variable). The forecast start at 30 % of the record uses the record length (not known
  live): to fix in the live definition (fixed start cycle). 8 batteries: shares are indicative, not significant.
  Prices are assumed and swept; only the ranking across the sweep is claimed.

## 5ax. Mission 3 · stage 2: operation (v5.33.0, engine section 57)
- Knobs: discharge current and cut-off voltage; temperature = the battery's environment (not a control). Options =
  only (current, cut-off) combinations NASA ran at the battery's tested temperature (each battery's own current,
  rounded 0.5 A; cut-off 0.1 V). NASA varies the cut-off inside every temperature group (2.0/2.2/2.5/2.7 V): depth is
  the best-identified knob; current varies only at 24 C (2 vs 4 A) and 4 C (1 vs 2 A).
- Economics MEASURED per option (median of the first 10 valid cycles of the batteries run that way): Ah delivered,
  Ah moved per cycle (cum_Ah, charge + discharge = the physics' axis), Wh out, Wh in, hours (discharge + CC + CV),
  round-trip efficiency; scale with SOH over life. M3OperationEconomics: sell price per Wh, buy price ratio, battery
  cost in own-option beginning-of-life J_op cycles (one physical battery, priced once), rest hours, EOL as SOH.
- Ageing: shared physics v2 (hybrid) with the option's conditions, scaled by one factor k fitted to the battery's
  measured fade (m3_fade_scale; corrects physics optimism); throughput cycle by cycle (m3_operation_life).
  Bug found by the test and fixed: throughput per cycle must be Ah MOVED (~4.1 Ah), not Ah delivered (~2 Ah).
- m3_operation_study: life, J_op, J_op per hour, profit rate with planned replacement at EOL, difference vs own
  option; flags when the current or cut-off sensitivity is limited (at a bound).
- Demo: ranking depends on the battery-cost ratio (S001: 1 A worse with cheap batteries, +9 % with realistic lives)
  -> the plan's trade-off; must be swept on NASA.
- scripts/m3_stage2.py: fits the shared physics on NASA (v2, Re, 30 starts, as the app), writes
  whatif_identifiability.csv, physics_params.csv, measured_economics.csv, options.csv, operation_study.csv
  (battery cost 30/60/120 x buy ratio 0.2/0.4/0.6 x EOL 0.7/0.8), best_option.csv.

## 5ay. Mission 3 in the app (v5.34.0; user rule: everything runs in the platform, no Colab)
- Operations & control = anomaly detection (unchanged) + Mission 3: "Prices and assumptions" (battery cost in cycles,
  buy/sell price, unplanned x, lead time, end-of-life rule), "1. Operation" (selected battery: tested current x
  cut-off options, chart + table, identifiability; button: every battery x 18 price settings -> which option is best,
  CSV), "2. Maintenance" (button: replays the frozen forecast + detector on every constant-condition battery that
  reached end of life; policies, share of the possible gain, replacement cycles, 27-setting sweep, CSV).
- End-of-life rule "NASA convention": 1.4 Ah, or 80 % of initial when 1.4 Ah is within 2 % of initial or above it
  (reproduces 5as/5aw: Reference 1.4 Ah, Hot/Cold 80 %). Alternative: control-bar setting.
- Uses the twin's fleet-wide parts (shared physics v2, anomaly scan; same as the Integrated twin, saved in backup).
- Old optimiser sections removed from the page (engine code kept, unused). Study results: M3 answers now from the
  new studies. scripts/m3_stage2.py withdrawn (not uploaded).
- Checks: AppTest of the page (build, option study, all-battery sweep, replacement study with both EOL rules, price
  change, Study results), all pages, 21 tests, md_check 0.

## 5az. First NASA run of Mission 3 in the app (v5.34.0) -> two errors found, fixed in v5.35.0
- Replacement study: twin 10 % of the possible gain (vs 42 % in 5aw) because the "NASA convention" rule gave Cold
  batteries starting at 1.5-1.7 Ah a 1.4 Ah threshold (~88-90 % SOH): EOL at 11-18 cycles (B0046 18, B0047 11,
  B0048 17 instead of 44/30/46), before the forecast starts. Fix: 1.4 Ah but never above 80 % of initial capacity
  (reproduces 5as: Reference 1.4 Ah, Cold 80 %). The v5.34.0 replacement numbers are INVALID.
- Operation study: each battery's best option was another battery's operation (B0005 own 2.7 V -> 2.4 V from
  B0006/B0018; B0006/B0018 own ~2.4 V -> 2.7 V from B0005/B0036): option economics came from 1-2 other cells, so
  cell-to-cell differences posed as a depth effect (2.4 V delivered MORE energy than 2.1 V: 6.87 vs 6.76 Wh; B0006
  starts at 2.04 Ah, B0007 at 1.89). The v5.34.0 operation numbers are INVALID.
  Also seen: fade scale k 0.14-3.06 (physics level far off for some cells, e.g. Pulsed 0.14-0.57, B0041 3.06);
  "current: limited" flags on 108 of 1278 rows; cut-off labels are measured end-of-discharge voltages (~0.1 V below
  NASA's nominal 2.0/2.2/2.5/2.7).
- Fix (engine): m3_depth_table reads the raw discharge traces (first 10 valid cycles) and measures INSIDE each battery
  the Ah, Wh and hours down to every voltage it reached (1.8-2.8 V). Option economics = this battery's own Ah to
  2.7 V x the peers' (same temperature, same current, reached that depth) Ah / Wh / hours per Ah-to-2.7 V; charge
  energy from peers' efficiency, charge time per Ah and Ah moved per Ah. Falls back to between-battery values (shown
  as "economics: between batteries") when the depth table is missing. App: depth table cached + in result backup.
- Tests 23: depth table monotone and matching capacity; option economics follow the target battery, not the peers.

## 5ba. Mission 3 on NASA in the app (v5.35.0) -> stages 1 and 2 validated
- Replacement study (NASA convention fixed): reproduces 5aw exactly: twin 42 % of the possible gain, twin + age 40 %,
  best fixed age 11 %, run to EOL 0; unplanned 3 / 3 / 5 / 8 of 8; Reference replacements B0005 117 (EOL 126),
  B0006 99 (110), B0018 89 (98); Cold EOL 29-60 cycles.
- Operation study (1278 rows, all "within battery" economics; flags: 108 "current: limited", cut-off never flagged
  -> the depth sensitivity is identified, the current one is not):
  - Depth: inside the batteries, discharging deeper than ~2.7 V adds almost no energy (Reference B0005: 6.55 Wh at
    2.7 V, 6.58 at 2.4 V, 6.59 at 2.1 V) but the physics charges extra fade (B0005 life 142 / 133 / 124 cycles).
    Median over Reference options: -0.3 V -> +0.2 % energy, -6 % life.
  - Best option over 18 price settings: Reference 2.7 V 56 %, 2.4 V 44 % (B0006/B0018/B0007 -> shallower; B0005/B0036
    stay at 2.7 V); Hot 2.6 V 100 %; High current 2.1 V 100 %; Cold 2.7 V 60 %, 2.0 V 38 %; Pulsed 2 A 69 % (current:
    limited, not claimed). Own operation best in 19-50 % of settings; ranking almost independent of the prices.
  - Size of the gain (best vs own, % of own operating profit per hour, median): Reference 2.6, High current 2.8,
    Hot 4.8, Pulsed 4.0, Cold 15.2 (Cold profits are near zero/negative: lives 18-60 cycles vs battery cost, so its
    percentages are unstable).
- Answer to the plan (aggressive operation vs faster degradation), within NASA's evidence: the lever NASA identifies
  is depth of discharge; stopping near 2.6-2.7 V (measured) instead of 2.0-2.4 V loses <1-2 % energy per cycle and
  gains life, worth a few % of operating profit per hour. Current: effect not identified (sensitivity at its
  bound), no claim. Maintenance timing with the twin (42 % vs 11 %) is the larger lever.

## 5bb. Mission 3 · stage 3: lifecycle profit, "what benefit is expected?" (v5.36.0, engine section 58)
- Live forecast window (fixes the leak noted in 5aw): frozen_live_window = start at 30 % of the TRAINING batteries'
  median record (safe_origin applied), horizon 1.4 x that median. Used by the Integrated twin and Mission 3
  (frozen_cached got optional n_start / n_max); the Live twin and the Mission 2 assessment keep their definitions.
  Expect changed Integrated-twin timelines and replacement numbers (e.g. B0041: forecast started at 53, now ~22).
- m3_lifecycle_study: per battery that reached EOL, lifecycle profit per hour (Stage 2 units: Wh x price, energy
  cost via efficiency, hours per cycle, battery priced once, unplanned x k + lead-time downtime) of: do nothing,
  usual practice (best fixed age, LOO), twin replacement (stage 1), twin strategy (operation chosen on the OTHER
  batteries of the group by mean gain vs own + twin replacement), upper bound (best operation + best replacement in
  hindsight). Measured life = reality; another operation scales it by the physics' life ratio; the twin's timing
  relative to EOL is assumed unchanged (stated assumption). Outputs: share of the possible gain, % vs usual practice
  of operating profit per hour, split into replacement and operation parts; 27-setting price sweep.
- App: Operations & control -> "Mission 3 · 3. Lifecycle" (automatic after the replacement study): three metrics
  (twin strategy vs usual practice, from replacement, from operation), chart, tables, sweep, CSV. Study results:
  M3 answer "expected benefit".
- Demo (synthetic, 3 batteries): twin strategy +72 % vs usual practice (+61 replacement, +11 operation), 25/27 settings
  positive. NASA numbers pending.

## 5bc. First NASA lifecycle run (v5.36.0) -> three errors, fixed in v5.37.0
- Live window effect on the replacement study: almost none (twin 42 %, fixed age 11 %; B0018 twin 87 instead of 89).
- Headline was twin strategy +71 % vs usual practice (+34 replacement, +37 operation): NOT VALID, because:
  1 gaps: lives counted only VALID cycles (B0041: 17 valid of 60; revenue for 17 cycles, cost over 60; its best fixed
    age 29 fell before its first valid cycle -> empty "usual practice"). Fix: _m3_series = every calendar cycle
    1..last, SOH interpolated across cleaned-out cycles (affects stages 1 and 3).
  2 charging side still between batteries: efficiency, charge time per Ah and Ah moved came from the PEERS of each
    depth (2.0 V peers B0041/B0045 -> efficiency 0.768 vs 0.820, B0041 4.9 vs 3.1 h per cycle). Fix: from the target
    battery itself (same CC-CV protocol for all); charge energy = option Ah x own Wh in per Ah.
  3 pooled headline: Cold batteries do not pay back their cost at these prices (lives 29-60 cycles), their large
    negative rates dominated the mean. Fix: lifecycle summary per group (+ "all" for completeness); page has a group
    selector (default Reference), a per-group table and per-group sweep; Study results lists each group.
- v5.36.0 Reference rows for orientation only (superseded): usual 0.308/0.238/0.204, twin strategy 0.383/0.265/0.237
  per hour (B0005/B0006/B0018).

## 5bd. Mission 3 on NASA after the fixes (v5.37.0)
- Operation (1278 rows): the best option is the SHALLOWEST tested cut-off in 100 % of settings for Reference, Hot,
  High current and 99 % for Cold; Pulsed: 2 A / 2.6 V in 74 % (current "limited", not claimed). Inside each battery
  the trade-off is monotone, e.g. B0005: 2.1 V 6.59 Wh 95 cycles 0.247/h; 2.4 V 6.58 Wh 102 cycles 0.288/h; 2.7 V
  6.55 Wh 108 cycles 0.318/h. Median gain of switching to the best option: ~5 % of operating profit per hour
  (Reference, Hot, High current, Pulsed); Cold 17 % (unstable: losses). Caveat: the depth ageing effect comes from the
  shared physics (cut-off sensitivity identified); Pulsed lives at 80 % are extrapolated far beyond the record
  (e.g. B0027 1275-1680 cycles, k small): ranking only.
- Replacement (calendar cycles now counted): twin + age limit 69 % of the possible gain (1 unplanned of 8), best
  fixed age 55 % (2), twin alone 47 % (3), run to EOL 0 (8). Cold fixed age 29 = the shortest Cold life: it works
  because the twin cannot warn in time in Cold (EOL soon after the forecast starts). -> combined rule is the answer.
  Per-group shares now shown on the page (app-only change, v5.37.0 kept).
- Still needed from the NASA run: per-group replacement table and the lifecycle section (CSV + per-group table).

## 5be. Replacement study per group (v5.37.0, app-only per-group table)
- Share of the possible gain (unplanned): Reference (3): twin 96 % (0), twin + age 93 % (0), best fixed age 58 % (1),
  run to EOL 0 (3). Cold (5): twin + age 62 % (1), fixed age 54 % (1), twin 34 % (3), run to EOL 0 (5).
- Replacement cycles: Reference twin B0005 117 (EOL 126), B0006 99 (110), B0018 87 (98); fixed age 98 for all.
  Cold fixed age 29 (= shortest Cold life); twin + age: B0041 54, others 29.
- Reading: where lives are long enough to be forecast, the twin is close to perfect knowledge; in Cold the forecast
  starts too late to order a part (EOL 7-24 cycles after it starts), so the age limit carries it. Recommended rule
  for all groups: twin + age limit. Small samples (3 and 5): direction, not precise percentages.

## 5bf. Mission 3 page speed (app-only, v5.37.0 kept)
- Cause of the very slow page: both 27-setting price sweeps sat inside expanders, which Streamlit executes on every
  rerun even when collapsed (each lifecycle sweep = 27 x (replacement study + ~25 operation studies)); any click,
  including the group selector, reran everything.
- Fix: sweeps run only on their button ("Run the price sweep", "Run the lifecycle price sweep"; results kept for the
  session); replacement and lifecycle results cached in the session per price setting. Demo: rerun / group switch
  1.1 s (was minutes); replacement + lifecycle 7 s; lifecycle sweep ~1 min (NASA: several minutes, optional).

## 5bf. Mission 3 page speed (app-only, v5.37.0 kept so saved results stay valid)
- On Streamlit Cloud the lifecycle study started by itself after the replacement study and simulated the whole
  fleet (~25 batteries, Pulsed lives up to ~1700 cycles): the page seemed to hang and the lifecycle section was not
  reached. Now: "Run the lifecycle study" button; it simulates only the batteries tested at the temperatures of the
  batteries that reached end of life; results kept until a price changes (sweeps were already behind buttons).

## 5bg. Lifecycle result (v5.37.0, NASA convention, default prices) and ANSWERS TO THE PLAN'S QUESTIONS
- Lifecycle, % of operating profit per hour vs usual practice (own operation + best fixed age); share of the possible
  gain in brackets. Reference (3): twin replacement +37 (93 %), twin + age limit +34 (91 %), twin strategy (twin +
  operation chosen on the other batteries) +40 (96 %), upper bound +44. Cold (5): twin replacement -49 (40 %), twin +
  age limit +14 (67 %), twin strategy -44 (42 %), upper bound +93. Operation part: Reference +3, Cold +5.
  "Twin + age limit" lifecycle values computed offline from the CSVs (exact identity for own operation: stage-1 rate x
  usual J_op per hour; checked on all 8 batteries); with the chosen operation added approx: Reference +37, Cold +20.
  -> Recommended rule: twin + age limit + shallowest tested cut-off; the only strategy above usual practice in both
  groups. Cold profits are negative at these prices (losses compared). TODO next engine release: "twin + age +
  operation" computed in the app (one rebuild).

### Answers (frozen 2026-10-29; numbers from the sections cited)
M1 · Which parameter best represents battery health? One or several? Best-ranked indicator: discharge energy
     (fitness 0.85; capacity family); best power-fade indicator: load-step DC resistance. PCA: PC1 49 % of indicator
     variance (|corr| with SOH 0.60), PC2 19 % -> ageing is multi-dimensional (capacity and power fade evolve partly
     independently): supports theta = [SOH, R_int, R_ct]. The twin FORECASTS capacity-based SOH and TRACKS the
     resistances (diagnostics / inputs); state this choice in the write-up. (Study results, v5.37.0)
M1 · How do operating conditions influence degradation? Cold (4 C): lithium plating dominant (71-77 % of the loss in
     both physics versions; B0041/B0045 also 37-56 % not explained). Room temperature: LAM dominant in both versions
     (robust). Hot (43 C): SEI vs LAM split depends on the physics version (not determined). Depth of discharge: deeper
     than ~2.7 V adds <1 % energy and costs 6-12 % of life (cut-off sensitivity identified). Discharge current: effect
     not identifiable from NASA (sensitivity at its bound; current tied to the group). (5ae, 5as, 5bd)
M2 · Can we predict future degradation accurately? Yes within a group: frozen models RMSE Reference 4.47, Hot 0.95,
     Cold 2.77 SOH points at 30 % of life; end-of-life forecast mean error 6.2 cycles on 8 batteries, late in 7 of 8;
     a self-correction from the twin's own track record did not fix the bias (5h, 5as, 5au).
M2 · How often should the model update? Anchor to the measured capacity every cycle; retraining barely matters
     (Reference never updated 4.35 -> anchored, no retraining 1.45) (5h).
M2 · Which variables are most informative? The measured capacity (anchor) carries the update; adding 19 measured
     state features to the PINN made it worse (no-physics twin Hot 0.93 -> 1.45, Cold 4.79 -> 7.81: overfitting with
     3-4 training batteries); physics helps only in Hot (5ap, 5p).
M2 · supplementary (Live-twin cohort validation, Study results CSV, 32 batteries, 20-cycle forecasts at 30 / 50 %
     origins, single-battery online models): median RMSE (SOH points) particle filter 2.03, hybrid 2.23, ensemble 2.25,
     mechanistic 2.38, trend 3.64, default twin 4.18; band coverage ensemble 1.00, PF 0.90, twin 0.38 (too narrow).
     Mechanistic fails on High current (39.7). Supports the Mission 2 choice: fleet-trained frozen models + anchoring
     rather than single-battery estimators. Default Live-twin estimator should not be presented as the best one.
M1 · NOTE: the Study results "Operating conditions" line comes from the set-aside stress regression (5av, flawed): use
     the M1 conditions answer above instead.

M3 · How to perform the integrated optimisation? Lifecycle profit per hour = J_op (measured energy delivered and
     charged per tested operation, ageing from the shared physics scaled to the battery) - J_maint (planned /
     unplanned replacement, lead-time downtime), evaluated on the real NASA lives with every data-driven choice made
     leave-one-out (5aw, 5ax, 5bb).
M3 · What benefit is expected? Reference: +40 % lifecycle profit per hour vs usual practice (96 % of the possible
     gain), mostly from replacement timing (+37). Cold: twin alone -49 %, twin + age limit +14 % (approx +20 % with
     shallow cut-off). Recommended: twin + age limit + shallowest tested cut-off. Assumed prices, 3 + 5 batteries.

## 5bh. Platform audit (2026-10-29, app-only changes; engine v5.37.0 unchanged, saved results stay valid)
- Fixed (stale content): Home M3 status chip read the old optimiser (now: Mission 3 studies); Home card text for
  Operations; Diagnostics "operating conditions" regression (shown twice) now labelled screening only, not used by
  the twin; Study results M1 answer: regression coefficients replaced by the shared-physics identifiability
  (identified / at the edge of its range / not identifiable; needs the fleet parts built); Study results methods:
  old Integrated O&M text and "operations plant / stress regression" rows replaced by the Mission 3 lifecycle method;
  Integrated twin model card: live forecast start wording; Watch options "Conservative" marked experimental (5au);
  identifiability tables explain "limited" (at 0 = no extra effect found, no claim); version guard now exact.
- Checked OK: all pages run; 27 tests; md_check 0; Mission 3 flow; hybrid caching fix (v5.30) in place.
- Watch items (not changed): Streamlit 1.66 deprecates st.components.v1.html (used for the animations, app.py ~2111):
  keep requirements below 1.66 or port to st.iframe before upgrading. Old optimiser code (fig_dp_policy, om/mm
  figures, POLICIES) is dead code: remove at the next engine release. Engine wording "limited" for a sensitivity at
  0 (whatif_identifiability) and the lifecycle strategy "twin + age limit + chosen operation": next engine release.
  Live-twin default estimator is the weakest in the cohort validation (5bg supplementary): do not present it as best.

## 5bi. Framework connections (2026-10-29, app-only; engine v5.37.0 unchanged)
- Review: backbone connected (SOH -> frozen forecaster per group -> anchor rule -> Integrated twin -> Mission 3 ->
  lifecycle profit). Supporting evidence only: Live-twin estimators and update-frequency study (justify fleet models),
  health-indicator ranking + PCA (justify SOH; power fade enters the decisions only via measured efficiency),
  electrochemistry (cross-check of the physics split). Independent / set aside: condition regression (screening),
  stress & safety thresholds (not constraints; all options are tested conditions), conservative Watch. Outside the
  Mission 3 benefit: Hot, High current, Pulsed (no EOL in record), Mixed (varying conditions). Two physics fits by
  design: the detector keeps its own validated fleet fit.
- Connected now: (1) anomaly alarms enter the replacement decision: policies "twin + alarm" and "twin + alarm + age
  limit" (Alarm status also triggers the order; margin re-chosen leave-one-out with alarms on), in the summary, by-group
  table and Study results; (2) Live twin frozen stream uses the live forecast start (frozen_live_window), consistent with
  the Integrated twin (shares its saved results when settings match).
- To run on NASA: replacement study -> compare twin + alarm (+ age limit) with 5be (B0006 FLEET 54, B0041 59, B0045 21).

## 5bj. PINN improvement · review of the code and step 0 diagnostic (2026-10-29, app-only)
- PINN v1 as built: trained WITHIN the group (3-4 other batteries + the target's cycles to the origin); inputs
  throughput + temperature, cold flag, current, which are CONSTANT inside a group -> the condition physics cannot be
  learned in the assessment, batteries differ only by a battery code from their first 30 %; the cut-off voltage (varies
  inside every group, identified by the shared physics) is neither an input nor in the equations (B0045, worst battery,
  is the 2.0 V Cold cell); physics = soft residual penalty, its parameters re-learned on 3-4 batteries per run.
- Correction: "physics in the structure" was already tested (5ac-5af: shared physics, hybrids, v3, speed factor):
  all optimistic because every equation decelerates and nothing accelerates late in life. Shared root cause.
- Hypothesis for the missing acceleration AND the missing cut-off: capacity is measured under load to a fixed cut-off;
  resistance growth adds a voltage drop x = I x (R - R0) that reaches the cut-off earlier (resistance-driven knee;
  Attia et al., J. Electrochem. Soc. 2022 knee review, to verify before citing).
- Step 0 (app, Models & forecasting -> "Diagnostic · is the loss the physics misses driven by resistance?"): physics
  gap = SOH physics v2 - SOH measured; leave-one-battery-out OLS per group: throughput + throughput^2 vs + x.
  Supported in a group if the x term removes >= 15 % of the remaining error AND its effect is positive in every
  leave-one-out fit; verdict over groups (>= 2 supported / 1 partly / none). Criteria calibrated by simulation (150 runs
  per setting): detection 76-98 %, false alarm 0-4 % with 3-5 batteries (a first design with an R^2 gain >= 0.05 and a
  rank correlation >= 0.5 failed its own positive control and gave spurious correlations: replaced).
  Demo: not supported (correct: synthetic data has no resistance-driven loss).
- Next: NASA run of the diagnostic. Supported -> PINN v2 (cut-off input + resistance observation term, pooled training,
  5-seed ensemble). Not supported -> next candidate: reversible rest-dependent term (regeneration).

## 5bk. Literature review (Advanced Research, 2026-10-29) and step 0 magnitude check (app-only)
- Review findings: (1) most likely cause of late-life optimism = missing OBSERVATION law: capacity measured under load
  to a fixed cut-off; linear R or LLI growth alone can give a measured knee through the OCV curve shape ("threshold"
  pathway; Attia et al. 2022 JES 169:060517; Mandli et al. 2019 IJER 43:2044). Shown on NMC/graphite, NOT on
  LCO/NASA; critique: may need resistance growth of hundreds of %. (2) Regeneration: rest time as exogenous input,
  transient additive observation term (Saha & Goebel 2009 PHM: C_{k+1} = eta C_k + b1 exp(-b2/dt_k); Olivares et al.
  2013 IEEE TIM; Qin et al. 2016 Energies 9:896; Energies 2024 17:1679). (3) Architectures: physics in the structure
  (Nascimento, Corbetta, Kulkarni, Viana 2021 JPS 513:230526 / PHM 2021 / Sci Rep 2023 13:13856, NASA, hybrid RNN
  cell + fleet prior, code public), UDE (Bills et al. 2020 Cellfit; Kuzhiyil et al. 2025 Appl. Energy), GP discrepancy
  (Fogelquist & Lin), adaptive loss weights (Wen et al. 2024 IEEE T-IV; Kendall et al. CVPR 2018), hierarchical
  Bayesian pooling (Guo, Li & Pecht 2015 JPS 281:173). PINN4SOH (Wang et al. 2024 Nat Commun 15:4332) = estimation,
  not forecasting. (4) No published NASA PINN uses LOBO forecasting from 30 % of life; sub-1 % errors use
  within-battery splits or past capacity as input -> our protocol is stricter (state in the thesis).
- Recommended PINN v2: latent Q_max(k), R(k) with existing physics -> measured capacity through the battery's own
  discharge curve, current and cut-off; rest-time regeneration term; one model over all groups with condition
  covariates (hierarchical); bounded UDE correction; GP discrepancy; metrics add late-life RMSE and signed bias.
- Step 0 extended (app): per battery the extra voltage drop NEEDED to remove the unexplained capacity, read on its
  own early discharge curve (Q(V) 3.95-3.10 V + raw depth table 2.8-1.8 V, gap interpolated), vs MEASURED x =
  I x (R - R0); "measured / needed" and "R growth needed (%)". Group supported only if the statistical test passes
  AND median measured/needed >= 0.5. Checked against a hand calculation (S002: 224 mV). Bug caught in testing:
  monotone curve used a running minimum instead of maximum (fixed).
  Demo: needed R growth 97-715 % vs measured 37-62 % -> not supported (the literature critique in action).

## 5bl. Step 0 on NASA: resistance-driven knee RULED OUT (v5.37.0 app, 20 batteries)
- Statistical test passes (LOBO, x = I x (R - R0) removes 58 % / 38 % / 55 % of the remaining error in Cold / Hot /
  Reference, positive in every fit) BUT magnitude fails: measured / needed voltage drop median 0.02 / 0.04 / 0.03.
  Load-step resistance grew only 1-29 % (x = 2-58 mV); removing the unexplained capacity through the cut-off needs
  hundreds of mV (25-50x more). Verdict: not supported. The literature caveat (needs hundreds of % R growth) holds
  for NASA LCO at 1-4 A.
- Reading: resistance growth CO-VARIES with the unexplained loss (shared cause; resistance = indicator, as in Mission 1)
  but does not CAUSE it through the cut-off. The magnitude check prevented building PINN v2 on a physically too-small
  mechanism (thesis point: statistical support is not physical support).
- Gap is battery-specific, not a uniform missing acceleration: positive B0005 (+8), B0006 (+20), B0018 (+8), B0041
  (+15), B0045 (+13); negative B0036 (-13), B0046-48 (-6 to -14), Pulsed (-2 to -5); Hot within +-3.
- Data quirk: B0033 x = -71 mV (noisy early load-step R); High current not scored (2 batteries).

## 5bm. PINN v2 (v5.38.0; user choice A, points 1 + 2)
- New assessment level "Level 4 · PINN v2 (pooled + cut-off + causal anchor) + ablation" (pinn:PINN-v2, pinn:PINN-v2-ablation):
  (1) pooled: trains on EVERY constant-condition battery except the forecast one (leave-one-battery-out; the group's
  train list is ignored), so temperature / current / cut-off effects become learnable; (2) cut-off input xV =
  max(0, 2.7 - V_min median)/0.7 and LAM depth term exp(a_V xV) in the physics residual (a_V reported);
  (3) regeneration-robust causal anchor: forecast battery (and CV cut batteries) use SOH_causal (trailing median 5 +
  running minimum, past cycles only) instead of the 3-cycle rolling median, so a regeneration bump at the origin
  cannot lift the forecast. Ablation = same network/data without physics.
- Rest-time term (Saha & Goebel) NOT implemented: the parquet conversion keeps only time within cycles, not the cycle
  start timestamps of the .mat files; adding them to the conversion would allow it later.
- PINN v1 byte-identical: demo forecast and CV error equal before/after; frozen-value test added
  (tests/test_pinn_v2.py, 3 tests; 30 tests total). Demo (S001): v1 RMSE 0.0031, v2 0.0041, v2 ablation 0.0109; v2
  ~1.5x v1 time per battery.
- To run on NASA: accuracy assessment, origin 30 %, Reference / Hot / Cold, levels "PINN (original inputs) +
  ablation" and "PINN v2 + ablation"; judge vs PINN v1, its ablation and the frozen models (4.47 / 0.95 / 2.77),
  incl. bias.

## 5bn. PINN v2 on NASA (v5.38.0, origin 30 %) -> NOT ADOPTED (full pooling = negative transfer)
- Mean RMSE (bias), SOH pts: Reference v1 5.61 (+0.9) / v1-abl 6.32 / v2 5.80 (+0.9) / v2-abl 9.07 / frozen MLP 4.47;
  Hot v1 0.98 / 0.93 / v2 13.48 (-12.7) / 3.15 / DT 0.95; Cold v1 8.44 (-5.5) / 4.79 / v2 7.38 (-4.0) / 4.13 / GP 2.77.
- Per battery: v2 better on B0018 (5.96 vs 9.28) and B0041 (2.89 vs 8.77); worse on B0036 (3.78 vs 1.10), B0048,
  all Hot (7.3-17.7). B0045 still fails (14.8).
- Cause: pooling itself (the no-physics twin also worsens when pooled: Hot 0.93 -> 3.15, Reference 6.32 -> 9.07):
  groups age differently (cold plating, hot short lives); 4-5 distinct condition sets are too few for one network to
  separate them from 3 condition inputs. Literature recommends PARTIAL (hierarchical) pooling, not full pooling.
- v2's own CV RMSE ~19 in every group (v1 1-12): internal CV flagged v2 before the test -> CV selection would reject it.
- Confounded design: pooling + cut-off input + causal anchor changed together; candidate clean ablation "v2 within
  group" (cut-off + anchor, v1 training pool).

## 5bo. PINN v2 within group (v5.39.0): clean ablation of 5bn
- New level "Level 4 · PINN v2 within group (cut-off + causal anchor) + ablation" (pinn:PINN-v2-group, -ablation):
  v2 features (cut-off input + LAM depth term, regeneration-robust causal anchor) with v1's within-group training.
  Engine: v2 switch split into features ("v2") and pooling ("pooled", default = v2); PINN v1 frozen values and PINN v2
  output unchanged (verified); tests 31.
- To run: Reference / Hot / Cold, origin 30 %, level "PINN v2 within group + ablation" (v1 and v2 results of 5bn can be
  reused). Reading: v2-group vs v1 = effect of cut-off + anchor; v2 vs v2-group = effect of pooling.

## 5bp. PINN v2 within group on NASA (v5.39.0) -> CORRECTION of 5bn
- Mean RMSE (bias, CV): Reference v1 5.61 / v2-group 5.83 (+2.2, 8.7) / v2 pooled 5.80 / v2-group-abl 5.06 /
  frozen 4.47; Hot v1 0.98 / v2-group 13.73 (-12.9, 14.1) / v2 13.48 / v2-group-abl 3.78 / 0.95; Cold v1 8.44 /
  v2-group 6.67 (-3.1, 9.8) / v2 7.38 / v2-group-abl 5.16 / 2.77.
- CORRECTION: the Hot collapse is NOT pooling (v2-group fails the same way). Cause = the causal anchor design:
  SOH_causal = running minimum of a trailing median from cycle 1; every Hot battery's first cycle reads 91-94 % (NASA
  first-cycle artefact, next cycle ~101 %), so the anchor locks 3-7 points low (B0029 at cycle 12: measured 97.0,
  v1 anchor 97.3, v2 anchor 93.0; B0031 98.2 vs 91.9). The 5bn "negative transfer" explanation is retracted.
- Pooling effect (v2 vs v2-group, physics on): ~none (5.80 vs 5.83, 13.48 vs 13.73, 7.38 vs 6.67).
- Cut-off input + anchor (v2-group vs v1): Cold better (8.44 -> 6.67; B0041 8.77 -> 3.28, B0047 9.27 -> 6.44),
  Reference unchanged, Hot invalid (anchor bug). Still far from the frozen models.
- Next: fix the anchor (running minimum only after a burn-in of the first cycles) and rerun v2-group once; then close.

## 5bq. PINN as the twin's core + twin scorecard (v5.40.0)
- Goal set by the user: the PINN must be the best model FOR THE DIGITAL TWIN. Seven criteria agreed BEFORE running
  (all weighted): (1) end-of-life forecast error over live cycles, (2) late-life bias, (3) decision value (replacement
  study share of perfect-knowledge gain), (4) band coverage (closest to nominal 80 %), (5) physical consistency (share
  of forecasts rising > 0.5 pt above their running minimum or above 102 %), (6) one model for forecast + explanation
  + what-if, (7) speed (training once at the live start; every cycle = re-anchoring).
- Engine: kind "PINN-twin" (pinn:PINN-twin, te.PINN_TWIN_KEY): v2 features within group + causal anchor with a
  3-cycle BURN-IN (running minimum starts after cycle 3: NASA's low first cycle cannot lock it, fixes 5bp) + 3-seed
  ensemble (seed 0 runs the CV band; band widened by the seed spread; metrics recomputed). Also an assessment level.
  PINN v1/v2/v2-group unchanged (verified); cv_folds=0 now allowed.
- App: sidebar "Twin core model" (Frozen data model (Mission 2) / PINN twin core) used by Live twin, Integrated twin
  and Mission 3 (cache key includes the core); Model updating always frozen (Mission 2). The PINN core goes through
  the same frozen_stream: trained once at the live start (frozen_live_window), re-anchored every cycle.
  Integrated twin page: "Twin scorecard" section scores both cores on every battery that reached end of life,
  per group, with the better core marked; CSVs.
- Demo (synthetic, not evidence): 3 batteries; PINN core EOL error 2.2 vs frozen 10.2 cycles; training 28 s vs 1 s.
- Tests 33 (burn-in anchor; ensemble usable by the twin stream).
- To run on NASA: Integrated twin -> Build -> "Score both twin cores" (PINN core trains once per battery, minutes the
  first time, then saved). Then decide; next lever if it loses on 1-2: late-life acceleration term (symbolic
  regression on the physics gap), or a physics-regularised version of the winning network.

## 5br. Twin scorecard on NASA (v5.40.0): PINN core better in Cold, one Reference failure (B0005)
- Cold (5): EOL error frozen 4.60 / PINN 2.48 cycles; |late bias| 2.19 / 0.53; decision value twin + age limit 0.62 /
  0.62 (tie), twin alone 0.34 / 0.31. PINN better on 4/5 (B0046 5.55->2.73, B0047 4.75->1.25, B0048 8.29->4.54;
  B0045 1.57->2.00).
- Reference (3): EOL error 8.47 / 19.53 driven by B0005 (7.74 -> 45.31, late bias +11.5); PINN better on B0006
  (11.53->8.37, bias +9.05 -> +0.20: removes the frozen model's optimism) and B0018 (6.12->4.90). Decision value
  twin alone 0.956 / 0.651 (B0005 replaced late), twin + age limit 0.929 / 0.930.
- All: PINN more accurate on 6/8 batteries; |late bias| 2.75 / 2.11; mean EOL error 6.05 / 8.87 (B0005).
- Criterion 4 uninformative: coverage ~1.00 for both (nominal 0.8) -> both bands too wide (to calibrate).
  Criterion 5: no impossible forecasts for either.
- Next: diagnose B0005 with the PINN core timeline (suspects: slow early fade learned by the battery code + decelerating
  physics; or one diverging seed -> median of seeds); then GA-PINN (professor's proposal) with nested LOBO fitness incl.
  late-life bias and physics-form genes; band calibration.
- B0005 timeline received (v5.40.0) was produced by the FROZEN core (mean |EOL error| 7.74 = the frozen score; the
  PINN core scored 45.3): the page did not show which core was active. Fixed (app-only): caption "Twin core model"
  on the Integrated twin page, "twin core" column in the timeline CSV, spinner names the core. Re-download needed
  with the sidebar on "PINN twin core".

## 5bs. B0005 with the PINN core: systematic slow slope (not a seed); retraining option (app-only)
- PINN-core timeline (twin core confirmed in CSV): EOL forecast 235 at cycle 56 (true 126), 193 at 72, 162 at 88,
  153 at 100, 125 at 124 -> smooth, monotone convergence = systematic, not a diverging seed (mean |error| 45.3).
- Cause: forecast fade ~0.10 SOH pts/cycle at cycle 56 vs actual ~0.25 afterwards; B0005 fades slowly before the
  live start (~0.13 pts/cycle) then ~2x faster: the PINN's battery code learns "slow" and the decelerating SEI law
  extrapolates it; trained once and only re-anchored, its slope never corrects. Textbook case of the missing
  late-life acceleration (5bj).
- App-only option: sidebar "PINN core: retrain every N cycles" (never / 10 / 20 / 30; default never = results of
  5br unchanged); used by the Integrated twin, Mission 3 and the scorecard (cache keys include it; frozen model stays
  trained once). Demo check: retrain every 30 -> fitted at 36, 66, 96.
- Next: scorecard with retrain every 20; structural fix = acceleration term (GA-PINN physics-form genes / symbolic
  regression), B0005 as the key test case.

## 5bt. Scorecard with the PINN core retrained every 20 cycles (v5.40.0)
- EOL error per battery (frozen / PINN once / PINN retrain-20): B0005 7.74 / 45.31 / 25.14; B0006 11.53 / 8.37 / 9.63;
  B0018 6.12 / 4.90 / 4.33; B0041 2.86 / 1.86 / 1.86; B0045 1.57 / 2.00 / 2.00; B0046 5.55 / 2.73 / 2.82;
  B0047 4.75 / 1.25 / 1.25; B0048 8.29 / 4.54 / 4.67.
- All 8 (frozen / PINN retrain-20): (1) EOL error 6.05 / 6.46; (2) |late bias| 2.75 / 1.35; (3) decision value
  twin + age limit 0.686 / 0.686, twin alone 0.471 / 0.472 (Reference twin alone 0.956 / 0.958: B0005 no longer
  replaced late); (4) coverage 1.00 / 0.96 (both too wide vs 0.8); (5) 0 / 0; (6) no / yes; (7) fast / slower.
- Verdict: PINN core better or equal on 5 of 7 criteria; more accurate on 6/8 batteries; half the late-life bias;
  same decisions; loses (1) by 0.4 cycles (B0005 only) and (7). Remaining target: B0005 (fade ~2x faster after the
  live start) -> GA-PINN with physics-form genes (acceleration term), nested LOBO fitness.
- App: scorecard "better" column now uses tie tolerances (0.25 cycles; 1 percentage point for shares).

## 5bu. GA-PINN (professor's proposal), third twin core, band calibration (v5.41.0)
- Engine: PINN genes behind parameters (default = unchanged; v1, v2, v2-group and twin core verified byte-identical):
  "sei_p" (SEI law 1/(L+0.005)^p; 1 = v1, 0.5, 0 = linear) and "accel" (LAM rate x exp(k_acc x total damage),
  k_acc learnable, reported as "Damage acceleration (k_acc)"). train_pinn_level now attaches cv_late_bias (mean
  held-out error, last third of horizon; + = optimistic) and phys_res (final physics residual).
- GA: genes hidden {16,32,48}, learning rate {0.002,0.005,0.01}, lambda_phys {0.3,1,3}, sei_p {1,0.5,0}, accel {0,1},
  iterations {600,900}; first population contains the current twin-core settings; elitism 2, tournament 2, uniform
  crossover, mutation 0.25; deterministic per seed (cached evaluations -> exact resume; deadline stops a run).
  Fitness (nested LOBO, the PINN's 2-fold CV on the TRAINING batteries cut at the live start; target future never
  used) = RMSE_val (pts) + 0.5 |late bias| (pts) + 0.05 log10(1 + physics residual) + 0.002 x seconds.
  Winner key pinn:PINN-ga|h..|lr..|w..|p..|a..|it.. -> 3-seed ensemble like the twin core.
- App: Integrated twin page "GA-PINN" section (population, generations, time budget per press, batteries; Run /
  continue; per-battery table: fitness current vs best, improvement, winning genes; fitness-per-generation chart;
  CSV of all evaluations; evaluations backed on disk). Third core "GA-PINN core" (sidebar) = each battery's GA winner
  (default PINN core until its search finishes). Scorecard now scores all three cores, best-of-three with tie
  tolerances, plus 4b "band coverage, calibrated": band half-widths scaled by the factor that gives 80 % coverage on
  the OTHER batteries of the group (grid 0.1-1.5), applied to the battery. Integrated twin: "Calibrated band" toggle
  when the scorecard has a scale for (core, battery).
- Demo (synthetic; not evidence): GA keeps the start settings (no acceleration in synthetic data); bands ~2x too wide
  for every core (scale ~0.5), calibration brings coverage 1.00 -> 0.70-0.77. Tests 35. End-to-end AppTest 308 s.
- To run on NASA: (1) optional: sidebar retrain every 10 + scorecard (user's step 2); (2) GA-PINN section, Reference
  and Cold batteries, population 6 x 4 generations, several presses of 10-20 min (B0005 first); (3) scorecard with
  the GA-PINN core; key question: does B0005 select sei_p < 1 or accel on, and does its error drop below the frozen
  model's 7.7 cycles.
- Fix (app-only, after the first NASA GA run): the Integrated twin failed with the GA winner's key
  ('pinn:PINN-ga|h48|lr0.01|w0.3|p0|a0|it900' = B0005's winner) because 7 display lookups used the fixed model
  registry; now model_label() (GA genes read from the key). "Which model does what" describes the active core and its
  retraining. GA table robust to incomplete entries. Reproduced with the exact key on all twin pages.

## 5bv. GA-PINN first NASA results (v5.41.0; GA finished for B0005 only, B0006 generation 1)
- B0005 GA (15 evaluations): winner h48 | lr 0.01 | lambda 0.3 | SEI p 0 | accel off | it 900; nested fitness
  11.72 (current settings) -> 4.68 (-60 %); validation late bias -8.21 -> +0.44 pts. Test on B0005 itself (twin,
  retrain 20): EOL error 25.14 -> 23.91 cycles, late-life bias -2.96 -> -0.61 cycles. Small transfer: nested LOBO can
  only tune for what the training batteries (B0006/7/18) show; B0005's slow-then-2x-faster fade is specific to it.
- PHYSICAL FINDING: all top-10 genomes use SEI p = 0 (steady SEI growth); every p = 1 genome (diffusion-limited,
  decelerating) ranks near the bottom (fitness 8.9-11.7); lambda 0.3 preferred; acceleration term mixed. The data
  rejects the decelerating SEI law for NASA Reference cells (confirms 5bj by systematic search).
- Scorecard all 8 (frozen / PINN / GA-PINN; GA-PINN = PINN core except B0005): EOL error 6.05 / 6.46 / 6.31;
  |late bias| 2.75 / 1.36 / 1.06; coverage 1.00 / 0.96 / 0.95; calibrated 0.79 / 0.78 / 0.77; decision values equal.
- App fix: "better" label names all tied leaders (e.g. "PINN twin core = GA-PINN core" in Cold, 2.52 vs frozen 4.60);
  previously printed "tie".
- Next: finish GA for the other 7 batteries; proposal: self-tuning twin = at each retraining choose genes by causal
  backtest on the battery's OWN recent cycles (adapts to battery-specific acceleration like B0005, no future data).

## 5bw. GA-PINN progress + retraining default (v5.41.0)
- GA finished for B0005, B0006, B0018 (B0041 4 evaluations; B0045-48 not started). Winners: B0005 h48|lr0.01|w0.3|p0|
  a0|it900; B0006 h48|lr0.005|w1|p1|a0|it600; B0018 h32|lr0.01|w0.3|p0|a1|it900; B0041 (partial) p0 + accel.
  p = 0 in all top-5 genomes of B0005 and B0018, 3/5 of B0006.
- That scorecard ran with retraining OFF (sidebar reset to "never" after the app reload: B0005 PINN 45.31 = trained-
  once value), so it is not comparable with 5bt/5bv. Within it, GA-PINN improves on the PINN core wherever the GA
  finished: B0005 45.31 -> 41.19, B0006 8.37 -> 7.97, B0018 4.90 -> 4.12. All: EOL error 6.05 / 8.87 / 8.21;
  |late bias| 2.75 / 2.11 / 1.64; twin-alone decision 0.471 / 0.381 / 0.473.
- App: default "PINN core: retrain every 20 cycles" (was never), so a reload cannot silently switch it off.

## 5bx. GA-PINN complete for all 8 batteries (131 evaluations)
- Winners (h | lr | lambda | SEI p | accel | it; improvement of nested fitness vs current settings): B0005 48|0.01|0.3|
  0|off|900 (60 %); B0006 48|0.005|1|1|off|600 (24 %); B0018 32|0.01|0.3|0|on|900 (51 %); B0041 32|0.01|0.3|0.5|on|900
  (4.5 %); B0045 16|0.002|0.3|0|on|900 (51 %); B0046 16|0.002|0.3|0|on|600 (62 %); B0047 16|0.002|0.3|0|on|600 (55 %);
  B0048 16|0.002|0.3|0|on|600 (60 %).
- Mean within-battery fitness rank by gene (0 best), Reference / Cold: lambda 0.3 0.40/0.35, 1 0.64/0.63, 3 0.85/0.83
  (strongest: lighter physics penalty); SEI p 0 0.48/0.55, 0.5 0.46/0.44, 1 0.67/0.51 (decelerating law rejected in
  Reference, no preference in Cold); accel on 0.51/0.56, off 0.61/0.37 (helps Reference; all Cold winners use it but
  not on average -> not robust in Cold); hidden 16 0.51/0.34 (Cold prefers small nets); iterations 600 0.41/0.43,
  900 0.61/0.58. GA sampling is selection-biased: averages indicative, not a controlled experiment.
- Next: scorecard with all GA winners and retrain 20 (first complete, consistent comparison).

## 5by. COMPLETE twin scorecard: all 8 GA winners, PINN cores retrained every 20 cycles (v5.41.0)
- All 8 (frozen / PINN / GA-PINN): (1) EOL error mean 6.05 / 6.46 / 6.25 (tie GA = frozen within 0.25), median
  5.84 / - / 3.34; (2) |late bias| 2.75 / 1.36 / 0.99 (GA ~3x smaller than frozen); (3) decision value twin + age
  limit 0.686 / 0.686 / 0.687, twin alone 0.471 / 0.472 / 0.478 (tie); (4) coverage 1.00 / 0.96 / 0.97; (4b)
  calibrated 0.787 / 0.778 / 0.793 (GA = frozen); (5) 0 / 0 / 0; (6) no / yes / yes; (7) frozen fastest.
- GA-PINN best or tied-best on 6 of 7 criteria (loses speed only) -> "best model for the digital twin" (criteria fixed
  before running, all LOBO). GA-PINN vs frozen EOL error: better on 6/8 (B0006 11.53->9.32, B0018 6.12->3.90, B0041
  2.86->1.86, B0046 5.55->2.77, B0047 4.75->2.25, B0048 8.29->4.17); worse B0045 (1.57->1.86) and B0005 (7.74->23.91).
- Groups: Cold GA = PINN best on 1, 2, 4b; Reference frozen best on 1 (8.47 vs 12.38, B0005), GA best on 2.
- Remaining weakness: B0005 (slow then ~2x faster fade, absent from its training batteries). Proposed next: self-tuning
  twin (genes chosen at each retraining by causal backtest on the battery's own recent cycles).

## 5bz. Self-tuning GA-PINN core (v5.42.0)
- Aim: battery-specific acceleration (B0005) that no training battery shows, which nested-LOBO GA cannot learn.
- Engine: kind pinn:PINN-self|<g1>~<g2>~... (te.pinn_self_key / pinn_self_candidates). At each (re)training point n0:
  for each candidate genome, CAUSAL backtest on the battery's own last SELF_TUNE_WINDOW = 20 cycles (train as at
  n0 - 20 with the training batteries + own cycles up to n0 - 20, re-anchor at n0 - 20, RMSE on own measured cycles in
  (n0 - 20, n0]); needs >= 15 own cycles before n0 - 20 (else candidate 0). Switching margin SELF_TUNE_MARGIN = 0.20:
  candidate 0 (GA winner) kept unless another backtests >= 20 % better (a first demo run without the margin switched
  on noise: EOL error 4.36 vs 2.42 GA-PINN). Then 3-seed ensemble at n0; choice and scores reported.
- Causality test: lowering the battery's SOH by 0.2 after n0 leaves the choice and the forecast at n0 identical.
- Demo with the margin (synthetic, no acceleration): EOL error frozen 10.15, PINN 2.30, GA-PINN 2.42, self-tuning 2.64
  (4.36 without the margin): noise-driven switching removed; small residual cost where nothing accelerates.
- App: fourth core "Self-tuning GA-PINN core" = each battery's top-3 nested-GA genomes + the default settings (default
  PINN core until its GA finishes); scorecard scores 4 cores; labels. GA-PINN and twin-core outputs unchanged.
- Tests 36.
- To run on NASA: sidebar retrain every 20 (default), then "Score the twin cores" (GA results reused; only the new core
  trains: 4 backtests + ensemble per retraining). Key question: B0005's EOL error (frozen 7.74, GA-PINN 23.91).

## 5ca. Scorecard v5.42.0 ran WITHOUT GA results -> restore from CSV (app-only)
- In the v5.42.0 scorecard every PINN core used pinn:PINN-twin (identical columns): GA results lived in the session
  (lost at the app restart) and GA evaluations are disk-cached per engine version (5.41.0 entries not reused). That run
  only reproduces 5bt (frozen 6.05 / PINN 6.46 mean; median 5.84 / 3.58; B0005 25.14).
- Fix: GA-PINN section "Restore GA results from an evaluations CSV": rebuilds every battery's winner and ranked
  candidates from ga_pinn_evaluations__*.csv (battery, genome, fitness); restored batteries are not rerun; table and
  evaluation CSV use the restored values. Tested on the user's 131-evaluation CSV: all 8 winners match the GA table.
- To do: restore from the latest evaluations CSV, then Score the twin cores (GA-PINN + self-tuning).

## 5cb. Resumable twin scorecard (app-only)
- On NASA, "Score the twin cores" ran until the platform stopped it, with no result: four cores x 8 batteries with
  retraining every 20 cycles = several hundred PINN trainings in one press (fresh caches after the 5.42.0 release),
  plus an extra full training per core for criterion 7; the table only appeared at the end.
- Now: "Cores to score" (multiselect) + "Time budget per press" (default 15 min). Each press streams (core, battery) pairs
  one at a time until the budget (a pair in progress finishes), records them, and says "k of N streamed, press again";
  the table is computed once all are streamed (fast: everything saved). Criterion 7 = median time to stream one
  battery (all trainings included), measured when first computed; no extra training. Streams shared with Mission 3.
- Demo test (2-min budget): 5 presses, 5 -> 7 -> 9 -> 11 -> 12 of 12, then the table; scores identical to the one-shot
  run (frozen 10.15 / PINN 2.30 / GA-PINN 2.42 / self-tuning 2.64); stream time per battery PINN 111 s, GA 70 s,
  self-tuning 116 s.
- NASA plan: restore GA from CSV, then score in several presses (optionally one or two cores per pass).

## 5cc. FINAL GA-PINN results: four cores, GA restored, retrain every 20 (v5.42.0) -> GA-PINN chapter CLOSED
- All 8 (frozen / PINN / GA-PINN / self-tuning): EOL error mean 6.05 / 6.46 / 6.25 / 6.25 (tie frozen = GA within 0.25),
  median 5.84 / 3.58 / 3.33 / 3.33; |late bias| 2.75 / 1.36 / 0.99 / 0.99; decision value (age limit) 0.686 / 0.686 /
  0.687 / 0.687, (alone) 0.471 / 0.472 / 0.478 / 0.478; calibrated coverage 0.787 / 0.778 / 0.793 / 0.793; impossible 0.
  Reproduces 5by exactly (restore verified).
- Self-tuning identical to GA-PINN on every battery: at no retraining did another candidate backtest >= 20 % better on
  the battery's own last 20 cycles -> kept the GA winner. Negative result, documented; core kept as an option.
- B0005 limit (GA-PINN 23.9 vs frozen 7.7 cycles): error concentrated in the early forecasts (cycles ~56-76), before
  its faster fade appears in any available data (its own past or the training batteries); not predictable causally.
- FINAL CLAIM: GA-PINN = best model for the digital twin, best or tied-best on 6 of 7 agreed criteria (loses speed);
  more accurate on 6/8 batteries; ~3x smaller late-life bias; equal decisions; calibrated uncertainty; one model
  for forecast + mechanisms + what-if. Physical findings 5bx (lambda 0.3; steady SEI law in Reference).
- NEXT (agreed with the user): full review of the platform (inputs, outputs, models, equations, methods, validation,
  numbers, limits; read from the code) -> review document + presentation for the professor, in ENGLISH.

## 5cd. Full technical review for the professor (in progress, English)
- Claude Doc "Self-Updating Digital Twin for Li-ion Batteries · Technical Review":
  https://claude.ai/artifact/WvcXeY8kCmGDB8YAzKzS16 (sections: purpose, data, Mission 1, Mission 2, physics, PINN, twin
  core, anomaly detection, Mission 3, validation and limits, open issues). Every statement checked against the code.
- Done (2026-11-03): purpose; data (groups from NASA READMEs + recorded conditions, pipeline steps, SOH / cleaned /
  causal / end-of-life definitions, limits); Mission 1 (12 indicators, Coble-Hines criteria, fitness = mean of 4,
  PCA, mechanism proxies, half-cell fit, stress index, conditions table, limits).
- Done (2026-11-03, second pass): Mission 2 (modes, inputs, model levels, protocol, frozen table, update study per
  group; flagged: re-anchoring clearly supported in Reference only, not better than no anchor in Hot/Cold); physics
  (v2/v3 equations, fit, hybrid, recovery test, forecasting results, option B); PINN (inputs, network, rate equations,
  loss, training, variants table, anchor bug, resistance diagnostic; ablation evidence mixed across studies, shown).
- Next: twin core (PINN core, GA-PINN, self-tuning, calibration, scorecard), anomaly detection, Mission 3,
  validation and limits, open issues; then the presentation.

## 5ce. GA-PINN as the default twin core for Mission 3; forecast plot; self-tuning removed from the choices
- User request: keep the first PINN core and GA-PINN, drop the other PINN core option, use GA-PINN in Mission 3.
- App: twin cores = Frozen data model (Mission 2) / PINN twin core / GA-PINN core (self-tuning removed from the
  sidebar and the scorecard; engine kind pinn:PINN-self kept). Default core = GA-PINN (Live twin, Integrated twin,
  Mission 3 maintenance and lifecycle; Model updating stays frozen; operation uses the physics ageing, unchanged).
  Sidebar warns when GA results are not loaded (each battery then uses the PINN twin core).
- Operations & control: new section "Mission 3 · Forecast behind the replacement decision" between operation and
  maintenance: battery (those reaching end of life), cores to compare (default frozen + GA-PINN), today slider; plot of
  measured SOH (seen / unseen), each core's forecast at today with 80 % band, predicted end-of-life diamonds,
  threshold and real end of life; table of predicted vs real end of life and error. Uses the same saved streams.
- Demo AppTest: page, build, plot (frozen + GA fallback) without errors.
- Note: run "Score the twin cores" (or restore + Mission 3 build) after an app restart; GA results need the restore.

## 5cf. Mission 2 Level 4: first PINN + GA-PINN only (app-only)
- User request. Models & forecasting, Level 4 section: model choices = "PINN" (v1, physics-informed) and "GA-PINN" (this
  battery's GA winner, 3-seed ensemble); both default; plot and table show both forecasts side by side (+ the usual
  level ensemble). GA-PINN is excluded from the hyperparameter editor, equation icons and automatic tuning (its
  settings come from the GA); without a GA result for the battery a warning is shown.
- Accuracy assessment, Level 4 options: "PINN (original inputs) + ablation" and new "GA-PINN (each battery's GA
  winner)" (virtual key pinn:GA-PINN: each held-out battery runs its own GA winner, chosen without it; rows relabelled
  as one model, family Physics-informed). The other PINN levels (all inputs, v2, v2 within group, twin core) are
  removed from the choices; their engine kinds and results stay documented (5ap, 5bn, 5bp, 5bq).
- Demo AppTests: Level 4 trains PINN + GA-PINN; assessment GA level runs on all batteries with the relabelled rows.
- GA results must be loaded (run or restore) before using GA-PINN anywhere.

## 5cg. GA-PINN loads automatically; Models page reorganised; best PINN before the GA added (app-only)
- GA results file results/ga_pinn_evaluations.csv (the user's 131-evaluation CSV, 8 NASA batteries) ships with the app:
  ga_autoload() loads it at startup when the session has no GA results (any page), so GA-PINN works without running or
  restoring the GA. A GA run or a restore also writes the file (ga_save; until the next restart unless committed).
  Sidebar caption only when no results are found.
- Models & forecasting order: Level 1 baselines -> Level 2 machine learning -> Level 3 deep learning -> Level 4
  physics-informed (first PINN, PINN twin core = best PINN before the GA, GA-PINN) -> leaderboard -> accuracy
  assessment -> "Explanation · Hybrid" (the twin's explanation model, not a forecaster). Removed from the page:
  shared-physics-only section and the resistance diagnostic (functions kept; the twin builds the shared physics itself;
  diagnostic result in 5bl). Header "levels 1-4". Fixed a duplicate key in LEVEL_INFO that labelled Level 4
  "Deep learning" in the level stepper.
- Assessment Level 4 options: PINN (original inputs) + ablation; PINN twin core (best before the GA); GA-PINN.
- Demo test: GA file under demo names auto-loaded in a fresh session; Level 4 trained all three (S001 got B0005's
  winner h48 | lambda 0.3 | p 0 | accel off).
- GA results now EMBEDDED in app.py (GA_EMBEDDED_CSV, 131 evaluations, 16 KB): app.py alone is enough; a
  results/ga_pinn_evaluations.csv (newer run / restore) takes precedence. Embedded copy parses to the 8 winners of 5bx.

## 5ch. Colab notebook for an extended GA-PINN study (user request; runs the same engine)
- GA_PINN_extended_search.ipynb (21 cells) + twin_engine.py 5.42.0; inputs battery_master_data.parquet (+ optional
  impedance_ground_truth.parquet) in a Drive folder; every evaluation appended to evaluations_raw.csv (resumable).
- Addresses the five limits: (1) larger GA (12 x 8) + optional full grid of 324; (2) re-ranking under 36 weightings,
  winner stability; (3) 3 seeds per configuration, paired gain / SE vs the default; (4) balanced random design of 36
  configurations + OLS of within-battery rank on genes with bootstrap CI; (5) B0005 oracle check (test-future error,
  diagnosis only) + causal self-tuning backtest every 20 cycles.
- Output ga_pinn_evaluations_colab_<BATTERY>.csv in the platform format (+ fitness_sd, n_seeds): restore or embed.
- Parallel use (user: one notebook per battery): MODE = "run" + BATTERY per notebook, own file
  evaluations_raw_<BATTERY>.csv (no concurrent writes); MODE = "merge" combines all files -> ..._ALL.csv and the
  cross-battery analyses; the B0005 self-tuning table is saved by B0005's notebook (selftuning_B0005.csv).
  Bug found by the merge test and fixed: reloaded evaluations lost column names (itertuples renames "late bias (pts)"),
  which also broke resuming; now to_dict("records"). Demo: 2 parallel runs + merge + resume (5 s, nothing recomputed).
- Smoke-tested end to end on the synthetic demo (DEMO = True). NASA cost: about 22 s per evaluation -> roughly 1.5-2 h
  per battery with 3 seeds (about 15 h for 8 batteries; budget 11 h per session, re-run to resume); full grid about
  6 h per battery.

## 5ci. GA-PINN searched on demand in Mission 2 (Level 4 and assessment), like the other models (app-only)
- User: restored results were not used on the Models page; GA-PINN should train for any battery like other models.
  Cause: saved GA winners are per battery at the TWIN's live start with the group's training batteries; Level 4 and
  the assessment use other forecast starts / training sets (and other batteries), so no winner matched.
- ga_search_key(ct, data_key, battery, n0, train): reuses a saved search only when battery, start and training set are
  identical; otherwise runs the GA now (population / generations of the GA-PINN section, default 6 x 4), nested
  fitness on the given training batteries, every evaluation disk-backed (re-run instant). Level 4: search at the
  scheme's start and training set. Assessment: search per held-out battery at its origin (safe_origin of
  origin x recorded life) with the group's other batteries. The twin keeps its own saved / embedded winners.
- Demo test without any saved GA: Level 4 searched (6 evaluations) and trained GA-PINN, re-run 1.9 s; assessment ran
  4 searches and scored GA-PINN on all 4 batteries. Cost on NASA: about 5-6 min per battery the first time.

## 5cj. Where GA-PINN is used (recovery test) + app clean-up (app-only)
- GA-PINN for the EXPLANATION? Mechanism-recovery test (simulated cohort with known splits, pinn_explain with GA genes,
  2 seeds; H1 hot / R2 reference / C1 cold): mean share error PINN v1 32.7, GA-PINN Reference winner (h48 lr0.01
  lambda 0.3 p0) 29.6, GA-PINN Cold winner (h16 lr0.002 lambda 0.3 p0 accel) 30.3 points (shared physics 1.1, 11/11).
  H1 truth SEI/pl/LAM 89/6/6 -> GA-PINN 30/9/61 and 23/26/51. -> NOT adopted for explanation (per-battery codes let
  each battery choose its split); explanation stays shared physics / hybrid.
- GA-PINN used: twin forecast/status (default core), Mission 3 maintenance + lifecycle, Mission 2 Level 4 + assessment.
  Not used: explanation (above); Mission 3 operation (condition effects need the shared physics; GA-PINN's within-group
  condition effects unvalidated); anomaly detection (validated 11/11; a forecast-residual alarm would need the same
  validation first).
- Clean-up (user choice 1-3): removed the pre-Mission-2 optimiser (view_ops_legacy + orphaned integrated_section,
  scenario_section, dp_section, run_policy, tune_rho_cached, fig_policy, fig_mismatch, fig_om_heatmap, fig_om_tradeoff,
  fig_scenarios, fig_dp_policy, dp_cached, om_study_cached), the Diagnostics "operating conditions" regression
  (condition_effects_section, both expanders, method-table mention), and the Integrated twin "Watch uses"
  conservative option (Watch = median forecast; CSV "watch uses" = median forecast). About 370 lines. Engine unchanged.
  Kept (user): Live twin estimators + ablation, update-frequency study. Hidden dead code left (not requested):
  ml_section, cross_cell_section, early_life_section, ml_methods_panel, ladder_deep/hybrid/first_principles,
  self_tuning_key.
- Checks: pyflakes no undefined names (caught a leftover 'basis', fixed); all pages; every Diagnostics tab; Integrated
  twin built; 36 tests.

## 5ck. Colab extended GA study: B0005 results (3 seeds, 91 configurations)
- Winner h32|lr0.01|w0.3|p1|a0|it600, mean fitness 4.80 (sd 0.26) vs default 7.23 (-34 %); default was 11.72 with one
  seed on the platform -> single-seed fitness is noisy.
- Limit 2 (weights): same winner under 64 % of 36 weightings, 2 distinct winners -> moderately robust.
- Limit 3 (seeds): paired gain vs default 2.43, SE 2.07, gain/SE 1.2 (< 2-3) -> on B0005 the GA's improvement is NOT
  established beyond training noise (default varies strongly across seeds).
- Limit 4 (balanced design, OLS on within-battery rank, bootstrap 95 % CI): lambda 3 +0.38 (CI 0.12-0.61), SEI p 1
  +0.30 (0.05-0.52), accel on +0.20 (0.02-0.36) all worse; lr 0.01 -0.23 (borderline better); hidden, iterations none.
  -> lighter physics and steady SEI law better ON AVERAGE (the single winner uses p 1: interactions); acceleration term
  not supported by B0005's training batteries.
- Limit 5 (B0005): oracle-best configuration h16|lr0.002|w0.3|p0.5|a0|it600 error 8.45 SOH pts on B0005's future;
  nested winner 9.76; default 10.57 -> even the best of 91 fails: model-form / information limit, not search.
  Causal self-tuning backtest would pick the oracle-best at cycles 70 and 90 (90: 1.78 vs 2.38, 25 % better, above the
  20 % margin) -> wider candidate set (e.g. top 10 of the balanced design) could help self-tuning after ~cycle 70.
- Pending: other batteries (one per group) and the merge.

## 5cl. Colab extended GA study: B0046 (Cold) + cross-group conclusions
- B0046: 97 configurations x 3 seeds; winner h16|lr0.002|w0.3|p0|a1|it600 = the platform's B0046 winner (confirmed);
  fitness 8.64 (sd 1.51) vs default 18.15 (-52 %); paired gain 9.51, SE 1.04, gain/SE 9.1 -> beyond noise; winner
  under 50 % of 36 weightings (2 distinct winners); winner's oracle error on B0046 4.79 SOH pts.
- Balanced design, Cold (B0046): lambda 1 +0.37 (CI 0.14-0.56), lambda 3 +0.45 (0.26-0.65) worse; hidden 32 +0.32,
  48 +0.24 worse; accel on +0.21 (0.06-0.38) worse; SEI p 0.5 -0.21 (-0.38 to -0.01) better than p 0, p 1 n.s.
- Cross-group: ROBUST = light physics weight (lambda 0.3) best in both groups; acceleration term not helpful on
  average in both (winners using it = interaction). GROUP-DEPENDENT = SEI law (Reference: steady p 0 best, decelerating
  p 1 worst; Cold: p 0.5 best, p 0 worst) -> CORRECTS 5bv/5bx "data rejects the decelerating SEI law" to Reference only.
  Cold prefers small networks.
- GA improvement beyond seed noise: yes for B0046 (9.1), not for B0005 (1.2).

## 5cm. Colab raw-file analysis: GA fitness does NOT predict the test battery's forecast (B0005, B0046, B0029 partial)
- evaluations_raw_B0046.csv also contained 78 evaluations of B0029 (Hot, 26 configurations x 3 seeds, unfinished):
  analyse per battery. B0046 confirmed: winner h16|lr0.002|w0.3|p0|a1|it600 8.64 vs default 18.15.
- Spearman (mean over seeds, configurations with 3 seeds) between GA fitness and the error on the battery's OWN future
  (oracle): B0005 0.07, B0046 0.06, B0029 -0.09; components: validation RMSE 0.01 / 0.06 / -0.19, |late bias| 0.07 /
  0.06 / 0.21, physics residual 0.03 / 0.16 / 0.21 -> none.
- Real error best / GA winner / default: B0005 8.45 / 9.76 / 10.57 (config spread 0.57 = seed noise 0.54: no real
  differences); B0046 4.02 / 4.79 / 4.08; B0029 1.33 / 2.29 / 2.27. Top-10 vs bottom-10 configurations: 9.83 vs 9.75,
  4.57 vs 4.74, 2.23 vs 2.07.
- Interpretation: with 3-4 training batteries cell-to-cell differences dominate; nested-validation selection does not
  transfer to a new battery's single forecast. The GA reliably delivers group-level robust settings (lambda 0.3 in
  both groups; small nets in Cold), training stability (B0005 top-10 seed sd ~0.4 vs default 3.7) and physical insight
  (SEI law group-dependent), not per-battery gains. Twin: GA-PINN's advantage over frozen comes mainly from the PINN
  twin core (cut-off input, burn-in, 3-seed ensemble, retraining); GA's own addition small (6.46 -> 6.25, bias 1.36 ->
  0.99), consistent.
- Recommendation: one robust configuration per group (lambda 0.3 family) or an average of the top configurations,
  instead of per-battery winners. Pending: B0006, B0018 (and finishing B0029 if wanted).

## 5cn. Colab: B0029 (Hot) complete (summary file; raw file pending for the oracle check)
- 92 configurations x 3 seeds. Winner h48|lr0.005|w1|p0|a0|it900 fitness 0.82 (sd 0.02); default 1.28 (sd 0.16), rank
  71/92; gain ~0.46, roughly 5x its uncertainty (unpaired estimate; paired test needs the raw file). Top-10 0.82-0.90.
- Physics weight is GROUP-DEPENDENT: top-10 with lambda 0.3 = Reference 9/10, Cold 7/10, Hot 1/10 (Hot top configs use
  lambda 1-3). Reading: strong physics helps where the equations fit the dominant mechanism (hot: SEI, Arrhenius),
  light physics where they are misspecified (reference LAM, cold plating). Consistent with the early seed study
  (physics helped only in Hot, 0.95 vs 1.43). CORRECTS 5cl "lambda 0.3 best in both groups" -> best in Reference and
  Cold, not Hot. SEI p 0 in 7 of Hot's top 8.

## 5co. Colab study, three groups complete (B0005 Reference, B0046 Cold, B0029 Hot; raw files; balanced design)
- GA vs default beyond seed noise (paired gain/SE): B0029 4.6 yes, B0046 9.1 yes, B0005 1.2 no.
- Fitness vs error on the battery's own future (Spearman): 0.07 / -0.27 / 0.06 (B0005 / B0029 / B0046); real error
  GA winner vs default 9.76 vs 10.57, 2.29 vs 2.27, 4.79 vs 4.08 -> selection does not improve the held-out battery's
  forecast in any group.
- Balanced-design gene effects on within-battery rank (+ worse; * CI excludes 0), Reference / Hot / Cold: accel on
  +0.20* / +0.23* / +0.21* (knee term not supported anywhere); SEI p 1 +0.30* / +0.25* / -0.13 (Cold: p 0.5 -0.21*);
  lambda 3 +0.38* / +0.05 / +0.45*, lambda 1 +0.22 / -0.23 / +0.37*; hidden 32/48 Cold +0.31* / +0.24*.
- Conclusions: GA value = group-level knowledge + training stability, not per-battery selection; physics weight
  light where equations are misspecified (Reference, Cold), no penalty in Hot (SEI-dominated, equations fit).
- Group configurations from these searches would be slightly leaky (a battery's search uses the other batteries'
  futures); honest robust alternative = average of each battery's own top-k configurations (its nested search).
- Pending: B0006, B0018 raw files.

## 5cp. Colab: B0006 (Reference) -> B0005 is the exception; selection does not transfer (4 batteries)
- B0006: 92 configs x 3 seeds; winner h48|lr0.002|w3|p0.5|a1|it600 6.66 (sd 0.53) vs default 8.85 (0.78), rank
  32/92; gain/SE 4.7 (beyond noise). Fitness vs own-future error Spearman 0.02; real error best / winner / default
  2.35 / 4.47 / 3.27 (winner worse than default); top-10 vs bottom-10 4.08 vs 4.40; top-10 with lambda 0.3: 3/10.
  Balanced design: no significant gene effect (accel +0.10, p1 +0.19, lambda 3 +0.03).
- Four batteries (B0005 / B0006 / B0029 / B0046): gain/SE 1.2 / 4.7 / 4.6 / 9.1; fitness-vs-real Spearman 0.07 /
  0.02 / -0.27 / 0.06. Accel on worse in all four (3 significant) -> robust; SEI p 1 worse in 3 of 4 (2 significant),
  not Cold; strong physics worse only in B0005 and B0046 -> lambda finding battery-dependent (corrects 5co wording
  "light physics best in Reference").
- B0005 = special case (oracle best 8.45 vs B0006 2.35). Limit is information (3-4 training batteries cannot select
  settings for a new battery), not search (good configurations exist: B0006 best 2.35 vs default 3.27).
- Pending: B0018.

## 5cq. Colab GA study COMPLETE: five batteries, three groups (B0005, B0006, B0018 Reference; B0029 Hot; B0046 Cold)
- B0018: 87 configs x 3 seeds; winner h16|lr0.01|w1|p0.5|a0|it600 5.42 (sd 0.66) vs default 11.10 (0.70), rank 78/87;
  beyond noise (paired gain/SE very large); fitness vs own-future error Spearman 0.52; real error best / winner /
  default 2.90 / 3.73 / 5.37; top-10 vs bottom-10 4.20 vs 5.52 -> the one battery where GA selection transfers.
- Five batteries: GA beats default on its fitness beyond noise in 4/5 (not B0005); fitness predicts the real future in
  1/5 (Spearman 0.07 / 0.02 / 0.52 / -0.27 / 0.06); real error GA winner vs default: better 2 (B0018 clear, B0005
  within noise), equal 1 (B0029), worse 2 (B0006, B0046); mean 5.01 vs 5.11 -> equal on average.
- Balanced-design gene effects (5 batteries): accel on worse in 5/5 (4 significant) = most robust; SEI p 1 worse in 4/5
  (3 significant), Cold exception; lambda 3 worse in 3/5 (significant), never better -> light physics safe; lr 0.01
  better in 2 Reference batteries; small nets better in B0018 and Cold.
- Conclusions: GA value = stable settings + group-level physical evidence; per-battery selection transfers only when
  the test battery resembles its training batteries; B0005 special case (no configuration forecasts it).
- DECISION (proposed): GA-PINN stays the twin core with its current winners (scorecard 5by/5cc); Colab 3-seed winners not
  swapped in (no average gain on the real future); the Colab study = robustness analysis + limitations in the thesis.

## 5cr. Mission 2 accuracy assessment with GA-PINN (v5.42.0, origin 30 %, LOBO, on-demand GA per held-out battery)
- Mean RMSE (SOH pts) GA-PINN / fleet shape / linear trend / persistence: Reference (5) 7.22 / 4.34 / 6.13 / 13.13;
  Hot (4) 1.58 / 3.21 / 4.50 / 5.09; Cold (5) 6.58 / 3.20 / 11.67 / 10.70. Earlier same-protocol references (5h, 5bn):
  frozen MLP 4.47, DT 0.95, GP 2.77; PINN v1 5.61 / 0.98 / 8.44.
- Per battery GA-PINN: Reference B0005 10.16, B0006 3.39 (fleet 6.37), B0007 8.84, B0018 6.20, B0036 7.52; Hot 2.24 /
  0.96 / 1.13 / 1.99; Cold B0041 3.39, B0045 14.57, B0046 4.79, B0047 5.76, B0048 4.41.
- End-of-life error, Reference: GA-PINN 7.0 cycles vs fleet shape 32.0, linear trend 22.7 (better EOL placement).
- VERDICT: as a stand-alone forecaster from 30 % (no re-anchoring / retraining) GA-PINN is NOT the best: frozen data
  models remain the Mission 2 forecasters (and fleet shape is hard to beat in Reference / Cold). GA-PINN remains the best
  twin core (5cc). Consistent with the Colab study (5cq): per-battery GA selection does not transfer to a single
  long-horizon forecast. Optional: one combined assessment run (PINN v1, twin core, frozen winners + GA-PINN).

## 5cs. More GA search does not help (winner's curse); physics + data ensemble test prepared
- Simulation on the Colab raw files (2000 random subsets per size): as the search grows from 5 to 80 configurations the
  chosen configuration's FITNESS improves strongly (B0046 15.0 -> 8.7) while its REAL error stays flat or worsens
  (B0005 9.78 -> 9.89, B0006 3.73 -> 4.34, B0029 2.17 -> 2.27, B0046 4.52 -> 4.78; only B0018 4.08 -> 3.83); a random
  configuration is as good or better in 4 of 5 (B0006 3.65); the oracle best in the set keeps improving (B0006 2.70 ->
  2.36). -> Winner's curse with 3-4 validation batteries: more genes / larger search (incl. layers, lambda_bc of the
  professor's plan) would improve the fitness, not the forecasts. Bottleneck = information for selection.
- Next test (user agreed): physics + data ensemble, Mission2 walkthrough part 11: per held-out battery GA-PINN, PINN v1,
  GP, MLP; te.ensemble_trajectories weights 1/CV^2 from training batteries only; compared with every single model.
  Demo check OK (ensemble ties the best single, beats MLP / GP 4/4, GA-PINN / PINN v1 3/4).

## 5ct. Colab GA notebook: POOL option (pooled training AND validation) + built-in forecast plot
- POOL = True: training and GA cross-validation on every constant-condition battery except the test one (engine
  "pooled": True -> NASA_PHYSICS_GROUPS protocol cells); forecast start kept = the per-group run's start (comparable);
  files suffixed _pool (merge keeps pooled and per-group apart); self-tuning backtest skipped in POOL mode.
- RUN_GA = False: only the default + 36 balanced configurations (enough to answer: does pooled validation make the
  fitness predict the real error?). Suggested: B0006 and B0046, POOL = True, RUN_GA = False (~1.5-2 h each).
- Part 10 (forecast plot with end-of-life markers) now built in; follows POOL. Demo: both modes + merge OK.
- Prior evidence (5bp): pooled TRAINING gave no gain for PINN v2 (Reference 5.80 vs 5.83, Cold 7.38 vs 6.67).

## 5cu. Pooled GA test, B0006 (balanced 36 + default, 3 seeds; twin start; ~18 pooled batteries)
- Same 37 configurations, own group vs pooled: fitness-vs-real Spearman -0.19 vs -0.00; real error of best-fitness
  configuration 4.47 vs 3.53, random configuration 3.65 vs 3.23 (chosen worse than random in both), top-10 vs bottom-10
  3.90 vs 3.50 / 3.40 vs 3.38, oracle 2.35 vs 2.21; default 3.27 vs 3.78; ~29 s vs 12 s per evaluation.
- -> Pooled VALIDATION does not fix the winner's curse (other groups age differently: more batteries help selection only
  if they resemble the test battery). Pooled TRAINING: same configurations 3.65 -> 3.23 on average (pooled better on
  22/37) but default worse (3.27 -> 3.78); contrasts with PINN v2 pooled (5bp, Reference 5.80 vs 5.83, Cold worse
  6.67 -> 7.38). Pending: B0046 (Cold) to decide.

## 5cv. Professor's idea: GA-PINN + GPR hybrid (walkthrough part 13)
- Professor (meeting): GPR with an optimised loss, combined with GA-PINN, "fit" B0005 with 0.996 (likely R^2 of a fit
  / interpolation; R^2 0.996 on B0005 ~ RMSE 0.55 SOH pts given SOH spread ~8-9 pts). Our B0005 numbers are forecasts
  from 30 % of life (8-10 pts): different questions. Code not available -> built from the description.
- Part 13: per held-out battery: GA-PINN (GA search 6x4 at the origin, or PINN twin core if USE_GA = False); GPR
  trained on the PINN's errors on the OTHER batteries' futures (each forecast from the same life fraction, features
  horizon, PINN prediction, SOH and slope at the origin); kernel Const*RBF(ARD)+White fitted by minimising the
  negative log marginal likelihood, 5 restarts; hybrid = PINN + GPR correction, 80 % band from the GPR std. Scores:
  RMSE, R^2 on the future, EOL error vs GA-PINN alone and GP alone. Second cell: GPR fit to one battery with 20 %
  random cycles held out -> interpolation R^2 (explains 0.99-type numbers).
- Demo check: hybrid 0.41 vs PINN 0.67 vs GP 1.32 (R^2 future 0.994); fit R^2 0.999. NASA pending.

## 5cw. GPR / GA-PINN step-by-step study in Colab (B0005, none of its capacity used) + professor's 0.996 explained
- Health indicators per discharge cycle (from the raw parquet): t(3.8->3.5 V), mean discharge V, temperature rise,
  load-step R, CC / CV charge times; proxies = time to peak temperature and discharge time (= capacity at CC; t_peak_T
  correlates ~1 with t_discharge).
- GPR (Matern, ARD, ML-II with restarts), trained on B0006/B0007/B0018 (+B0036, the only 2.7 V cut-off like B0005):
  absolute indicators 0.956; relative to own first 5 cycles + smoothed (chosen by LOBO CV on training batteries only)
  0.957 -> with B0036 0.9636; backward feature selection by CV RMSE kept t(3.8-3.5 V), t_CC, t_CV -> 0.9630
  (RMSE 1.97). Remaining error: late-life plateau (~75 % vs 70 %). Lessons: R^2 is a poor CV metric for B0036 (little
  SOH variation) -> RMSE; B0005's 2.7 V cut-off outside the 2.2-2.5 V of its campaign.
- Professor's 0.996 explained: with proxies, absolute 0.9816 (t_peak_T most important), relative + smoothed 0.9905
  (RMSE 0.99) -> his result very likely used discharge-time-type features.
- GA-PINN alone (JAX, cycle clock with nominal 2 Ah, conditions only, zero-shot): 0.7125 (default 0.7095), GA chose
  p 1 -> steep early, too flat late. GPR + GA-PINN, residual correction (chosen by LOBO CV): R^2 0.9753, RMSE 1.61,
  fixes the late-life plateau -> professor's combination validated honestly.
- Improved PINN (measured resistance in the loss for all batteries incl. target, shared physics, LOBO GA): mechanism
  recovery test (8 simulated batteries, known splits, 1500 iterations): all variants ~31 share pts, 5/8 dominant ->
  the PINN collapses the loss into SEI (rates of other mechanisms can shrink to ~0 at no cost; resistance cannot pin
  L_SEI because g is learned). Explanation stays with the shared physics (1.1 pts, exact solutions); PINN / GPR for SOH.

## 5cx. Part C: physics-structured GA-PINN + GPR + combination for all 18 batteries (Reference, Hot, Cold, Pulsed)
- Physics core: exact solutions, parameters shared by all batteries (fitted on all groups together so mechanisms are
  separable): SEI dL/da = f/(L+delta)^p, f = k e^{b_T x_T} (closed form for any p); plating k_pl e^{a_pl x_I} x_C a
  (cold only, by construction); LAM k_LAM e^{a_I x_I + a_V x_V} a; SOH = (1-s-pl)(1-m); R/R0 = 1 + g s; least squares
  on SOH/0.01 and R/0.03, multi-start. Bounded neural correction (MLP, clip +-bound) on the total SOH.
- GA genes: hidden, learning rate, bound, SEI law p, iterations; fitness = LOBO RMSE over all batteries + 0.5|late
  bias| + time (one setting for all batteries: slight selection optimism, stated).
- Recovery test (8 simulated batteries, same law forms): 0.10 share pts, all dominant correct (earlier PINN ~31) ->
  identifiable when the laws are right (does not prove the laws).
- GPR (relative + smoothed t(3.8-3.5 V), t_CC, t_CV + cycle, Matern 1.5) and GPR + physics residual correction, each
  battery held out within its group; physics predictions of training batteries from LOBO fits. Mechanism shares from
  the core fitted on all 18. Demo: all cells run; combination can be worse than GPR alone (group-dependent). NASA pending.

## 5cy. Data cleaning (Colab cell 2b) and final Part C results (all groups, none of the target's capacity used)
- Cleaning: aborted discharges (V_min > battery median + 0.15 V) dropped; invalid first cycles (capacity < 0.93 x
  stable level of cycles 6-25) dropped; Hampel filter on capacity (9 neighbours, 4 robust std, >= 3 % of stable
  level) drops far random points (regeneration kept); Hampel on indicators (>= 5 % from local median) replaces values
  by the local median; start capacity = median of the first 10 kept cycles. Tested with planted bad points: exact.
- B0005 after cleaning: GPR alone 0.985 (was 0.963); GPR + GA-PINN 0.977 (was 0.975); GA-PINN alone 0.695 (default
  0.706). Combination no longer helps once the GPR's late-life gap is gone.
- Part C after cleaning, GPR alone RMSE: Cold ~3.0 (B0046 1.62, B0047 1.84, B0048 2.29, B0045 6.35; was 10.3), Hot
  ~1.2 (0.56-1.97), Pulsed ~0.6 (0.35-1.08; B0026 dropped < 25 cycles), Reference ~3.5 (B0005 1.43, B0007 1.97, B0018
  1.86, B0006 4.92, B0036 7.28). Physics-structured GA-PINN 1.3-14 pts; combination better than GPR only for B0006.
- Mechanism shares (SEI / plating / LAM) shift 20-30 pts after cleaning (Hot 75-86 -> 47-63 SEI; Reference 41-55 ->
  18-26 SEI; Cold plating 60-77 -> 37-53, LAM 40-57). Robust qualitative ranking: SEI share rises with temperature;
  plating only in the cold (by structure); LAM large at room temperature. Quantitative shares uncertain ~+-20 pts.
  Physics fit: Cold now 0.7-0.9; Pulsed poor (-0.8 to -11: the pulsed load is not represented).
- FINAL: SOH = honest GPR on cleaned indicators; mechanisms = shared physics (qualitative ranking, with fit R^2 as
  confidence); professor's combination useful only when the data-driven model has a systematic gap.

## 5cz. FINAL Mission 3 with the GA-PINN twin core (v5.42.0, maintenance + lifecycle CSVs)
- Share of the ideal benefit (perfect knowledge vs run to EOL), group sums: Reference twin alone 0.947 (0 unplanned),
  twin + age limit 0.895 (0), best fixed age 0.583 (1), run to EOL 0 (3); Cold twin alone 0.389 (3 unplanned),
  twin + age limit 0.746 (1), best fixed age 0.706 (1), run to EOL 0 (5). Twin replacements: B0005 122 (EOL 126),
  B0018 92 (98), B0006 94 (110). Recommended rule: twin + age limit (all groups).
- Lifecycle profit per hour (group means), do nothing / usual / twin replacement / twin strategy / upper bound:
  Reference -0.687 / 0.092 / 0.278 / 0.304 / 0.341; Cold -1.320 / -0.672 / -0.886 / -0.862 / -0.249 (Cold loses money
  under these economics; twin does not help there). Chosen operation: 2.7 V cut-off for all.
- vs frozen core (5be): Reference 96 / 93 % -> 95 / 90 %; Cold 34 / 62 % -> 39 / 75 % (but best fixed age also moved
  54 -> 71 %, independent of the core -> other settings differed; compare within run). Decisions ~equal, consistent
  with the scorecard.

## 5da. v5.43.0: SOH ESTIMATION in the platform (engine section 59 + Models page section)
- User request: estimation (today's SOH from today's measurements) next to forecasting, GA-PINN usable for both;
  estimator switchable later as results improve.
- Engine: estimation_table (honest indicators from the cycle table: charge between 3.8 and 3.5 V = qv_18 - qv_06,
  t_cc_s, t_cv_s; Hampel replacement >= 5 %; rolling median 5; relative to own first 5 cycles; cycle clock with nominal
  2 Ah; operating conditions); training batteries cleaned with their SOH (invalid first cycles < 93 % of stable level,
  outliers, Hampel >= 3 pts); the battery being estimated is BLIND: nothing selected from its SOH, SOH checks only set
  score_ok. estimate_soh(kind = "GPR" | "PINN-est"): GPR Matern 1.5 ARD + white, restarts 3; estimation PINN (engine
  Tensor/Adam): inputs [a, xT, xC, xI, xV, 3 indicators], no battery code, shared rate laws enforced on all measured
  rows (target: measurements only), genes as GA_PINN_GENES. ga_estimation: ga_pinn_run, fitness = LOBO over training
  batteries RMSE + 0.5|late bias|. CACHE_VERSION = 5.42.0 (existing results stay valid; ENGINE_VERSION 5.43.0).
- tests/test_estimation.py (6): relative indicators start at 1; finite estimates; TARGET BLIND (SOH scaled,
  oscillated, far points, low first cycles, outlier flags changed -> estimates identical on every cycle) for GPR and
  PINN-est; GA runs. A first version leaked (SOH-based row selection set the indicator baseline) -> fixed, test stricter.
- App: Models & forecasting, first section "Mission 2 · SOH estimation": group, battery, estimators (GPR, estimation
  PINN default, estimation GA-PINN with GA on the training batteries), one battery (plot + band + table) or every
  battery of the group (LOBO table, CSV); "estimator for the twin's field mode" selector kept for the next release.
- App-only follow-up: the estimation GA-PINN search is resumable (every PINN training saved as it finishes through
  est_cached; time budget per press, default 15 min; progress line); NASA cost ~16 settings x 4 training batteries =
  ~64 trainings (30-60 min on Streamlit Cloud). Demo: 2-min budget, press 1 stopped, press 2 finished in 16 s with the
  same winner as an uninterrupted run.
- 42 tests pass; all pages; md_check 0. Next: field-mode anchor with the chosen estimator + update study; estimate-vs-
  forecast fault indicator (validate on simulated faults); platform restructure.

## 5db. v5.43.1: estimation PINN fix (NASA B0005: GPR 0.978 / RMSE 1.52 vs estimation PINN default -0.041 / 10.49)
- Diagnosis: the network received the per-battery operating conditions as inputs; in Reference only the cut-off
  differs between batteries, and B0005 shares 2.7 V with B0036 alone (which barely fades: 87 %) -> the network could
  recognise "the 2.7 V battery" and copy B0036's curve (~10 pts error). The GPR never sees the conditions.
- Fix (ESTIMATION_VERSION 2): network inputs = clock + measured indicators only; conditions enter only the rate laws.
  Demo: S001 0.829 -> 0.923, S008 0.814 -> 0.972 (GPR -1.463 / 0.930); shortcut test (S001 given S002's conditions,
  S002 ages differently): 0.932. App: estimator cache keyed by ESTIMATION_VERSION (old PINN results never reused).
- 42 tests, all pages, md_check 0. NASA rerun pending (B0005 and all groups).

## 5dc. Part D results (NASA, cleaned, 16 batteries, none of each battery's capacity used)
- RMSE (SOH pts) median / mean / worst; B0005: own group 1.73 / 2.27 / 7.28 (B0036), 1.43; all groups (+ conditions,
  1000 pooled cycles) 1.53 / 2.03 / 3.92, 3.92; BOTH GPRs weighted by their own predictive std 1.61 / 1.84 / 4.25,
  1.32; GPR (all groups) + physics core residual, uncertainty-weighted 1.09 / 1.90 / 4.23, 4.22.
- Group means (own / all / both / +physics): Reference 3.49 / 1.95 / 2.12 / 2.28; Hot 1.20 / 1.14 / 1.14 / 0.93;
  Cold 3.03 / 3.23 / 2.99 / 3.18; Pulsed 0.68 / 1.74 / 0.75 / 0.88 (pulsed load: other groups mislead).
- All groups fixes out-of-range batteries (B0006 4.92 -> 2.45, B0036 7.28 -> 1.30, B0031, B0045); own group better on
  typical ones (B0005, B0029, Cold B0046-48, Pulsed). Both-GPR weighting: lowest mean, best B0005, worst case 7.28 ->
  4.25; partial switch only (own-group GPR overconfident when extrapolating: B0036 3.03 vs 1.30).
- Recommended platform estimator: both GPRs, uncertainty-weighted (principled, nothing tuned on the target).

## 5dd. GA-PINN for ESTIMATION reaches the professor's level (none of the target's capacity used, no proxies)
- Platform (v5.43.1, fixed estimation PINN, GA on the training batteries): B0006 R^2 0.991, RMSE 1.16 (GPR own group
  0.845 / 4.92: B0006 fades below its training range); GA winner h32 lr0.01 lambda 1 p1 accel 900.
- Colab Part F (JAX; inputs = GPR inputs; conditions only in the physics; GA minimises LOBO validation RMSE on the
  training batteries; final = 3-seed mean; combination weights 1/validation RMSE^2): B0005 GA-PINN 0.9887 / RMSE 1.10,
  GPR 0.9810 / 1.43, GA-PINN + GPR 0.9873 / 1.16 (weights 0.66 / 0.34). GA winner h64 lr0.01 lambda 0 p1 it1000
  (validation RMSE 1.76; GPR 2.43).
- Reading: the physics-SHAPED network (states from 0, non-negative, multiplicative SOH) carries the gain and can
  extrapolate below the training range; the best residual weight differs per battery (0 or 1).
- Colab Part F, B0006: GA-PINN 0.9904 / RMSE 1.22, GPR 0.8452 / 4.92, combined 0.9850 / 1.53 (weights 0.82 / 0.18;
  validation RMSE 1.94 vs 4.21). Independent implementations agree on B0006 (JAX 0.990, engine 0.991).
  Batch notebook GA_PINN_vs_GPR_batch.ipynb (B0007, B0018, B0029, B0030, B0046, B0047, B0025, B0028; GA 4 x 3) running.
- Platform, B0007: estimation GA-PINN R^2 0.995 / RMSE 0.62 (GPR own group 0.947 / 1.97); GA winner identical to
  B0006 (h32 lr0.01 lambda 1 p1 accel 900), chosen independently -> stable setting for Reference.
  Reference so far (GA-PINN vs GPR RMSE): B0005 1.10 vs 1.43, B0006 1.16-1.22 vs 4.92, B0007 0.62 vs 1.97. Professor's GA-PINN
  0.96-0.97 exceeded honestly. Next: GA-PINN estimation on all batteries; make it the platform's default estimator.

## 5de. v5.44.0: Mission 3 OPERATING WINDOW (engine section 60 + Operations page section 1b)
- User goal: test a battery over a range of operating conditions and pick the candidates that last longest before
  end of life, starting from today's state (measured or ESTIMATED by the estimation GA-PINN).
- Engine m3_operating_window(h, cell, n_now, soh_now, soh_eol, T, I, V grids, depth, groups): shared-physics
  mechanism laws (identifiable) for each candidate (_whatif_cd), scaled by m3_fade_scale; today's damage
  (1 - soh_now)/k placed on each candidate's loss curve (same physics-equivalent damage), ageing continues from there
  with throughput per cycle scaled by the measured Ah to the candidate cut-off; Wh per cycle from the depth table
  (pooled median Wh / Ah to 2.7 V x own Ah to 2.7 V). Outputs remaining cycles, energy to EOL, vs own, flags
  (combination not tested, value not tested, condition not identifiable / limited, no measured energy).
  Assumption stated: equivalent-damage transfer between conditions; physics used for differences.
- App (Operations & control, after stage 1): battery, today's cycle, today's state (measured / estimated from the
  Models page estimation, best available GA-PINN > PINN > GPR), grids, criterion (most energy / longest life),
  "rank only candidates supported by the data" (default on); metrics, top-8 table, trajectories of the top 3 vs own,
  heat map current x cut-off at a temperature, CSV.
- tests/test_window.py (2): table + flags; more damage today -> never more remaining life. 44 tests, all pages,
  md_check 0.

## 5df. Estimation batch: GA-PINN vs GPR vs combined on 10 batteries (Colab, JAX, GA 4 x 3 for the batch)
- RMSE (SOH pts) GA-PINN / GPR / combined: B0005 1.10 / 1.43 / 1.16; B0006 1.22 / 4.92 / 1.53; B0007 1.34 / 1.97 /
  1.49 (platform 6x4: 0.62); B0018 2.50 / 1.87 / 2.28; B0029 0.65 / 0.56 / 0.59; B0030 0.80 / 0.70 / 0.75; B0046 1.74
  / 1.62 / 1.57; B0047 4.59 / 1.84 / 2.48; B0025 0.50 / 0.45 / 0.45; B0028 1.44 / 0.41 / 0.81. Batch-of-8 means
  1.70 / 1.18 / 1.30.
- GA-PINN better on 3/10 (B0005-07, Reference: long lives, 4 training batteries, B0006 below its training range);
  GPR better on 7/10 (Hot, Cold, Pulsed: 2-4 training batteries, short lives -> the network overfits, the GPR's
  smoothness prior copes). Failures: B0028 collapse in the last 2 cycles (clock beyond the training lives),
  B0047 too high late (noisy Cold indicators).
- B0018 with the full GA (6 x 4, Colab): GA-PINN 0.929 / 2.25 (batch 4x3: 2.50), GPR 0.951 / 1.86, combined 0.937 /
  2.12 (weights 0.79 / 0.21 from training validation 2.16 vs 4.17: the GPR's LOBO failures on B0006 / B0036 drive
  the weights). Both models overestimate cycles 35-105 the same way -> an input-information limit, not a model one.
  Reference: GA-PINN better on 3 of 4 (B0005, B0006, B0007).
- Platform (6 x 4 GA), B0005: estimation GA-PINN 0.990 / RMSE 1.01 (Colab 0.989 / 1.10), GA winner h16 lr0.01
  lambda 3 p0.5 no accel 900 (Colab: lambda 0, p1) -> accuracy robust to the settings. Platform Reference so far:
  B0005 1.01, B0006 1.16, B0007 0.62 (GPR 1.43 / 4.92 / 1.97).
- Combination (weights = 1 / validation RMSE^2 on the training batteries) is the robust choice: worst 2.48 vs 4.59,
  keeps the B0006 gain. Fixable: bound the clock input to its training range; larger GA (6 x 4).

## 5dg. v5.44.1: operating-window fixes (NASA test on B0006 showed only 2.7 V and identical energy per current)
- Bug: _window_depth_factors read the depth table's row number as the battery (column is 'battery') -> only 2.7 V
  rows matched -> every other cut-off lost its measured energy and fell out of map and ranking (demo has 2.7 V only).
- Energy per cycle now from the batteries run at the same discharge current (measured), pooled otherwise: higher
  current -> less Wh per cycle. Ageing vs current is still flat where the physics' current sensitivity is limited
  (flagged).
- Colab notebook Operating_window_Colab.ipynb: START "new" (whole life from cycle 0) / "measured" / "estimated";
  SCOPE "tested" (only combinations NASA ran: inside the training data) / "supported" / "all"; top N (5) by total
  energy to end of life (default) or life. Reproduces the platform exactly (demo: 110 cycles 629.0 Wh / best 114
  cycles 649.3 Wh in both).
- Tests: depth factors with several cut-offs and currents (4 window tests); suite and pages pass.
- NASA B0006, new battery (START new), SCOPE tested (5 of 60 candidates), most energy to EOL: #1 32 C · 2 A · 2.7 V
  (~127 cycles, 751 Wh) > own 32 C · 2 A · 2.4 V (~119, 708 Wh) > 35 C · 4 A · 2.4 V (~116) > 54 C · 4 A · 2.4 V
  (~102) > 9 C · 1 A · 2.7 V (~32: plating). Map at 32 C: energy rises with shallower cut-off in every current
  column (2.0 V 644 -> 2.7 V 751 Wh at 2 A); +6 % energy, ~+7 % life vs own = same direction as Mission 3 stage 2.
  Caveat: 1 A energy factors come from the cold batteries only (current confounded with temperature).
- Notebook fix: the battery's own conditions are always added to the grid (own row was missing when the grid
  lacked its 2.4 V cut-off).

## 5dh. GENERAL operating window (Colab cells 9-11, engine 5.44.1): 504 scenarios x 20 batteries = 10,080 lives
- Scenarios inside NASA's ranges: T 9-54 C (9 values), I 1-4 A (7), cut-off 2.0-2.7 V (8); every battery simulated
  as new; energy relative to each battery's best, then averaged. 9 s.
- Identifiability: cut-off identified; temperature limited; discharge current limited; charge current not identifiable.
- Top: 20.2 C · 2 A · 2.7 V, mean relative energy 0.972, median 219.5 cycles (10-90 %: 137-534), 1,033 Wh; top 10 %
  for 95 % of batteries (worst battery 0.44). Then 20 C · 2.6 V, 26 C · 2.7 V, 20 C · 1 A, 20 C · 1.5 A.
- Robust: shallower cut-off better at every T and I (31.5 C: 2.0 V ~172 -> 2.7 V 206 cycles, +20 %). Temperature:
  9 C 40-46 cycles (plating), optimum ~20-26 C (interpolated, sensitivity limited -> state "20-30 C"), 54 C 136-178
  (-15-20 % vs 31.5 C). Current: no reliable conclusion (limited ageing effect; energy factors per current come from
  different groups -> confounded with temperature). Large battery spread in absolute life: ranking general, life not.

## 5di. Estimation: B0036, B0029 (platform) and the v2 batch (input clipping fix)
- B0036 (Colab, full GA; a plot titled B0029 is the same B0036 run, TEST_F changed before replotting): GA-PINN 0.937,
  GPR -3.549 (collapses to 83 %: other campaign), combined 0.423 (validation weights misled).
- B0029 platform (Hot): GA-PINN 0.872 / RMSE 1.25 vs GPR own group 0.975 / 0.56.
- v2 batch (inputs clipped to the training range; training-only standardisation), R^2 GA-PINN before -> after:
  B0028 -0.284 -> 0.483, B0047 0.642 -> 0.721, B0018 0.912 -> 0.931, B0025 0.813 -> 0.778, B0007 0.975 -> 0.956, B0029
  0.966 -> 0.963, B0030 0.948 -> 0.942, B0046 0.949 -> 0.929. Combined after: 0.955 0.943 0.971 0.952 0.948 0.921 0.880
  0.860 (mean 0.929 vs GPR 0.935; worst 0.86 vs 0.60 before).
- 14 batteries: GA-PINN wins Reference 4/5 (B0005, B0006, B0007, B0036 incl. the two the GPR fails), GPR wins Hot,
  Cold, Pulsed (2-4 training batteries, short lives). Conclusion: physics helps most where data cannot cover the
  battery (out of range, other campaign).

## 5dj. v5.45.0: the Colab estimation improvements in the platform (ESTIMATION_VERSION 3)
- Platform NASA results before this release (RMSE, GPR vs estimation GA-PINN / PINN): B0029 0.56 vs 1.25; B0030
  1.45 vs 3.89 (GA-PINN flat after the cycle-17 bump: extrapolation); B0031 1.48 vs 0.92; B0032 1.43 vs 0.76; B0041
  6.00 vs 11.93 (only cycles 46-67 valid; 4 A, unlike its group); B0045 2.15 vs 2.19; B0046 2.68 vs 3.46; B0047 3.11
  vs 2.27; B0048 2.09 vs 1.87. Low R^2 in Hot = little SOH variation (100 -> 89 %); far points are mostly real
  regeneration bumps.
- Engine: ga_pinn_run(genes=, start=) (defaults unchanged: the twin core's GA is identical); EST_GENES (hidden 16-64,
  lambda 0/0.1/0.3/1/3, p, accel, iterations 600/1000/1500), EST_START; _est_pinn: training-only scaling, inputs
  clipped to the training range, cosine learning-rate decay; estimate_soh PINN-est params "seeds" (final estimate =
  mean of seeds; GA candidates 1 seed); second score excluding far points (Hampel window 7, k 3, >= 1.5 pts; scoring
  only): rmse_clean, r2_clean, n_far.
- App: estimator "GA-PINN + GPR" weighted 1/validation RMSE^2 (GA winner's LOBO RMSE on the training batteries,
  GPR's LOBO RMSE); final GA-PINN and default PINN average 3 seeds; table with far-point-excluded scores; window
  start prefers GA-PINN + GPR. 46 tests, all pages, md_check 0.

## 6. Next steps
1. Writing (all six plan questions answered, 5bg): Mission 2 chapter -> physics chapter -> Mission 3 chapter ->
   Mission 1 dataset report + literature survey -> final presentation (demonstration case: B0006).
2. Optional engine release (one rebuild): lifecycle strategy "twin + age limit + chosen operation" in the app.
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
