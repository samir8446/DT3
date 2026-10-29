# STATUS · handover for a new conversation · engine/app v5.37.0 · 2026-10-29

Read this first. It replaces every earlier STATUS.md. Full change history: README.md.

## 1. Project in one paragraph
Internship (Yokogawa, UM6P collaboration): a self-updating digital twin for the NASA Ames 18650 Li-ion ageing data
(LiCoO2/graphite, 34 cells, 4-44 C, 1-4 A). Three missions from the research plan: M1 health indicators and degradation
mechanisms; M2 self-updating twin (can degradation be predicted, which variables are informative, how often to update);
M3 integrated operation and maintenance. Deliverable: a Streamlit app (app.py) + a tested engine (twin_engine.py).

## 2. Files and versions (always deliver both .py files from the same release)
- twin_engine.py  ENGINE_VERSION = "5.37.0"   (all computation; no Streamlit)
- app.py          REQUIRED_ENGINE = "5.37.0"  (exact version guard + required-function check)
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
