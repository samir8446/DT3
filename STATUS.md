# STATUS · handover for a new conversation · engine/app v5.16.1 · 2026-10-14

Read this first. It replaces every earlier STATUS.md. Full change history: README.md.

## 1. Project in one paragraph
Internship (Yokogawa, UM6P collaboration): a self-updating digital twin for the NASA Ames 18650 Li-ion ageing data
(LiCoO2/graphite, 34 cells, 4-44 C, 1-4 A). Three missions from the research plan: M1 health indicators and degradation
mechanisms; M2 self-updating twin (can degradation be predicted, which variables are informative, how often to update);
M3 integrated operation and maintenance. Deliverable: a Streamlit app (app.py) + a tested engine (twin_engine.py).

## 2. Files and versions (always deliver both .py files from the same release)
- twin_engine.py  ENGINE_VERSION = "5.16.1"   (all computation; no Streamlit)
- app.py          REQUIRED_ENGINE = "5.16.1"  (exact version guard + required-function check)
- tests/test_twin_engine.py: 80 tests (passing at v5.11.0; not rerun since, file not in the Project) +
  tests/test_cv_selection.py (5) + tests/test_frozen_live.py (10), passing at v5.15.0
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

## 6. Next steps (full-signal physics phase)
1. USER: upload v5.16.1 (twin_engine.py, app.py, tests/test_dqdv.py) to GitHub; CI green.
2. PyBaMM benchmark (scripts/pybamm_benchmark.py, in progress, 5u): fix cold calibration, the 0.32 Ah cell, the
   identifiability bug, add LAM of lithiated material; then the full design: LCO/graphite cells with known LLI, LAM_NE, LAM_PE (each 0-20 %, and
   mixtures); each cell fully charged (CC-CV to 4.2 V) then discharged at NASA rates (1 A, 2 A, 4 A scaled) and
   temperatures (4, 24, 43 C); compute the same dQ/dV features (main peak height/width/area/position, shoulder,
   charge/discharge shift). Result: which features identify which mode at NASA rates.
3. Only validated features go back into PINN v2 (dq_weight > 0 with the validated mapping); attribution with
   honest uncertainty where unidentifiable (cold).

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
