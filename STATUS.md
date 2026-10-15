# STATUS · handover for a new conversation · engine/app v5.14.1 · 2026-10-14

Read this first. It replaces every earlier STATUS.md. Full change history: README.md.

## 1. Project in one paragraph
Internship (Yokogawa, UM6P collaboration): a self-updating digital twin for the NASA Ames 18650 Li-ion ageing data
(LiCoO2/graphite, 34 cells, 4-44 C, 1-4 A). Three missions from the research plan: M1 health indicators and degradation
mechanisms; M2 self-updating twin (can degradation be predicted, which variables are informative, how often to update);
M3 integrated operation and maintenance. Deliverable: a Streamlit app (app.py) + a tested engine (twin_engine.py).

## 2. Files and versions (always deliver both .py files from the same release)
- twin_engine.py  ENGINE_VERSION = "5.14.1"   (all computation; no Streamlit)
- app.py          REQUIRED_ENGINE = "5.14.1"  (exact version guard + required-function check)
- tests/test_twin_engine.py: 80 tests (passing at v5.11.0; not rerun at v5.12, file not in the Project) +
  tests/test_cv_selection.py: 3 tests (v5.12), passing
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
- B0041 (mixed 4 A / 1 A) was in Cold only because the Reference-condition toggle was on; from v5.14 the
  assessment groups follow the recorded conditions, so Cold = B0045-B0048 and B0041 is in Mixed.
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

- Cold rerun (2026-10-14) identical to the first one to 1e-15 (still B0041, EOL at cycles 2-44): app version or
  EOL setting not changed. v5.14.1 writes version and settings into the CSVs.

## 6. Next steps (in order)
1. USER (v5.14.1; check the version in the CSV columns; 'Best 2 per level', locally, to completion, without touching the page): rerun Cold (now
   B0045-B0048, EOL 80 % of initial); run Pulsed load (40-50 %) and Mixed conditions (30 %; reference condition is
   applied automatically), both exploratory. Reference and Hot v5.13 results stand. Send both CSVs per group.
2. Star agreement so far (v5.13): good in Hot (ML measured 1.5/7), poor in Cold (ML forecast 4.8/7); decide per
   group whether the CV star can be trusted.
3. Freeze the best model per group (model card: model, settings, scores).
4. Self-updating twin: add the frozen model to the Live twin (measured mode re-estimates SOH each cycle, forecast
   restarts from it); compare with EKF/PF; update-frequency study (last open M2 question).
5. M3: end-of-life prediction with band -> Operations (maintenance timing, what-if conditions).
6. Report: tables/figures per group, informative variables, mechanisms, limitations.
Later: alkaline water electrolyser version of the platform (prompt written; skills battery-digital-twin,
degradation-forecasting, streamlit-science-app packaged as .skill files).

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
