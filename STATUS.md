# STATUS · handover for a new conversation · engine/app v5.12.1 · 2026-10-14

Read this first. It replaces every earlier STATUS.md. Full change history: README.md.

## 1. Project in one paragraph
Internship (Yokogawa, UM6P collaboration): a self-updating digital twin for the NASA Ames 18650 Li-ion ageing data
(LiCoO2/graphite, 34 cells, 4-44 C, 1-4 A). Three missions from the research plan: M1 health indicators and degradation
mechanisms; M2 self-updating twin (can degradation be predicted, which variables are informative, how often to update);
M3 integrated operation and maintenance. Deliverable: a Streamlit app (app.py) + a tested engine (twin_engine.py).

## 2. Files and versions (always deliver both .py files from the same release)
- twin_engine.py  ENGINE_VERSION = "5.12.1"   (all computation; no Streamlit)
- app.py          REQUIRED_ENGINE = "5.12.1"  (exact version guard + required-function check)
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

## 6. Next steps (in order)
1. USER: rerun the accuracy assessment per group with v5.12 ('Best 2 per level' mode, locally), to completion, baselines included:
   Reference 30 %; Hot 30 %; Cold B0045-B0048 30 % with EOL 80 % of initial; Pulsed 40-50 %;
   Mixed 30 % with Reference-condition SOH on. Then send scoreboards/CSVs.
2. Check on real data with the new star-agreement table: star/self-optimisation agree with test results? Bayesian Ridge CV no longer absurd?
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
