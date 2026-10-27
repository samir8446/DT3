# Self-updating digital twin for Li-ion battery diagnostics (v5.30.1)

Streamlit platform and Python engine for the internship study on NASA Ames 18650 LiCoO₂/graphite ageing
data. A physical cell and its virtual copy are synchronised cycle by cycle to diagnose ageing, forecast
remaining life and optimise operation and maintenance.


## Integrated twin fixes after the first NASA run (v5.30.1)

1. Status Watch uses the median forecast end of life (a wide band's early edge kept B0005 on Watch for 60 cycles).
2. End of life at or above a battery's initial capacity (Cold with 1.40 Ah: 158 %) is left undefined, with a warning
   and a one-click switch to 80 % of initial capacity.
3. Chart: no forecast end-of-life marker after end of life; SOH axis from valid cycles; forecast shown ahead of today;
   forecast line and band without markers.
4. Tests updated (9 pass); AppTest of the page with the warning and the switch.

## Integrated twin page (v5.30.0)

1. New page "Integrated twin" (Mission 3, option B): one battery, one "Today" cycle; status with reason; forecast
   (frozen model, re-anchored to measured capacity) with band and end of life; detector verdict per cycle and event
   log; where the capacity went (hybrid v2) with the v3 model-form check; what-if limited to conditions NASA varied
   and values NASA tested; CSV timeline. Engine section 55: integrated_snapshot, integrated_timeline,
   whatif_tested, whatif_identifiability, whatif_projection, hybrid_portable, FROZEN_SCORES.
2. Fix: the hybrid result held a closure that cannot be pickled, so st.cache_data / the result backup raised and the
   Models button "Fit the hybrid to the NASA batteries" failed. The cached hybrid now omits it (hybrid_portable).
3. tests/test_integrated_twin.py (9): snapshot causal, status levels, timeline shares, what-if refuses untested
   or non-identifiable conditions, limited flag, hybrid picklable, version guard. AppTest of the page and all pages.

## PINN with all inputs (v5.29.0)

1. PINN v1 saw less than the Level 2/3 models (no cut-off voltage, CV-charge time, dQ(V), temperature rise; SOH and
   resistance only as fitted targets). New "PINN · all inputs" (pinn:PINN-full): same physics equations as PINN v1;
   network inputs = throughput + conditions + the battery's measured state at the origin as constants (the Level-2
   approach): SOH and trend, resistance and trend, CV-charge time and trend, discharge time, CC-charge time, mean
   voltage, energy efficiency, temperature rise and trend, cell temperature, C-rate, charge current, cut-off voltage,
   dQ(V), EIS Re and Rct (pinn_state_raw, PINN_FULL_FEATURES; standardised across the batteries of each fit, missing
   -> median). Every battery's state is taken at the SAME origin cycle n0. Plus its no-physics twin
   (pinn:PINN-full-ablation). PINN v1 unchanged (fingerprint identical).
2. Accuracy assessment: the physics-forecaster levels (shared physics v2/v3, hybrids, ageing speed) are removed and
   replaced by "Level 4 · PINN (original inputs) + ablation" and "Level 4 · PINN all inputs + ablation". The shared
   physics and hybrid sections (explanation, option B) stay. Models view: the PINN section defaults to all inputs.
3. Demo (Reference, 30 %): PINN all inputs 0.68, fleet shape 1.04, without physics 1.26 SOH points (demo only).
4. tests/test_pinn_full.py (state vector causal, forecasts causal, assessment); 44 tests pass; AppTest; all pages.

## End of life in the detection; removed cycles shown (v5.28.0)

1. Engine: eol_cycle_from_replay (first cycle whose 3-cycle median SOH is at or below the end-of-life threshold, causal);
   anomaly_events adds an EOL event ("End of life reached ... plan the replacement"); anomaly_cycle_status adds "Past
   end of life" (alarms keep priority). (A first version reused the name eol_crossing, which already existed and is
   used by the assessment: caught by 3 failing tests, renamed.)
2. Live detection replay: dashed end-of-life line and a marker at the crossing; verdict box says "past end of life";
   cycles removed by the data cleaning (the regeneration cycle after a rest + the 2 transient cycles after it, and
   invalid measurements) are shown as hollow circles with an explanation, instead of gaps.
3. 40 tests pass; AppTest of the replay at the end of life; all 7 pages.

## Faster live twin (result backup) and live detection replay (v5.27.0)

1. Live twin: "Retrain every" now defaults to never (the update study's best and cheapest setting: measured-capacity
   anchor without retraining); 5-50 still selectable.
2. Result backup (backed() in app.py): heavy results (live replays, frozen streams, multi-model frames, anomaly scan,
   shared physics, hybrid) are saved to results/cache/, keyed by engine version + a fingerprint of the input CONTENT
   (not paths) + settings; a new session reloads them. Demo: Live twin 26 s first time, 2 s in a new session.
   Sidebar "Result backup": download all saved results as a zip; upload the files to results/cache/ on GitHub so the
   app starts warm after restarts. A new engine version rebuilds them.
3. Operations: live detection replay (play / step / back / reset / speed): SOH points appear cycle by cycle coloured by
   the detector's verdict at that cycle (normal, regeneration, FLEET, OWN, common test event, FLEET + OWN), the
   indicators grow below, a verdict box shows the current state, and "What happened" lists the events up to the
   cursor. Engine section 54: anomaly_cycle_status (causal per-cycle labels).
4. 39 tests pass; AppTest: live twin (first and new session), live detection replay, all 7 pages.

## Live anomaly detection, common-cause events, new page order (v5.26.0)

1. FLEET indicator needs 12 cycles of history (was 8): removes the NASA B0034 false alarm (early kappa swing); the
   2.5x-fast validation batteries are still caught (cycle 16).
2. Engine section 53: anomaly_common_cause (an OWN alarm shared by another battery of the same campaign within +-5
   cycles = COMMON event of the test, else INDIVIDUAL) and anomaly_events (dated plain-language events: FLEET alarm /
   cleared, OWN individual / common / cleared, OWN indicator active, capacity regeneration after a rest = normal).
   Validation: R1, R3, R4 sharing a test event -> common; knees R2, C2, H3 -> individual; 11/11 correct.
3. Live twin: "Live anomaly detection" toggle (on by default): both indicators run with the replay cursor, status of
   each indicator, event log of everything that happened up to the cursor, and the indicator chart up to the cursor.
4. Operations & control: OWN alarm type (individual / common with ...) per battery and "What happened" log for the
   replayed battery.
5. Page order: Home, Diagnostics, Models & forecasting, Model updating (new page: the update studies, moved from the
   Live twin), Live twin, Operations & control, Study results. Pages referenced by name (V_*), not by position.
6. 38 tests pass; AppTest of the Live twin panel, Model updating page, Operations; all 7 pages.

## Mission 3 · anomaly detection with two indicators, robust to regeneration (v5.25.0)

1. NASA scan with v5.24: alarms on most long-life batteries (NASA's rest periods: capacity jumps then relapses);
   B0041 / B0045 (abnormal from the start) not seen. But kappa was highest for exactly the batteries the hybrid could
   not explain (B0041 3.00, B0045 2.65, B0006 2.61, B0018 1.75, B0005 1.46; others 0.30-1.22).
2. Engine section 52: FLEET indicator (kappa refitted every cycle vs the fleet physics fitted on the other batteries;
   alarm when kappa > 1.8 for 5 cycles, after 8 cycles of history) and OWN indicator (Theil-Sen slope of the last 30
   cycles vs the 30 before, z-test with the Theil-Sen intervals; alarm when z > 3 for 3 cycles).
3. Validation cohort with NASA-like regeneration (+1.5 % every 18-25 cycles, decaying over ~4): 11/11 correct -
   knees (R2, C2, H3) by OWN after 27-35 cycles, batteries 2.5x faster from the start (R3, C3) by FLEET at cycle 12,
   6 healthy quiet on both (kappa 0.44-1.30). The v5.24 detector raised false alarms on all 8 non-knee batteries of
   this cohort.
4. App, Operations & control: anomaly section with both indicators (thresholds, check, NASA scan, three-panel replay:
   SOH, kappa, slope z), CSV. 36 tests pass; AppTest; all views.

## Mission 3 · anomaly detection (v5.24.0)

1. Engine section 51: causal replay of every battery; expectation for the next 10 cycles = continuation of the
   battery's own last 15 cycles (in throughput); shortfall divided by the regression's prediction error
   (sigma * sqrt(1 + 1/m + (a - mean)^2 / Saa)); one-sided CUSUM (k = 0.5), alarm above h = 8. The shared physics
   (fitted leave-one-out) gives the ageing-speed factor kappa as context.
2. Validation on the knee cohort (knees on R2, C2, H3 from cycle 53; 8 healthy): fleet-physics expectation 5 false
   alarms; own trend / cycle noise 2; own trend / prediction error, h = 8: 3/3 knees, 0/8 false alarms, alarm when the
   knee adds 0.9-1.5 SOH points (noise ~0.3). h = 5: a few cycles earlier, 1 false alarm.
3. App, Operations & control (top): "Mission 3 · Anomaly detection": threshold, built-in check, NASA scan (alarms per
   battery with kappa), replay chart (measured vs expected, CUSUM, alarm), CSV.
4. tests/test_anomaly.py (validation, causality); 34 tests pass; AppTest; all views.

## Ageing-speed factor for the physics forecasts (v5.23.0)

1. Physics forecasts stayed too optimistic with v3; the shared physics has no battery-specific information.
   physics_forecast kinds phys2k, hyb2k, phys3k, hyb3k: after the shared fit (unchanged), one factor kappa in
   [0.3, 3] is fitted on the forecast battery's own history only; its mechanisms run kappa times faster per Ah (the
   same for all three: the split is kept). Assessment level "Level 4 · Physics + ageing-speed factor".
2. Demo (Reference, 30 %): worse than without (v2 5.73 -> 7.53, v3 4.87 -> 5.53): kappa changes the speed, not the
   shape of the fade; when the physics' shape differs, fitting the early history extrapolates worse. NASA decides.
3. Tests: causal; a battery with faster history gets a larger kappa. 32 tests pass; AppTest; all views.

## Physics version 3 (v5.22.0)

1. Physics-based forecasts (v5.21) were too optimistic on almost every NASA battery: SEI decelerates, nothing
   accelerates. nasa_mechanisms_v3: SEI = f_sei * a^z with the exponent z learned in [0.3, 1.5] (steady growth on
   cracking particles allowed); plating triggered by lost graphite at any temperature (dp/da = k_tr (LAM - m_c)+) on
   top of the cold-driven plating; LAM linear (v2's unused knee term removed). 12 shared parameters.
2. Identifiability: on batteries generated with the v3 laws, v3 recovers the split (0.10 share points), z 0.86
   (truth 0.9), m_c 0.030 (0.03), Ea 50.2 (50); all starts agree. Caveat: on batteries whose knee has no mechanism,
   v3 reads the knee as triggered plating (7.5 vs 3.4 share points for v2): the most common cause, not the only one.
3. Hybrid and forecasters are version-aware (hybrid_fit_cds / nasa_hybrid_fit version=, physics_forecast keys
   "phys3:Shared physics v3", "hyb3:Hybrid v3").
4. App: Version 3 is the default in "Shared physics on the NASA batteries"; the hybrid has a physics-version choice;
   the assessment has "Level 4 · Shared physics v3 + hybrid v3" next to the v2 level.
5. Tests: v3 recovers its own split and exponent; v3 forecasters causal. 29 tests pass; AppTest of the three
   sections with v3; all views.

## Physics-based forecasters in the accuracy assessment; joint-training test (v5.21.0)

1. Engine section 50, physics_forecast: shared physics v2 ("phys:Shared physics v2"), frozen hybrid ("hyb:Hybrid"),
   joint hybrid with guardrails ("hybj:Hybrid") as forecasters. Trained on every other battery of the constant-
   condition groups plus the target's history up to n0 only; future extrapolated cycle by cycle (throughput from the
   predicted capacity); starts from SOH now (median of the last 3 clean cycles); band from the prefix fit error.
   Test: corrupting the target's data after n0 changes nothing.
2. Hybrid joint training with guardrails (hybrid_fit_cds: rounds, battery_code, ortho = penalty on the correction's
   projection onto the SEI / plating / LAM curves). Validation on the known split: frozen 3.43 share points; joint
   without guardrails 5.08; no battery code 5.09; + orthogonality 10: 4.96; 100: 4.25; code + 100: 4.17 (fit worse
   as guardrails tighten). Frozen stays the explanation model; the joint variant is kept only to compare forecasts.
3. App: accuracy assessment level "Level 4 · Shared physics + hybrids".
4. tests/test_physics_forecast.py (3); 26 tests pass; AppTest of the assessment with the new level; all views.

## Hybrid fixes (v5.20.1)

1. The network correction is now zero at the start of life (multiplied by a / (a + 0.05)): on B0045 it started at
   ~8 % at cycle 2, when the battery had lost almost nothing.
2. Bound raised from 8 to 25 SOH points: on B0006 the physics alone misses ~20 points at end of life, and the 8-point
   cap hid most of that unexplained part. The physics is frozen, so the bound cannot change the split.
3. Validation unchanged: split 3.43 (identical), knee-battery fit 0.73 -> 0.33. Tests, AppTest and all views pass.

## Hybrid model; PINN v2 and recovery-test sections removed from the app (v5.20.0)

1. Engine section 49, hybrid: shared physics v2 fitted first and FROZEN (the mechanism split), then a small tanh
   network (bounded to +-8 SOH points, penalised) fitted to what the physics leaves; its correction is reported as
   "Not explained by the physics". _shared_physics_core factored out of nasa_physics_fit (results identical).
2. Validation (hybrid_validation): batteries following physics v2 plus an unmodelled knee on R2, C2, H3. Every
   variant in which the network also shaped the physics worsened the split (3.9-6.8 share points vs 3.4 for the
   physics alone: alternating refits, robust loss, knee-free physics, refit without flagged batteries). Frozen
   physics: split identical (3.43), fit error on knee batteries 0.73 -> 0.33, others 0.45 -> 0.31.
3. App, Models & forecasting, Level 4: "Shared physics on the NASA batteries (no neural network)" kept and labelled;
   new "Hybrid" section (fit, built-in check on simulated batteries, split with the unexplained part, fit
   improvement, batteries the physics does not describe, trajectory, CSV). The "PINN v2 (experimental)" and
   "Recovery test" sections are removed from the app; their engine code, tests and scripts remain.
4. 23 tests pass; AppTest: hybrid section runs (check and NASA fit); all views.

## Shared physics on NASA, version 2 (v5.19.0)

1. First NASA run (v5.18.0) was unique but not credible: plating dominated at 30-40 C (a catch-all for steady
   fade), the charge-current sensitivity cannot be learned (every NASA battery charged at 1.5 A), several
   parameters at their limits, fit errors 9.5-13 SOH points on knee batteries.
2. Version 2 (nasa_mechanisms_v2): plating only below T_COLD_C = 15 C cell temperature; no charge-current term; a
   shared knee term (LAM accelerating with throughput, faster with deeper discharge); fit the SEI to Re or to the
   load-step resistance, the other one held out as a check. Version 1 kept, unchanged.
3. App: choice of model version and of the evidence; the card lists every parameter that ends at a limit
   (= not determined by the data) and both held-out checks; CSV records version, evidence and parameters at a limit.
4. Tests: plating is exactly zero on warm batteries with version 2; 22 tests pass; AppTest; all views.

## Shared physics on the real NASA batteries (v5.18.0)

1. Engine section 48: nasa_physics_fit, nasa_physics_table, nasa_physics_trajectory. The model that passed the
   recovery test (one shared parameter set, SEI / plating / LAM from exact rate-law solutions) fitted to every
   battery of Reference, High current, Hot, Cold and Pulsed load at once, from many starts. Real data: cleaned
   capacity, relative load-step resistance, missing values skipped; Re optionally held OUT and used as a check
   (rank correlation of the SEI state with the measured Re).
2. App, Models & forecasting, after the recovery test: "Level 4 · Shared physics on the NASA batteries": checks first
   (agreement between starts, fit error per battery, learned physics and bounds, held-out Re check), split of every
   battery by group, group means against electrochemical expectations, per-battery table, trajectory of a chosen
   battery, CSV.
3. 21 tests pass; AppTest: section renders and runs; all views.

## Recovery test in the app: can a model find the TRUE mechanisms? (v5.17.0)

1. Engine section 47: 11 simulated batteries with a KNOWN split of SEI, plating and LAM (4/24/43 C, 1-4 A,
   charge 0.75-1.5 A, cut-off 2.0-2.7 V; rate laws of PINN v2's forms), recovery_cohort, recovery_true_shares,
   recovery_physics_fit (one shared parameter set, exact solutions, many starts), recovery_physics_table,
   recovery_pinn_table. scripts/pinn_recovery.py now uses these.
2. App, Models & forecasting, after PINN v2: "Level 4 · Recovery test · Can a model find the TRUE degradation
   mechanisms?": the batteries and their true split, run pure physics / PINN v2 / PINN v1 (targets, seeds), summary
   table (error in share points, dominant correct), truth-vs-model stacked bars per battery, conclusion card, CSV.
3. Result: pure physics 1.1 share points, 11/11, 30/30 starts agree, Ea 47.9 kJ/mol (truth 50); PINN v2 44 points.
4. tests/test_recovery.py; 20 tests pass; AppTest: section renders and runs; all views.

## PINN v2: faulty dQ/dV rule switched off, attribution labelled not validated (v5.16.1, fix)

1. On five NASA batteries v2 attributed 98-100 % of the loss to LAM: the global dQ/dV scale follows the capacity
   whatever the mechanism, so "scale = 1 - LAM" forced the result. Both dQ/dV rules (scale, shift) are now OFF in the
   training loss by default (dq_weight = 0); v2 keeps capacity, resistance and Re, the corrected physics, the bounds
   and the cross-group training.
2. App: section marked "experimental" with a "Not yet validated" notice; the evidence chart shows the resistance
   evidence (Re and load-step resistance) instead of dQ/dV curves that only echoed the rule.
3. Test: absurd dQ/dV values leave the v2 attribution unchanged. PINN v1 unchanged (fingerprint). 19 tests pass.

## PINN v2: degradation mechanisms explained with voltage-curve and impedance evidence (v5.16.0)

1. Engine section 45, dQ/dV degradation-mode indicators for every clean cycle: Q(V) from the current, IR-corrected,
   smoothed, dQ/dV on a 5 mV grid, main peaks, fit Q_n(V) = a * Q_ref(V - delta). Per cycle: scale a (LAM-like),
   LLI-like shift = (charge + discharge shift) / 2, polarisation = (charge - discharge shift) / 2.
   dqdv_cell, dqdv_modes, dqdv_indicators, attach_dqdv. scripts/dqdv_features.py now uses these.
2. Engine section 46, PINN v2 (_pinn2_fit, pinn_explain): inputs add charge current and depth of discharge; plating
   driven by cold and charge current, LAM by discharge current and depth of discharge, SEI Arrhenius with Ea bounded
   to 20-80 kJ/mol; sensitivities non-negative; must reproduce SOH, R_dc, EIS Re (SEI film), dQ/dV scale (= 1 - LAM)
   and dQ/dV shift (= kappa * LLI). Trained on several groups; attribution with its spread over seeds.
   PINN v1 is untouched (output identical before and after: checked with a fingerprint).
3. App, Models & forecasting: new section "Level 4 · PINN v2 · Which degradation mechanisms explain this battery's
   capacity loss?": dQ/dV curves and indicators of the selected battery; explanation run (train on all constant-
   condition groups or this group; 1/3/5 seeds; optional PINN v1 comparison): mechanisms stacked over life against
   the measured loss, v2 vs v1 attribution table (mean +- SD), evidence check (dQ/dV scale, shift, Re: observed vs
   model), learned physics, electrochemical expectation of the group, stamped CSV. Mixed and corrupted batteries get
   the dQ/dV analysis only.
4. Tests: tests/test_dqdv.py (3): pure-scaling synthetic recovered; dQ/dV columns attached; PINN v2 reproduces the
   evidence and keeps its bounds. 18 tests pass; AppTest: new section renders and runs; all views.

## Update study uses the assessment's data rule (v5.15.2, bug fix)

1. The frozen-model update study now uses the same data as the accuracy assessment, whatever the control-bar
   toggle says: Mixed conditions at each battery's reference condition, every other group as recorded. Before, it
   followed the Reference-condition toggle, and two Mixed runs with the toggle off scored capacity jumps at load and
   temperature changes instead of ageing (never/no-anchor 8.6-23.0 vs ~0.7 in the assessment).
2. A caption says so for Mixed batteries; the study CSV records the data mode ("data" column).
3. Checked: AppTest with demo batteries relabelled as Mixed and the toggle off (reference data used, no errors);
   15 tests; all views.

## Frozen-model panel in the Live twin (v5.15.1)

1. A framed panel "Frozen assessment model · battery · group" sits above the Live models list. It shows the toggle,
   retraining interval and anchor when the frozen model can run, and otherwise the exact reason: no group, a group
   without a frozen model (e.g. High current), or fewer than two other usable batteries in the group.
2. Fixed: a group with a frozen model but too few batteries was reported as having no frozen model.
3. In the Live models list the frozen entry reads "Frozen <group> model · <model>" when running, and "not running for
   this battery" otherwise (it used to look available even when it was not).

## Frozen models in the self-updating twin (v5.15.0)

1. `FROZEN_MODELS`: the assessment winner per group (forecast, SOH-now), default settings (STATUS 5h).
2. `frozen_stream`: from 30 % of life the frozen forecaster runs live on one battery: retrained every N cycles
   (0 = never) on the group's other batteries plus this battery's history up to now, and re-anchored every cycle to
   the current SOH: measured capacity (lab), the frozen SOH-now model's estimate (field: no capacity after the start,
   retraining also sees only estimates) or none. "Never, none" reproduces the assessment forecast exactly.
3. Live twin: toggle "Frozen <group> model" (retraining interval, anchor). It joins the live chart, the causal
   5-step-ahead scoreboard and the ensemble weights like the filters. Forecast grids end at the last recorded cycle;
   beyond it the forecast is extended with its final slope (never rising).
4. New section "Mission 2 · How often should the frozen model update?": retraining 5/10/20/50/never x anchor;
   forecast RMSE over the next 10/20/40 measured cycles, SOH-now RMSE (field), EOL error, retrainings per 100
   cycles; this battery or the whole group (cached per battery, resumes); chart error vs cost; stamped CSV.
5. Tests (tests/test_frozen_live.py, 10): forecasts at cycle m unchanged when the battery's data after m are
   corrupted (ML, DL, PINN); SOH-now estimates causal (ML, DL); field mode unchanged when every capacity value
   after the start is corrupted; never/none = assessment; anchored curve passes through the measured SOH; the live
   twin scores and weights the frozen model; study table. AppTest: lab and field replay, study, all views.

## Self-describing assessment CSVs (v5.14.1)

1. Both assessment downloads now carry, on every row: app version, group, EOL setting ("1.40 Ah" or "80 % of
   initial capacity"), data mode ("as recorded" / "reference condition") and the group's batteries. File names:
   `assessment_<group>__v<version>.csv` and `..._cv_picks.csv`. Reason: a Cold rerun came back identical to the
   previous run and the CSV could not show whether the new version or the new EOL setting had been used.

## Every usable group in the accuracy assessment (v5.14.0)

1. Pulsed load and Mixed conditions are now offered as "(exploratory)" groups with a warning that explains the
   caveat; only Corrupted logging (B0049-B0056) stays excluded. The Models-view training pools are unchanged.
2. Mixed conditions are always scored at each battery's reference condition (Reference-condition SOH applied
   automatically, own cache key), so SOH measures ageing, not load or temperature changes.
3. Group membership in the assessment always follows the recorded test conditions. Before, switching on the
   Reference-condition toggle could move mixed-load cells into other groups (B0041 appeared in Cold). Every other
   group is assessed on the data as recorded, independent of that toggle.
4. AppTest: Mixed and Pulsed offered, warned, run and rendered; Corrupted never offered; group list identical with
   the toggle on and off; all views render twice.

## Level 2 CV no longer sees the test cycles (v5.13.0)

1. Bug: in `train_trajectory_model` the predicted battery's own past rows were in the CV pool, so a fold could hold
   out the predicted battery and score it on its cycles after the origin, i.e. the test cycles. The test RMSE was
   not affected (the final model is fitted separately), but the CV error, the CV picks / star, self-optimisation and
   the Level 2 forecast bands (coverage) were. Found while checking a real Reference run. DL and PINN CV were clean.
2. Fix: CV folds are built from the training batteries only; the predicted battery's rows always stay in the fit.
3. Measured mode: a held-out battery now keeps its own cycles up to the same share of its life in the fold's fit,
   exactly like the predicted battery in the test (before, CV was a harder task: errors 12-22 vs ~3 on test).
4. Test `test_cv_never_sees_the_target_future`: halving the target's SOH after the origin leaves the CV error
   unchanged (forecast and measured, two models); it fails on v5.12.2.

## Interrupted runs no longer produce a CV pick (v5.12.2)

1. `select_by_cv(res, k, keys)`: when not every model of a level has run on a (battery, origin), that level gets no
   pick there (status "incomplete: n of N models of this level ran"); choosing among the models that happened to
   finish is not the method being assessed. Found on a real Reference run interrupted after 3 of 7 ML models.
2. The app warns about such incomplete levels and tells how to resume. Test added (4 tests in test_cv_selection.py).

## Charts saved as named images (v5.12.1)

1. The Plotly toolbar is always visible on every chart; its camera icon saves a PNG (2x resolution, in the browser, no
   extra package needed) named `chart-id__battery-or-group__chart-title__date__version.png`, e.g.
   `acc-board__Reference__prediction-accuracy-assessment-same-batteries-same-protocol__2026-10-14__v5.12.1.png`.
   Battery for single-battery charts, condition group for the accuracy-assessment charts.
2. The export menu (sidebar toggle "Export menus on charts") uses the same name for HTML, CSV, and for PNG / SVG / PDF
   when `kaleido` works.
3. Checked with AppTest: every chart in every view on the demo cohort carries a PNG camera with a safe file name.

## Accuracy assessment: best 2 models per level, chosen by CV in each fold (v5.12.0)

1. New default in "What to score": *Best 2 per level (chosen by CV in each fold)*. For every held-out battery and
   origin, all models of a level are fitted; the 2 with the lowest CV error on the OTHER (training) batteries become
   "best by CV" and "2nd by CV" and are scored on the held-out battery. Levels: Level 2 ML forecast / measured,
   Level 3 DL forecast / measured; Level 4 (PINN + no-physics ablation) and the 3 baselines are always shown as they
   are. Picking the best 2 from the full scoreboard afterwards would be optimistic (same batteries choose and confirm).
2. Every assessment row now records the forecast's own CV error (`CV RMSE (SOH pts)`; NaN for baselines).
3. `select_by_cv(res, k)` and `cv_pick_agreement(picks)` in the engine; in the app: a table "does the star agree with
   the test?" (share of folds where the CV pick was also best on test, mean test rank), the model picked on each
   battery with its test rank, every model's score in an expander (reference only), CSV of the picks.
4. Run time is unchanged versus running all models (every model must be fitted to know its CV error).
5. Tests: tests/test_cv_selection.py (choice follows CV, never the test error; failure rows when too few models have a
   CV error; real protocol records CV). AppTest: both modes, two renders, all views on the demo cohort.

## Fixes from the first real results across condition groups (v5.11.0)

1. SOH at the origin = median of the last 3 valid, transient-free cycles (a running minimum stayed pinned by a low
   first cycle, e.g. B0029 at 93 %, and pulled every baseline and Level-2 forecast ~4 points down; synthetic check:
   forecast error 0.043 -> 0.008).
2. Held-out (CV) error on the same task as the test: each held-out training battery is cut at the same share of life
   (3 cuts around it) and forecast with the same anchoring; drives the recommended model, self-optimisation and
   ensemble weights. On synthetic data agreement with the test error is about the same as before (no gain shown
   there); the mismatch was large on real data.
3. Inputs limited to the training range widened by 50 % each side, and forecasts at most 2 SOH points above today's
   value (Bayesian ridge reached 105 % and CV errors of 10^9 on real data; hard clipping froze legitimate
   extrapolation, so the range is widened).
4. `safe_origin`: at least 10 valid cycles before every origin (moved forward and stated otherwise), in the app and in
   the assessment.
5. End of life as a fixed capacity (Ah) or as % of each battery's initial capacity (`soh_eol_for`: values <= 1 are
   fractions).
6. No 95 % interval with fewer than 3 batteries; error bars stop at zero.
7. `reference_condition_ct` + "Reference-condition SOH" switch: keep only cycles at each battery's initial current
   and ambient, so SOH measures ageing only in the mixed-conditions group.

## Indicators for every model (v5.10.2)

A model trained without permutation importance returned a zero-filled table, so the indicator chart showed all zeros.
It now returns no table; under each level's results a selector chooses the model whose indicators to show (Level 2
models and the MLP), computed on demand with that run's training batteries and settings, then cached. The accuracy
assessment now says when results cover only part of the group (interrupted run) and when no baseline was selected.

## Faster training with many batteries (v5.10.1)

Training on all ~33 batteries built ~14,000 forecast-mode rows, fitted 4 times per model (SVM 60 s, Random Forest 33 s,
PINN 62 s per battery). Training rows are now sampled evenly across training batteries (max 6,000; SVM 4,000; the
predicted battery keeps all its rows): SVM 63 s -> ~10 s (RMSE 0.0040 -> 0.0048), Random Forest 36 s -> 17 s (same or
better RMSE). PINN: fewer cycles and collocation points per battery when there are many batteries, and the two
held-out fits that set the band use 60 % of the iterations: 62 s -> 35 s, same forecast error.

## Speed and self-optimisation (v5.10.0)

Profiling a Level-2 run showed most time in work repeated for every model: the training table (39 %) and the
condition-group classification (26 %, recomputed 280 times). Now cached on a data fingerprint (identity, size, invalid
cycles and sums of the main measured columns), with identical results: 33.4 s -> 25.4 s for five models; permutation
importance only for the recommended model; trained results cached by Streamlit, so the same run repeats in ~1.4 s
instead of ~29 s. PINN early stop (< 0.2 % loss improvement over 300 iterations): small gain, identical results.
Self-optimisation: 'Self-optimise' per level (each model tunes itself with Bayesian search on held-out training
batteries and the best settings are remembered per battery set: first run ~1-3 min per model, then instant), a
self-weighted ensemble per level (`ensemble_trajectories`, weights 1 / CV error^2), and a ★ recommended model (lowest CV
error, chosen without the test battery's future).

## Training-data diagnosis, equations, banner restored (v5.9.0)

- `training_influence`: retrains a Level-2/3/4 model once without each training battery and reports how the predicted
  battery's error changes (negative = the battery perturbs the model), with a comment naming the likely cause for groups
  with known data issues (`GROUP_ISSUES`). The data are kept as they are; this shows which batteries actually hurt.
  Diagnostic only (it compares with the battery's actual later cycles). App: "Which training batteries perturb the
  models?" under each level.
- Equations: a ∑ button next to every model (Levels 1-4) shows its defining equations (`MODEL_EQUATIONS`), plus how
  Levels 2-3 turn a model into a forecast.
- The animated cell <-> twin banner is back at the top (the compact title bar is removed); status strip below it.

## Level 4 · Physics-informed neural network (v5.8.0)

`train_pinn_model` / `train_pinn_level`: three latent mechanism states (LLI by SEI growth, LLI by lithium plating, LAM)
learned from all training batteries and the predicted battery's cycles up to the origin. Electrochemistry links them to
the measurements (SOH = (1 - LLI_sei - LLI_pl)(1 - LAM); R/R0 grows with the SEI film). Degradation equations enforced as
residuals at collocation points, including the predicted battery's future: diffusion-limited SEI growth with Arrhenius
temperature dependence; plating promoted by cold, high current and LAM (knees); LAM with cycling. Per-battery
multipliers through a learned battery code. Band and CV error from held-out forecasts (training batteries cut at the
same share of life). "Same network without physics" included as an ablation; the physics weight is a tunable parameter.
Mechanism decomposition chart and learned-physics table. Forecast mode only.
Synthetic check (3 batteries, origin 30 %): CV error of held-out forecasts 0.033 with physics vs 0.062 without; on the
test batteries themselves mixed (physics better on 1 of 3: 0.035 vs 0.069; worse on 2: 0.048 vs 0.011, 0.048 vs 0.019).
The synthetic generator creates knees by a rate change at a fixed cycle, not by LAM-driven plating, so its physics is
partly mis-specified for the PINN: judge on real data with the ablation and the assessment.

## Fix: training with all batteries (v5.7.1)

Training Level 2 / Level 3 on all batteries failed ("Found array with 0 samples") when one training battery had no
valid cycle before an origin (e.g. corrupted-logging cells): the cleaning step received an empty table, and
`condition_groups` failed for a battery without any valid cycle. Cleaning is now empty-safe, such batteries get no
group, unusable training batteries are skipped individually and listed under the chart, and the app warns when the
training set contains batteries from the usually excluded groups (mixed loads, corrupted logging, pulsed load).

## Parameters, automatic selection, reliable assessment, new design (v5.7.0)

- Every Level-2 and Level-3 model has a 'Model parameters' panel (its own settings, from te.MODEL_SPECS; specs added
  for MLP, GRU, Transformer) and an 'Automatic parameter selection' (`tune_level_model`: Bayesian optimisation or
  random search, defaults always tried first, scored by the CV error on training batteries held out during
  cross-validation, never on the test battery's future). Tuned values are written into the panel and used by the next
  Train. Synthetic check: MLP measured CV 0.056 -> 0.021; GRU forecast CV 0.028 -> 0.024 and test RMSE 0.033 -> 0.015;
  Random Forest -3 %.
- Accuracy assessment: each forecast is a cached task (`run_protocol_task`); the scoreboard fills in live, partial
  results survive an interruption and repeated runs reuse finished forecasts (synthetic: 31 s -> 1.4 s). New headline
  metrics and a per-battery table (best model per battery, battery wins). Verdict bug fixed (it ignored the
  'better on all N batteries' wording). Defaults: origin 30 %, fewer models; time estimate before running.
- Design: instrument-panel look (IBM Plex Sans with tabular figures, one teal accent, calm hero without the feature
  chips, sentence-case labels, status bar without monospace capitals, reduced-motion respected).

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
