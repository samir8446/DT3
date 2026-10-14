"""v5.12: best models per level chosen by CV inside each fold."""
import numpy as np
import pandas as pd
import twin_engine as te


def _fake_res():
    rows = []
    for c in ("A", "B", "C"):
        for m, cv, test in (("ML · RF", 1.0, 5.0), ("ML · GP", 2.0, 1.0), ("ML · SVM", 3.0, 0.5), ("ML · DT", np.nan, 0.1)):
            rows.append({"Cell_ID": c, "origin": 0.3, "n0": 30, "model": m, "key": m, "family": "Machine learning (forecast)",
                         "test cycles": 50, "status": "ok", "RMSE (SOH pts)": test, "CV RMSE (SOH pts)": cv,
                         "EOL error (cycles)": np.nan, "coverage": np.nan})
        rows.append({"Cell_ID": c, "origin": 0.3, "n0": 30, "model": "Fleet shape", "key": "shape", "family": "Baselines",
                     "test cycles": 50, "status": "ok", "RMSE (SOH pts)": 3.0, "CV RMSE (SOH pts)": np.nan,
                     "EOL error (cycles)": np.nan, "coverage": np.nan})
    return pd.DataFrame(rows)


def test_choice_uses_cv_not_test():
    out, picks = te.select_by_cv(_fake_res(), 2)
    best = picks[picks["Choice"] == "best by CV"]
    assert set(best["Picked model"]) == {"ML · RF"}          # lowest CV, although worst on test
    assert set(picks[picks["Choice"] == "2nd by CV"]["Picked model"]) == {"ML · GP"}
    assert "ML · DT" not in set(picks["Picked model"])        # no CV error -> never chosen (best on test!)
    assert (best["Test rank"] == 4).all()                     # diagnostic reports the bad pick honestly
    sb = te.protocol_scoreboard(out)
    assert abs(sb.loc["Level 2 · ML forecast · best by CV", "RMSE (SOH pts)"] - 5.0) < 1e-9
    assert "Fleet shape" in sb.index and "ML · SVM" not in sb.index
    ag = te.cv_pick_agreement(picks)
    assert ag.loc["Level 2 · ML forecast", "Star also best on test"] == 0.0


def test_too_few_candidates_is_a_failure_row():
    r = _fake_res()
    r.loc[r["model"].isin(["ML · GP", "ML · SVM"]), "CV RMSE (SOH pts)"] = np.nan
    out, picks = te.select_by_cv(r, 2)
    second = out[out["model"] == "Level 2 · ML forecast · 2nd by CV"]
    assert len(second) == 3 and second["status"].str.startswith("failed").all()


def test_real_protocol_records_cv_and_selects():
    master, imp, _ = te.make_synthetic_master(n_cells=4, n_cycles=80, seed=3)
    store = te.ParquetStore.from_dataframe(master)
    ct = te.build_cycle_table_with_report(store)[0]
    cells = sorted(ct["Cell_ID"].unique())[:4]
    keys = ["shape", "traj:Bayesian Ridge", "traj:Random Forest", "traj:Decision Tree"]
    res = te.run_protocol(store, ct, imp, cells, keys, (0.3,), only_cells=[cells[0]])
    ok = res[res["status"] == "ok"]
    assert ok.loc[ok["key"].str.startswith("traj:"), "CV RMSE (SOH pts)"].notna().all()
    assert ok.loc[ok["key"] == "shape", "CV RMSE (SOH pts)"].isna().all()
    out, picks = te.select_by_cv(res, 2)
    assert len(picks) == 2 and picks["CV RMSE (SOH pts)"].is_monotonic_increasing
    assert set(out["model"]) >= {"Fleet shape", "Level 2 · ML forecast · best by CV", "Level 2 · ML forecast · 2nd by CV"}


def test_interrupted_level_gets_no_pick():
    r = _fake_res().rename(columns={})
    r["key"] = r["model"].map({"ML · RF": "traj:Random Forest", "ML · GP": "traj:Gaussian Process",
                               "ML · SVM": "traj:SVM", "ML · DT": "traj:Decision Tree"}).fillna(r["key"])
    keys = ["shape", "traj:Random Forest", "traj:Gaussian Process", "traj:SVM", "traj:Decision Tree", "traj:Bayesian Ridge"]
    out, picks = te.select_by_cv(r, 2, keys)                  # Bayesian Ridge never ran -> level incomplete everywhere
    assert picks.empty
    sel = out[out["model"].str.contains("by CV")]
    assert len(sel) == 6 and sel["status"].str.startswith("incomplete: 4 of 5").all()
    out2, picks2 = te.select_by_cv(r, 2, keys[:-1])           # complete -> picks as before
    assert len(picks2) == 6


def _leak_setup():
    master, imp, _ = te.make_synthetic_master(n_cells=5, n_cycles=90, seed=11)
    store = te.ParquetStore.from_dataframe(master)
    ct = te.build_cycle_table_with_report(store)[0]
    cells = sorted(ct["Cell_ID"].unique())
    return ct, imp, cells[0], cells[1:]


def test_cv_never_sees_the_target_future():
    """Changing the predicted battery's cycles after the origin must not change its CV error (forecast and measured)."""
    ct, imp, tgt, train = _leak_setup()
    n0 = 30
    bad = ct.copy()
    fut = (bad["Cell_ID"] == tgt) & (bad["n"] > n0)
    for col in ("SOH", "Q_dis_Ah"):
        if col in bad:
            bad.loc[fut, col] = bad.loc[fut, col] * 0.5          # absurd future: would dominate a leaking CV
    for mode in ("forecast", "measured"):
        for model in ("Bayesian Ridge", "Random Forest"):
            a = te.train_trajectory_model(ct, imp, tgt, n0, model, mode, train)
            b = te.train_trajectory_model(bad, imp, tgt, n0, model, mode, train)
            assert np.isclose(a.cv_rmse, b.cv_rmse, rtol=0, atol=1e-12), (mode, model, a.cv_rmse, b.cv_rmse)
