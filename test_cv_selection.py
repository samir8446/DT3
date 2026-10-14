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
