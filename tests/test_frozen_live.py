"""v5.15: frozen assessment winners in the self-updating twin."""
import numpy as np
import pandas as pd
import pytest
import twin_engine as te


@pytest.fixture(scope="module")
def data():
    master, imp, _ = te.make_synthetic_master(n_cells=4, n_cycles=80, seed=5)
    ct = te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(master))[0]
    cells = sorted(ct["Cell_ID"].unique())
    return ct, imp, cells[0], cells[1:]


def _corrupt_after(ct, cell, m, cols=("SOH", "Q_dis_Ah", "T_mean_C", "R_dc_ohm", "V_mean_V", "t_cc_s", "t_cv_s")):
    bad = ct.copy()
    sel = (bad["Cell_ID"] == cell) & (bad["n"] > m)
    for c in cols:
        if c in bad:
            bad.loc[sel, c] = bad.loc[sel, c] * 0.6
    return bad


@pytest.mark.parametrize("key", ["traj:Bayesian Ridge", "dl:MLP", "pinn:PINN"])
def test_forecast_at_m_ignores_the_future(data, key):
    ct, imp, tgt, train = data
    m = 30
    a = te._protocol_forecast(key, None, ct, imp, tgt, m, train, te.DEFAULT_EOL_AH)
    b = te._protocol_forecast(key, None, _corrupt_after(ct, tgt, m), imp, tgt, m, train, te.DEFAULT_EOL_AH)
    assert np.allclose(a.soh, b.soh, atol=1e-12)


@pytest.mark.parametrize("key", ["trajm:Bayesian Ridge", "dlm:GRU"])
def test_soh_now_estimate_is_causal(data, key):
    """The estimate for cycle n uses cycles <= n only: corrupting cycles after k leaves estimates up to k unchanged."""
    ct, imp, tgt, train = data
    m, k = 20, 45
    a = te._protocol_forecast(key, None, ct, imp, tgt, m, train, te.DEFAULT_EOL_AH)
    b = te._protocol_forecast(key, None, _corrupt_after(ct, tgt, k), imp, tgt, m, train, te.DEFAULT_EOL_AH)
    sel = (np.asarray(a.n_grid) > m) & (np.asarray(a.n_grid) <= k)
    assert sel.sum() > 5 and np.allclose(np.asarray(a.soh)[sel], np.asarray(b.soh)[sel], atol=1e-12)


def test_never_retrain_no_anchor_reproduces_the_assessment(data):
    ct, imp, tgt, train = data
    fr, tr = te.frozen_stream(ct, imp, tgt, train, "traj:Bayesian Ridge", None, 0.8, refit_every=0, anchor="none")
    n0 = tr.attrs["n_start"]
    f = te._protocol_forecast("traj:Bayesian Ridge", None, ct, imp, tgt, n0, list(train), te.DEFAULT_EOL_AH)
    fut, med, _, _ = fr[n0]
    inside = fut <= np.asarray(f.n_grid)[-1]
    assert np.allclose(med[inside], np.interp(fut[inside], f.n_grid, f.soh), atol=1e-12)
    assert tr.attrs["refits"] == 1


def test_capacity_anchor_passes_through_the_measurement(data):
    ct, imp, tgt, train = data
    fr, tr = te.frozen_stream(ct, imp, tgt, train, "traj:Bayesian Ridge", None, 0.8, refit_every=20, anchor="capacity")
    ok = tr[tr["status"] == "ok"]
    assert tr["refit"].sum() == tr.attrs["refits"] >= 2
    for _, row in ok.iloc[[3, 7, 15]].iterrows():
        n, m = int(row["n"]), int(row["fitted_at"])
        f = te._protocol_forecast("traj:Bayesian Ridge", None, ct, imp, tgt, m, list(train), te.DEFAULT_EOL_AH)
        assert abs(np.interp(n, f.n_grid, f.soh) + row["shift"] - row["soh"]) < 1e-12   # curve at n = measured SOH
        fut, med, _, _ = fr[n]
        assert np.allclose(med[:3], np.interp(fut[:3], f.n_grid, f.soh) + row["shift"])


def test_field_mode_never_sees_capacity_after_the_start(data):
    ct, imp, tgt, train = data
    kw = dict(refit_every=20, anchor="estimate")
    fr_a, tr_a = te.frozen_stream(ct, imp, tgt, train, "traj:Bayesian Ridge", "trajm:Bayesian Ridge", 0.8, **kw)
    n0 = tr_a.attrs["n_start"]
    bad = _corrupt_after(ct, tgt, n0, cols=("SOH", "Q_dis_Ah"))   # capacity only; operando signals intact
    fr_b, _ = te.frozen_stream(bad, imp, tgt, train, "traj:Bayesian Ridge", "trajm:Bayesian Ridge", 0.8, **kw)
    assert fr_a.keys() == fr_b.keys()
    for n in fr_a:
        assert np.allclose(fr_a[n][1], fr_b[n][1], atol=1e-12), n


def test_live_twin_scores_and_weights_the_frozen_model(data):
    ct, imp, tgt, train = data
    fr, _ = te.frozen_stream(ct, imp, tgt, train, "traj:Bayesian Ridge", None, 0.8, refit_every=10, anchor="capacity")
    frames, track = te.live_multi_model(ct, tgt, None, 0.8, models=("trend",), extra={"frozen": fr})
    assert "w_frozen" in track and track["err_frozen"].notna().sum() > 5
    sk = te.live_skill_table(track)
    assert "frozen" in set(sk["key"])


def test_update_study_table(data):
    ct, imp, tgt, train = data
    te.FROZEN_MODELS["_test"] = ("traj:Bayesian Ridge", "trajm:Bayesian Ridge")
    try:
        cache = {}
        tab = te.frozen_update_study(ct, imp, tgt, train, "_test", 0.8, intervals=(10, 0), fit_cache=cache)
    finally:
        te.FROZEN_MODELS.pop("_test")
    assert len(tab) == 6 and (tab["status"] == "ok").all()
    never = tab[(tab["Retrain every"] == "never")]
    assert (never["Refits per 100 cycles"] < tab[tab["Retrain every"] == "10"]["Refits per 100 cycles"].max()).all()
    assert tab["Forecast RMSE (SOH pts)"].notna().all()
