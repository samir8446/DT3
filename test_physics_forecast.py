"""Physics-based forecasters (twin_engine section 50): causal, and usable by the accuracy assessment."""
import numpy as np
import pytest
import twin_engine as te


@pytest.fixture(scope="module")
def data():
    master, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=80, seed=7, ambients=(24, 43, 4, 24, 43, 4),
                                              currents=(2, 2, 2, 4, 1, 1))
    store = te.ParquetStore.from_dataframe(master)
    ct = te.build_cycle_table_with_report(store)[0]
    return store, ct, imp, sorted(ct["Cell_ID"].unique())


@pytest.mark.parametrize("kind", ["phys", "hyb", "phys3", "hyb3", "phys2k", "hyb3k"])
def test_forecast_never_sees_the_future(data, kind):
    store, ct, imp, cells = data
    tgt, n0 = cells[0], 30
    a = te.physics_forecast(ct, imp, tgt, n0, kind, restarts=2)
    bad = ct.copy()
    sel = (bad["Cell_ID"] == tgt) & (bad["n"] > n0)
    for col in ("SOH", "Q_dis_Ah", "R_dc_ohm", "cum_Ah", "T_mean_C"):
        if col in bad:
            bad.loc[sel, col] = bad.loc[sel, col] * 0.6
    b = te.physics_forecast(bad, imp, tgt, n0, kind, restarts=2)
    fut = a.n_grid > n0
    assert np.allclose(a.soh[fut], b.soh[fut], atol=1e-10)
    assert np.isclose(a.soh[a.n_grid > n0][0], np.nanmedian(a.soh[(a.n_grid <= n0)][-3:]), atol=0.05)


def test_assessment_runs_physics_models(data):
    store, ct, imp, cells = data
    keys = ["shape", "phys:Shared physics v2", "hyb:Hybrid"]
    res = te.run_protocol(store, ct, imp, cells[:3], keys, (0.4,), only_cells=[cells[0]])
    ok = res[res["status"] == "ok"]
    assert set(ok["key"]) == set(keys) and ok["RMSE (SOH pts)"].notna().all()


def test_physics_v3_recovers_its_own_split_and_exponent():
    import math
    truth = {"k_sei": 0.025, "Ea": 50.0, "z": 0.9, "k_pl": 0.35, "b_C": 1.0, "k_lam": 0.06, "a_I": 0.6, "a_V": 1.5,
             "k_tr": 8.0, "m_c": 0.03, "g": 3.0, "g_e": 5.0}
    rng = np.random.default_rng(1)
    cds, re, sh = {}, {}, {}
    for cid, (T, I, Ic, Vc, N) in te.RECOVERY_CELLS.items():
        Tk = T + 273.15 + 0.5 * I
        cd = {"x_t": 1 / 298.15 - 1 / Tk, "cold": max(0.0, te.T_COLD_C - (Tk - 273.15)) / 10, "x_ic": math.log(Ic / 1.5),
              "x_i": math.log(I / 2.0), "depth": max(0.0, 2.7 - Vc) / 0.7}
        cd["a"] = np.cumsum(np.full(N, 2.0 * 0.85 / te.PINN_A_SCALE)); cd["n"] = np.arange(1, N + 1, dtype=float)
        s, pl, m = te.nasa_mechanisms_v3(truth, cd)
        cd["soh"] = (1 - s - pl) * (1 - m) + rng.normal(0, 0.003, N)
        cd["r"] = (1 + truth["g"] * s) * (1 + rng.normal(0, 0.01, N))
        re[cid] = (cd["n"][::10], (1 + truth["g_e"] * s[::10]) * (1 + rng.normal(0, 0.01, len(cd["n"][::10]))))
        v = np.array([s[-1], pl[-1], m[-1]]); sh[cid] = v / v.sum(); cds[cid] = cd
    p = te._shared_physics_core(cds, re, 3, "Re", 8, 0)[0]["params"]
    err = []
    for c, cd in cds.items():
        s, pl, m = te.nasa_mechanisms_v3(p, cd); v = np.array([s[-1], pl[-1], m[-1]])
        err.append(100 * np.mean(np.abs(v / v.sum() - sh[c])))
    assert np.mean(err) < 1.0 and abs(p["z"] - 0.9) < 0.1 and abs(p["m_c"] - 0.03) < 0.01


def test_ageing_speed_follows_the_battery(data):
    """A battery whose history fades twice as fast as recorded gets kappa > 1; the split mechanism is untouched."""
    store, ct, imp, cells = data
    tgt, n0 = cells[0], 40
    fast = ct.copy()
    sel = (fast["Cell_ID"] == tgt) & (fast["n"] <= n0)
    fast.loc[sel, "SOH"] = 1 - 2 * (1 - fast.loc[sel, "SOH"])
    f0 = te.physics_forecast(ct, imp, tgt, n0, "phys2k", restarts=2)
    f1 = te.physics_forecast(fast, imp, tgt, n0, "phys2k", restarts=2)
    assert f1.kappa > f0.kappa and te.PHYS_KAPPA_RANGE[0] <= f0.kappa <= te.PHYS_KAPPA_RANGE[1]
