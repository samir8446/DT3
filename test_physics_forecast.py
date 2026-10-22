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


@pytest.mark.parametrize("kind", ["phys", "hyb"])
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
