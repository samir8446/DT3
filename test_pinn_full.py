"""Full-input PINN (v5.29): same physics as PINN v1, plus the battery's measured state at the origin as inputs."""
import numpy as np
import pytest
import twin_engine as te


@pytest.fixture(scope="module")
def data():
    master, imp, _ = te.make_synthetic_master(n_cells=5, n_cycles=80, seed=5)
    store = te.ParquetStore.from_dataframe(master)
    ct = te.build_cycle_table_with_report(store)[0]
    return store, ct, imp, sorted(ct["Cell_ID"].unique())


def test_state_vector_uses_only_the_past(data):
    _, ct, imp, cells = data
    a = te.pinn_state_raw(ct, imp, cells[0], 30)
    bad = ct.copy()
    sel = (bad["Cell_ID"] == cells[0]) & (bad["n"] > 30)
    for col in ("SOH", "R_dc_ohm", "t_dis_s", "t_cv_s", "T_mean_C", "eff_energy", "V_mean_V"):
        if col in bad:
            bad.loc[sel, col] = bad.loc[sel, col] * 0.5
    b = te.pinn_state_raw(bad, imp, cells[0], 30)
    assert len(a) == len(te.PINN_FULL_FEATURES)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


@pytest.mark.parametrize("kind", ["PINN-full", "PINN-full-ablation"])
def test_full_pinn_forecast_is_causal(data, kind):
    _, ct, imp, cells = data
    kw = dict(model_params={"iterations": 300})
    f1 = te.train_pinn_model(ct, imp, cells[0], 30, kind, "forecast", cells[1:], **kw)
    bad = ct.copy()
    sel = (bad["Cell_ID"] == cells[0]) & (bad["n"] > 30)
    bad.loc[sel, "SOH"] = bad.loc[sel, "SOH"] * 0.5
    f2 = te.train_pinn_model(bad, imp, cells[0], 30, kind, "forecast", cells[1:], **kw)
    fut = f1.n_grid > 30
    assert np.allclose(f1.soh[fut], f2.soh[fut], atol=1e-9)


def test_assessment_runs_full_pinn(data):
    store, ct, imp, cells = data
    keys = ["shape", "pinn:PINN-full", "pinn:PINN-full-ablation"]
    res = te.run_protocol(store, ct, imp, cells[:4], keys, (0.4,), only_cells=[cells[0]])
    ok = res[res["status"] == "ok"]
    assert set(ok["key"]) == set(keys)
