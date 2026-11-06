"""SOH estimation (engine section 59, v5.43.0): the test battery's capacity is never used."""
import warnings

import numpy as np
import pytest

import twin_engine as te

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def ct():
    m, imp, _ = te.make_synthetic_master(n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24),
                                         currents=(2, 2, 2, 2, 1, 4, 4, 2),
                                         k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    return te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(m))[0]


def test_estimation_table_relative_indicators(ct):
    tab = te.estimation_table(ct, ["S001", "S005"])
    assert set(tab["Cell_ID"]) == {"S001", "S005"}
    for f in te.ESTIMATION_FEATURES:
        first = tab.groupby("Cell_ID")[f + "_rel"].apply(lambda s: s.head(5).median())
        assert np.allclose(first.to_numpy(), 1.0, atol=0.02)


@pytest.mark.parametrize("kind,params", [("GPR", None), ("PINN-est", {"iterations": 120})])
def test_estimate_finite(ct, kind, params):
    r = te.estimate_soh(ct, "S001", ["S005", "S008"], kind, params)
    assert len(r["pred"]) == len(r["soh"]) > 50 and np.all(np.isfinite(r["pred"]))


@pytest.mark.parametrize("kind,params", [("GPR", None), ("PINN-est", {"iterations": 120})])
def test_target_capacity_never_used(ct, kind, params):
    """Changing the TEST battery's capacity / SOH heavily (scaling, oscillation, planted far points, low first
    cycles) must leave its estimate identical on EVERY cycle: only its measured indicators reach the model."""
    base = te.estimate_soh(ct, "S001", ["S005", "S008"], kind, params)
    ct2 = ct.copy()
    m = ct2["Cell_ID"] == "S001"
    idx = ct2.index[m]
    ct2.loc[m, "SOH"] = ct2.loc[m, "SOH"] * 0.85 + 0.05 * np.sin(ct2.loc[m, "n"])
    ct2.loc[idx[:4], "SOH"] *= 0.7                         # low first cycles
    ct2.loc[idx[[30, 60, 90]], "SOH"] *= 0.6               # far points
    ct2.loc[m, "Capacity_Ah"] = ct2.loc[m, "Capacity_Ah"] * 0.85
    ct2.loc[idx[[40, 70]], "outlier"] = True               # capacity-based outlier flags change too
    alt = te.estimate_soh(ct2, "S001", ["S005", "S008"], kind, params)
    assert np.array_equal(base["n_all"], alt["n_all"])
    assert np.allclose(base["pred_all"], alt["pred_all"], atol=1e-9)


def test_ga_estimation_runs(ct):
    out = te.ga_estimation(ct, ["S005", "S008", "S001"], pop=2, gens=1)
    assert set(te.GA_PINN_GENES) <= set(out["best"]) and len(out["table"]) >= 1
