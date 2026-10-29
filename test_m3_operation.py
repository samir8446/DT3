"""v5.33.0 · Mission 3 stage 2: operation options, measured economics, physics-scaled ageing."""
import numpy as np
import pandas as pd
import pytest

import twin_engine as te


@pytest.fixture(scope="module")
def demo():
    master, imp, _ = te.make_synthetic_master(
        n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24), currents=(2, 2, 2, 2, 1, 4, 4, 2),
        k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    ct, _ = te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(master))
    groups = te.condition_groups(ct)
    cells = sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(ct, g)})
    cds, re_ = te._nasa_inputs(ct, imp, cells)
    h = {"data": cds, "re": re_, "params": dict(te.HYBRID_TRUTH), "version": 2}
    return ct, groups, h


def test_economics_are_measured(demo):
    ct, groups, h = demo
    eco = te.m3_condition_economics(ct)
    first = ct[~ct["outlier"]].sort_values("n").groupby("Cell_ID").head(10)
    assert eco.loc["S001", "Wh out"] == pytest.approx(first[first.Cell_ID == "S001"]["E_dis_Wh"].median())
    assert (eco["efficiency"] > 0.5).all() and (eco["efficiency"] <= 1).all()


def test_options_are_tested_combinations_only(demo):
    ct, groups, h = demo
    opts = te.m3_operation_options(h, ct, "S001", groups)
    assert opts["own"].sum() == 1
    meta = te.m3_condition_economics(ct)
    for _, o in opts.iterrows():                       # every option was run by at least one named battery
        for b in o["tested on"].split(", "):
            assert abs(round(meta.loc[b, "I_dis"] / 0.5) * 0.5 - o["I_dis"]) < 1e-9


def test_fade_scale_and_own_life_reproduce_the_measured_fade(demo):
    ct, groups, h = demo
    cell = "S001"
    k = te.m3_fade_scale(h, cell)
    assert 0.1 < k < 10
    own = te.m3_operation_options(h, ct, cell, groups).query("own").iloc[0].to_dict()
    r = te.m3_operation_life(h, cell, own, te.M3OperationEconomics(soh_eol=0.5, max_cycles=150), k)
    meas = ct[(ct.Cell_ID == cell) & ~ct.outlier].sort_values("n")
    n = np.arange(1, len(r["soh path"]) + 1)
    sim = np.interp(meas["n"], n, r["soh path"])
    assert np.sqrt(np.mean((sim - meas["SOH"]) ** 2)) < 0.03   # within 3 SOH points over the record


def test_study_prices_one_battery_and_compares_to_own(demo):
    ct, groups, h = demo
    r = te.m3_operation_study(h, ct, "S003", te.M3OperationEconomics(), groups)
    assert r["replacement cost"].nunique() == 1
    assert r.loc[r["own"], "vs own (per hour)"].iloc[0] == pytest.approx(0.0)
    assert (r["cycles to EOL"] > 0).all() and r["reached EOL"].all()
    cheap = te.m3_operation_study(h, ct, "S003", te.M3OperationEconomics(battery_cost_cycles=0.0), groups)
    assert np.allclose(cheap["profit rate (per hour)"], cheap["J_op per hour"])


# ---- v5.35.0: depth economics measured inside batteries (cell-to-cell confound removed) ----
def test_depth_table_is_monotone_and_matches_capacity(demo):
    ct, groups, h = demo
    master, imp, _ = te.make_synthetic_master(
        n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24), currents=(2, 2, 2, 2, 1, 4, 4, 2),
        k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    store = te.ParquetStore.from_dataframe(master)
    dep = te.m3_depth_table(store, ct, ["S001", "S003"])
    for b, d in dep.groupby("battery"):
        d = d.sort_values("V", ascending=False)
        assert (np.diff(d["Ah"]) >= -1e-9).all() and (np.diff(d["Wh"]) >= -1e-9).all()   # deeper = more
        cap = ct[(ct.Cell_ID == b) & ~ct.outlier].sort_values("n").head(10)["Capacity_Ah"].median()
        assert d["Ah"].max() == pytest.approx(cap, rel=0.03)


def test_option_economics_follow_the_target_battery_not_the_peers(demo):
    ct, groups, h = demo
    cap = {"S001": 2.2, "S002": 1.6, "S005": 1.9, "S008": 1.7, "S003": 2.0, "S004": 1.8, "S006": 2.1, "S007": 1.5}
    dep = pd.DataFrame([{"battery": b, "V": te.M3_DEPTH_REF, "Ah": q, "Wh": 3.5 * q, "hours": 0.5 * q}
                        for b, q in cap.items()])
    opts = te.m3_operation_options(h, ct, "S001", groups, dep)
    own = opts[opts["own"]].iloc[0]
    assert own["economics"] == "within battery"
    assert own["Wh out"] == pytest.approx(3.5 * cap["S001"])     # peers' capacities do not leak in
    assert own["Ah"] == pytest.approx(cap["S001"])



def test_charging_side_comes_from_the_target_battery(demo):
    ct, groups, h = demo
    cap = {"S001": 2.2, "S002": 1.6, "S005": 1.9, "S008": 1.7, "S003": 2.0, "S004": 1.8, "S006": 2.1, "S007": 1.5}
    dep = pd.DataFrame([{"battery": b, "V": te.M3_DEPTH_REF, "Ah": q, "Wh": 3.5 * q, "hours": 0.5 * q}
                        for b, q in cap.items()])
    eco = te.m3_condition_economics(ct).loc["S001"]
    o = te.m3_operation_options(h, ct, "S001", groups, dep).query("own").iloc[0]
    assert o["Wh in"] == pytest.approx(o["Ah"] * eco["Wh in"] / eco["Ah"])
    assert o["Ah moved"] == pytest.approx(o["Ah"] * eco["Ah moved"] / eco["Ah"])
