"""Mechanism-recovery test (twin_engine section 47): known split, can the shared physics recover it?"""
import twin_engine as te


def test_shared_physics_recovers_the_known_split():
    ct, imp, truth = te.recovery_cohort()
    ts = te.recovery_true_shares(truth)
    assert ts.loc["H1"].idxmax() == "LLI by SEI growth" and ts.loc["C1"].idxmax() == "LLI by lithium plating"
    fits, true_cost, _ = te.recovery_physics_fit(ct, imp, list(te.RECOVERY_CELLS), restarts=6)
    tab, s = te.recovery_physics_table(fits, true_cost, ts)
    assert s["mean error"] < 3 and s["dominant correct"] == 1.0
    assert abs(s["learned"]["Ea"] - te.RECOVERY_TRUTH["Ea"]) < 5


def test_nasa_physics_fit_runs_on_cycle_tables():
    master, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=80, seed=7, ambients=(24, 43, 4, 24, 43, 4),
                                              currents=(2, 2, 2, 4, 1, 1))
    ct = te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(master))[0]
    cells = sorted(ct["Cell_ID"].unique())
    res = te.nasa_physics_fit(ct, imp, cells, restarts=4)
    tab, s = te.nasa_physics_table(res, te.condition_groups(ct))
    assert len(tab) == len(cells) and s["starts"] >= 1
    shares = tab[[f"{m} (%)" for m in te.MECHANISMS]].sum(axis=1)
    assert ((shares - 100).abs() < 1e-6).all()
    assert 20 <= s["learned"]["Ea"] <= 80
    tr = te.nasa_physics_trajectory(res, cells[0])
    assert {"SOH observed", "SOH (model)"} <= set(tr.columns) and len(tr) > 10


def test_nasa_physics_v2_no_plating_when_warm():
    master, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=80, seed=7, ambients=(24, 43, 4, 24, 43, 4),
                                              currents=(2, 2, 2, 4, 1, 1))
    ct = te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(master))[0]
    cells = sorted(ct["Cell_ID"].unique())
    res = te.nasa_physics_fit(ct, imp, cells, restarts=3, version=2, fit_with="Re")
    tab, s = te.nasa_physics_table(res, te.condition_groups(ct))
    warm = tab["cell temperature (°C)"] > te.T_COLD_C
    assert warm.any() and (tab.loc[warm, "LLI by lithium plating (%)"] == 0).all()
    assert s["version"] == 2 and not s["fit_r"] and isinstance(s["at_bound"], list)
