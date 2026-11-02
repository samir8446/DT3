"""v5.38.0 · PINN v2: pooled training, cut-off input + LAM depth term, regeneration-robust causal anchor.
PINN v1 must stay identical (Mission 2 results depend on it)."""
import numpy as np
import pytest

import twin_engine as te


@pytest.fixture(scope="module")
def demo():
    master, imp, _ = te.make_synthetic_master(
        n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24), currents=(2, 2, 2, 2, 1, 4, 4, 2),
        k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    ct, _ = te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(master))
    return ct, imp


def test_v1_reference_values_unchanged(demo):
    """Frozen reference of PINN v1 on the demo (300 iterations, seed 0), recorded before v2 was added."""
    ct, imp = demo
    f = te.train_pinn_model(ct, imp, "S001", 36, "PINN", train_cells=["S002", "S005", "S008"],
                            model_params={"iterations": 300})
    assert np.allclose(f.soh[-3:], [0.789862, 0.787825, 0.785786], atol=2e-6)
    assert f.cv_rmse == pytest.approx(0.009694, abs=2e-6)


def test_v2_pools_all_constant_condition_batteries(demo):
    ct, imp = demo
    f = te.train_pinn_model(ct, imp, "S001", 36, "PINN-v2", train_cells=["S002"],
                            model_params={"iterations": 120})
    expected = sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(ct, g)} - {"S001"})
    assert sorted(f.train_cells) == expected                     # train_cells argument ignored: leave-one-out pool
    assert "S001" not in f.train_cells
    assert "Depth sensitivity of LAM (a_V)" in f.inputs_at_origin
    a = te.train_pinn_model(ct, imp, "S001", 36, "PINN-v2-ablation", model_params={"iterations": 120})
    assert "Depth sensitivity of LAM (a_V)" not in a.inputs_at_origin


def test_causal_anchor_ignores_a_regeneration_bump(demo):
    ct, _ = demo
    d = ct.copy()
    m = (d.Cell_ID == "S001") & (d.n.between(34, 36))
    d.loc[m, "SOH"] += 0.03                                     # capacity regeneration at the forecast origin
    plain = te._pinn_battery(d, "S001", n_max=36)
    causal = te._pinn_battery(d, "S001", n_max=36, causal=True)
    clean = te._pinn_battery(ct, "S001", n_max=36, causal=True)
    assert causal["soh"][-1] <= plain["soh"][-1] - 0.005        # the bump lifts the old anchor, not the causal one
    assert causal["soh"][-1] == pytest.approx(clean["soh"][-1], abs=0.006)
    assert "xV" not in plain and "xV" not in causal              # depth input only added inside v2 training


def test_v2_within_group_keeps_the_group_pool(demo):
    ct, imp = demo
    f = te.train_pinn_model(ct, imp, "S001", 36, "PINN-v2-group", train_cells=["S002", "S005", "S008"],
                            model_params={"iterations": 120})
    assert sorted(f.train_cells) == ["S002", "S005", "S008"]     # v1's within-group pool
    assert "Depth sensitivity of LAM (a_V)" in f.inputs_at_origin  # v2 features on
    assert "within group" in f.model


def test_twin_core_burn_in_anchor_ignores_low_first_cycle(demo):
    ct, _ = demo
    d = ct.copy()
    first = d[(d.Cell_ID == "S001") & ~d.outlier].n.min()
    d.loc[(d.Cell_ID == "S001") & (d.n == first), "SOH"] -= 0.12          # NASA-like low first cycle (5bp)
    locked = te._pinn_battery(d, "S001", n_max=36, causal=True)            # v2 anchor: locks onto it
    burned = te._pinn_battery(d, "S001", n_max=36, causal=True, burn_in=3)
    clean = te._pinn_battery(ct, "S001", n_max=36, causal=True, burn_in=3)
    assert burned["soh"][-1] > locked["soh"][-1] + 0.03
    assert burned["soh"][-1] == pytest.approx(clean["soh"][-1], abs=0.006)


def test_twin_core_is_a_seed_ensemble_usable_by_the_twin_stream(demo):
    ct, imp = demo
    f = te._protocol_forecast(te.PINN_TWIN_KEY, None, ct, imp, "S001", 36, ["S002", "S005", "S008"],
                              te.DEFAULT_EOL_AH)
    assert "3-seed ensemble" in f.model and hasattr(f, "mechanisms")
    assert (f.lo <= f.soh + 1e-9).all() and (f.hi >= f.soh - 1e-9).all()
    frames, track = te.frozen_stream(ct, imp, "S001", ["S002", "S005", "S008"], te.PINN_TWIN_KEY, None, 0.8, 0,
                                     "capacity")
    assert track.attrs["refits"] == 1 and len(frames) > 10               # trained once, re-anchored every cycle


def test_ga_pinn_key_roundtrip_and_search():
    g = {**te.GA_PINN_START, "sei_p": 0.5, "accel": 1, "iterations": 600}
    back = te.ga_pinn_genome(te.ga_pinn_key(g))
    assert back["sei_p"] == 0.5 and back["accel"] is True and back["iterations"] == 600 and back["hidden"] == 32
    toy = lambda g: (g["hidden"] - 48) ** 2 / 100 + g["sei_p"] + (0 if g["accel"] else 1) + abs(g["physics_weight"] - 1)
    r1 = te.ga_pinn_run(toy, pop_size=6, generations=6, seed=1)
    r2 = te.ga_pinn_run(toy, pop_size=6, generations=6, seed=1)
    assert r1["best_fitness"] <= toy(te.GA_PINN_START)                  # never worse than the starting settings
    assert r1["best_key"] == r2["best_key"] and r1["fitness"] == r2["fitness"]   # deterministic -> resumable
    part = te.ga_pinn_run(toy, pop_size=6, generations=6, seed=1, deadline=0.0)
    assert part["done"] is False


def test_ga_pinn_fitness_is_nested_and_winner_trains(demo):
    ct, imp = demo
    r = te.ga_pinn_fitness(ct, imp, "S001", 36, ["S002", "S005", "S008"], te.DEFAULT_EOL_AH,
                           {**te.GA_PINN_START, "iterations": 300})
    assert np.isfinite(r["fitness"]) and np.isfinite(r["RMSE_val (pts)"])
    g = {**te.GA_PINN_START, "sei_p": 0.0, "accel": 1, "iterations": 300}
    f = te._protocol_forecast(te.ga_pinn_key(g), None, ct, imp, "S001", 36, ["S002", "S005", "S008"],
                              te.DEFAULT_EOL_AH)
    assert f.model.startswith("GA-PINN") and np.isfinite(f.soh).all()
    assert "Damage acceleration (k_acc)" in f.inputs_at_origin
