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
