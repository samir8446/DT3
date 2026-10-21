"""dQ/dV degradation-mode indicators (twin_engine section 45) recover a known answer; PINN v2 explains with them.

Synthetic cells fade by pure capacity scaling while their resistance grows: the discharge scale must follow SOH,
the LLI-like shift (charge + discharge) must stay near zero, and the polarisation (charge - discharge) must grow.
"""
import numpy as np
import pytest
import twin_engine as te


@pytest.fixture(scope="module")
def synth():
    master, imp, _ = te.make_synthetic_master(n_cells=5, n_cycles=60, seed=4, ambients=(24, 43, 4, 24, 43),
                                              currents=(2, 2, 2, 4, 1))
    store = te.ParquetStore.from_dataframe(master)
    ct, _ = te.build_cycle_table_with_report(store)
    return store, ct, imp, sorted(ct["Cell_ID"].unique())


def test_pure_scaling_is_recovered(synth):
    store, ct, _, cells = synth
    f, curves = te.dqdv_cell(store, ct, cells[0])
    assert {"discharge", "charge"} <= set(f["mode"]) and curves["discharge"]
    md = te.dqdv_modes(f).sort_values("n")
    end, start = md.tail(5).median(numeric_only=True), md.head(3).median(numeric_only=True)
    assert abs(end["dq_scale"] - end["SOH"]) < 0.02                     # LAM-like scale follows the capacity
    assert abs(end["dq_lli_mV"]) < 15                                    # no lithium-inventory shift in the truth
    assert end["dq_pol_mV"] > 10 and end["dq_pol_mV"] > start["dq_pol_mV"]   # resistance growth = polarisation


def test_attach_dqdv_columns(synth):
    store, ct, _, cells = synth
    ind = te.dqdv_indicators(store, ct, cells[:2])
    ct2 = te.attach_dqdv(ct, ind)
    assert len(ct2) == len(ct) and {"dq_scale", "dq_lli_mV", "dq_pol_mV"} <= set(ct2.columns)
    assert ct2.loc[ct2["Cell_ID"] == cells[0], "dq_scale"].notna().sum() > 20


def test_pinn_v2_explains_with_the_evidence(synth):
    store, ct, imp, cells = synth
    ct2 = te.attach_dqdv(ct, te.dqdv_indicators(store, ct, cells))
    r = te.pinn_explain(ct2, imp, cells[0], cells, version=2, seeds=(0, 1), model_params={"iterations": 800})
    last = r["frames"].sort_values("n").groupby("seed").tail(1)
    assert (abs(last["SOH (model)"] - last["SOH observed"]) < 0.03).all()     # reproduces the capacity
    ea = r["params"]["Activation energy, SEI (kJ/mol)"]
    assert ((ea >= 20) & (ea <= 80)).all()                                   # physical bounds hold
    assert abs(r["attribution"]["share mean"].sum() - 1) < 1e-6
    r1 = te.pinn_explain(ct, imp, cells[0], cells, version=1, seeds=(0,), model_params={"iterations": 800})
    assert set(te.MECHANISMS) <= set(r1["frames"].columns)


def test_dqdv_rules_are_off_by_default(synth):
    """v5.16.1: the global dQ/dV scale tracks capacity whatever the mechanism, so it must not constrain the split."""
    store, ct, imp, cells = synth
    ct2 = te.attach_dqdv(ct, te.dqdv_indicators(store, ct, cells))
    kw = dict(seeds=(0,), model_params={"iterations": 600})
    a = te.pinn_explain(ct2, imp, cells[0], cells, version=2, **kw)
    ct3 = ct2.assign(dq_scale=0.5, dq_lli_mV=50.0)                       # absurd evidence: must change nothing
    b = te.pinn_explain(ct3, imp, cells[0], cells, version=2, **kw)
    cols = list(te.MECHANISMS)
    assert np.allclose(a["frames"][cols].to_numpy(), b["frames"][cols].to_numpy(), atol=1e-12)
