"""v5.36.0 · Mission 3 stage 3: live forecast window and lifecycle profit of the strategies."""
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


def _lives(ct, cells, soh_eol=0.9, bias=3.0):
    out = {}
    for c in cells:
        m = ct[(ct.Cell_ID == c) & ~ct.outlier].sort_values("n")
        E = te.eol_cycle_from_replay(m[["n", "SOH"]].reset_index(drop=True), soh_eol)
        if E is None:
            continue
        tl = pd.DataFrame({"cycle": m["n"].to_numpy(float), "SOH measured": m["SOH"].to_numpy(float),
                           "EOL forecast": np.where(m["n"] >= 20, E + bias, np.nan), "status": "Normal"})
        out[c] = (tl, "Reference", float(E))
    return out


def test_live_window_uses_only_training_batteries(demo):
    ct, groups, h = demo
    train = ["S002", "S005", "S008"]
    a = te.frozen_live_window(ct, "S001", train)
    short = ct[~((ct.Cell_ID == "S001") & (ct.n > 60))]                 # target's record cut: no change
    assert te.frozen_live_window(short, "S001", train) == a
    lens = [ct.loc[ct.Cell_ID == c, "n"].max() for c in train]
    assert a[0] == max(10, round(0.3 * np.median(lens))) and a[1] >= a[0] + 20


def test_lifecycle_bound_and_consistency(demo):
    ct, groups, h = demo
    ref = te.protocol_cells(ct, "Reference")
    lives = _lives(ct, ref)
    assert len(lives) >= 3
    per, summ = te.m3_lifecycle_study(h, ct, lives, groups, None, te.M3OperationEconomics(soh_eol=0.9),
                                      soh_eols={c: 0.9 for c in lives})
    assert len(per) == len(lives)
    for s_ in ("do nothing", "usual practice", "twin replacement", "twin strategy"):
        assert (per["upper bound"] >= per[s_] - 1e-9).all(), s_
    assert summ.loc[("all", "do nothing"), "share of the possible gain"] == pytest.approx(0.0)
    assert summ.loc[("all", "upper bound"), "share of the possible gain"] == pytest.approx(1.0)
    assert ("Reference", "twin strategy") in summ.index
    same = per["chosen option"] == per["own option"]
    assert np.allclose(per.loc[same, "twin strategy"], per.loc[same, "twin replacement"])
    assert (per["chosen on"] == "other batteries of the group").all()



def test_lives_count_every_calendar_cycle():
    tl = pd.DataFrame({"cycle": [1.0, 2, 3, 30, 31, 60], "SOH measured": [1.0, 0.99, 0.98, 0.9, 0.89, 0.8],
                       "EOL forecast": np.nan, "status": "Normal"})
    n, soh = te._m3_series(tl)
    assert len(n) == 60 and n[0] == 1 and soh[29] == pytest.approx(0.9)          # gaps 4-29, 32-59 filled
    r = te.m3_fixed_age(tl, 60, 10, te.M3Economics())
    assert r["cycles used"] == 10 and np.isfinite(r["profit rate"])               # an age inside a gap still works
