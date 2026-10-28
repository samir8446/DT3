"""v5.32.0 · Mission 3 rebuilt, stage 1: replacement decisions on real lives using only what the twin knew."""
import numpy as np
import pandas as pd
import pytest

import twin_engine as te


def _life(eol=80, fc_bias=0.0, start=20, n_last=100, rate=-0.004):
    n = np.arange(1, n_last + 1, dtype=float)
    soh = 1 + rate * n
    fc = np.where(n >= start, eol + fc_bias, np.nan)
    return pd.DataFrame({"cycle": n, "SOH measured": soh, "EOL forecast": fc, "status": "Normal"})


def test_life_profit_accounts_unplanned_cost_and_downtime():
    e = te.M3Economics(battery_cost_cycles=50, unplanned_factor=3, lead_cycles=10)
    tl = _life()
    n, soh = tl["cycle"].to_numpy(), tl["SOH measured"].to_numpy()
    p = te.m3_life_profit(n, soh, 80, False, e)
    u = te.m3_life_profit(n, soh, 80, True, e)
    assert p["J_maint"] == 50 and u["J_maint"] == 150 and u["downtime"] == 10
    assert p["J_op"] == pytest.approx(soh[:80].sum())
    assert u["profit rate"] < p["profit rate"]


def test_twin_policy_uses_only_the_forecast_and_lead_time():
    e = te.M3Economics(lead_cycles=10)
    good = te.m3_twin_policy(_life(eol=80), 80, e)                   # accurate forecast: planned, on time
    assert not good["unplanned"] and good["ordered at"] == 70 and good["replace at"] == 80
    late = te.m3_twin_policy(_life(eol=80, fc_bias=8), 80, e)       # forecast 8 cycles late: misses
    assert late["unplanned"] and late["replace at"] == 80
    safe = te.m3_twin_policy(_life(eol=80, fc_bias=8), 80, e, margin=10)
    assert not safe["unplanned"] and safe["replace at"] <= 80


def test_twin_policy_ignores_data_after_end_of_life_and_age_limit_fallback():
    e = te.M3Economics(lead_cycles=10)
    tl = _life(eol=30, start=25)                                     # forecast starts too late to warn
    assert te.m3_twin_policy(tl, 30, e)["unplanned"]
    r = te.m3_twin_policy(tl, 30, e, age_limit=28)                   # age rule orders at 18, arrives at 28
    assert not r["unplanned"] and r["replace at"] == 28


def test_oracle_is_an_upper_bound_for_planned_policies():
    e = te.M3Economics()
    tl = _life()
    o = te.m3_oracle(tl, 80, e)
    for t in (40, 60, 80):
        assert o["profit rate"] >= te.m3_fixed_age(tl, 80, t, e)["profit rate"] - 1e-12


def test_study_is_leave_one_out_and_reports_shares():
    lives = {f"C{i}": (_life(eol=E, fc_bias=b), "G", float(E))
             for i, (E, b) in enumerate([(70, 2), (80, 4), (90, 3), (85, 5)])}
    per, summ = te.m3_maintenance_study(lives, te.M3Economics(lead_cycles=5))
    assert set(summ.index) >= {"perfect knowledge", "twin-informed", "best fixed age", "run to end of life"}
    assert summ.loc["perfect knowledge", "share of perfect-knowledge gain"] == pytest.approx(1.0)
    assert summ.loc["run to end of life", "share of perfect-knowledge gain"] == pytest.approx(0.0)
    assert summ.loc["run to end of life", "unplanned"] == 4
    # the margin chosen for one battery must not change when that battery's own data change
    a = per[(per.battery == "C0") & (per.policy == "twin-informed")]["chosen margin"].iloc[0]
    lives2 = dict(lives, C0=(_life(eol=70, fc_bias=40), "G", 70.0))
    per2, _ = te.m3_maintenance_study(lives2, te.M3Economics(lead_cycles=5))
    assert per2[(per2.battery == "C0") & (per2.policy == "twin-informed")]["chosen margin"].iloc[0] == a
