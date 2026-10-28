"""v5.30.0 · integrated twin (Mission 3, option B): snapshot causality and status, what-if guardrails, picklable
hybrid, version guard. Fast: built on the synthetic cohort, no network training."""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import twin_engine as te

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo():
    master, imp, _ = te.make_synthetic_master(
        n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24), currents=(2, 2, 2, 2, 1, 4, 4, 2),
        k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    ct, _ = te.build_cycle_table_with_report(te.ParquetStore.from_dataframe(master))
    groups = te.condition_groups(ct)
    cells = sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(ct, g)})
    cds, re_ = te._nasa_inputs(ct, imp, cells)
    h = {"data": cds, "re": re_, "params": dict(te.HYBRID_TRUTH), "version": 2,
         "delta": {c: np.zeros(len(cd["n"])) for c, cd in cds.items()}}
    return ct, imp, groups, h


def _frames(n0, n_last, soh0=0.9, rate=-0.002, width=0.02):
    out = {}
    for n in range(n0, n_last + 1):
        fut = np.arange(n + 1, 2 * n_last, dtype=float)
        med = soh0 + rate * (fut - n0)
        out[n] = (fut, med, med - width, med + width)
    return out


def _meas(n_last=100, soh0=1.0, rate=-0.002):
    n = np.arange(1, n_last + 1, dtype=float)
    return pd.DataFrame({"n": n, "SOH": soh0 + rate * n})


# ---- snapshot: causal and correct status ----
def test_snapshot_ignores_everything_after_today():
    meas, fr = _meas(), _frames(30, 100)
    a = te.integrated_snapshot(50, 0.7, meas, fr)
    meas2 = meas.copy()
    meas2.loc[meas2["n"] > 50, "SOH"] = 0.1                       # catastrophic future
    fr2 = {k: (v if k <= 50 else (v[0], v[1] - 0.5, v[2] - 0.5, v[3] - 0.5)) for k, v in fr.items()}
    b = te.integrated_snapshot(50, 0.7, meas2, fr2)
    for k in ("SOH measured", "EOL forecast", "EOL early", "EOL late", "status", "reason", "forecast from"):
        assert a[k] == b[k] or (pd.isna(a[k]) and pd.isna(b[k])), k
    assert a["forecast from"] == 50


def test_snapshot_status_levels():
    meas, fr = _meas(), _frames(30, 100)
    assert te.integrated_snapshot(20, 0.7, meas, fr)["reason"].startswith("the forecast starts")
    eol = te.integrated_snapshot(100, 0.85, meas, fr)            # measured SOH 0.8 < 0.85
    assert eol["status"] == "End of life reached"
    near = te.integrated_snapshot(55, 0.8, meas, fr, watch_cycles=30)
    assert near["status"] == "Watch" and np.isfinite(near["EOL early"])
    # v5.30.1: a wide band's early edge alone does not set Watch (median end of life far away)
    wide = _frames(30, 100, width=0.15)
    w = te.integrated_snapshot(40, 0.8, meas, wide, watch_cycles=10)
    assert w["EOL early"] - 40 <= 10 and w["EOL forecast"] - 40 > 10
    assert w["status"] == "Normal" and "early edge" in w["reason"] and w["band half-width (points)"] == pytest.approx(15)
    # end of life undefined (threshold above the initial capacity): never "End of life reached"
    u = te.integrated_snapshot(100, float("nan"), meas, fr)
    assert u["status"] != "End of life reached" and np.isnan(u["EOL forecast"])
    far = te.integrated_snapshot(31, 0.5, meas, fr, watch_cycles=10)
    assert far["status"] == "Normal"
    an = pd.DataFrame({"n": meas["n"], "SOH": meas["SOH"], "kappa": 1.0, "fleet alarm": meas["n"] >= 40,
                       "slope z": 0.0, "own alarm": False})
    assert te.integrated_snapshot(45, 0.5, meas, fr, an)["status"] == "Alarm"
    assert te.integrated_snapshot(35, 0.5, meas, fr, an)["status"] == "Normal"     # alarm not yet raised at 35


def test_timeline_one_row_per_cycle(demo):
    ct, imp, groups, h = demo
    c = next(iter(h["data"]))
    meas = ct[(ct["Cell_ID"] == c) & ~ct["outlier"]][["n", "SOH"]]
    tr = te.hybrid_trajectory(h, c)
    tl = te.integrated_timeline(0.7, meas, None, None, None, tr, tr)
    assert len(tl) == len(meas) and set(tl["status"]) <= set(te.INTEGRATED_STATUS)
    sh = tl[[f"{m} (%)" for m in (*te.MECHANISMS, te.HYBRID_UNEXPLAINED)]].dropna()
    assert np.allclose(sh.sum(axis=1), 100, atol=1e-6)
    assert np.allclose(tl["split gap (points)"].dropna(), 0)      # same trajectory twice: no model-form gap


# ---- what-if guardrails ----
def test_whatif_own_conditions_change_nothing(demo):
    ct, imp, groups, h = demo
    c = next(iter(h["data"]))
    d, s = te.whatif_projection(h, c, 0.7, groups=groups)
    assert np.allclose(d["physics SOH, own conditions"], d["physics SOH, changed conditions"])
    assert s["changed"] == {} and s["combination tested"]


def test_whatif_refuses_untested_and_unidentifiable(demo):
    ct, imp, groups, h = demo
    c = next(iter(h["data"]))
    with pytest.raises(ValueError, match="not tested"):
        te.whatif_projection(h, c, 0.7, groups=groups, T_C=60.0)
    ident = te.whatif_identifiability(h, groups).set_index("condition")
    assert ident.loc["Charge current (A)", "status"] == "not identifiable"
    for key, spec in te.WHATIF_CONDITIONS.items():
        tested = te.whatif_tested(h, groups)["values"][key]
        if len(tested) < 2:
            assert ident.loc[spec["label"], "status"] == "not identifiable"
            with pytest.raises(ValueError):
                te.whatif_projection(h, c, 0.7, groups=groups, **{key: tested[0] + 10 * spec["round"]})


def test_whatif_limited_when_sensitivity_at_bound(demo):
    ct, imp, groups, h = demo
    h2 = dict(h, params={**h["params"], "a_I": te.NASA_V2_BOUNDS["a_I"][0]})
    ident = te.whatif_identifiability(h2, groups).set_index("key")
    if len(te.whatif_tested(h2, groups)["values"]["I_dis"]) >= 2:
        assert ident.loc["I_dis", "status"] == "limited"


def test_whatif_tested_value_changes_the_projection(demo):
    ct, imp, groups, h = demo
    tested = te.whatif_tested(h, groups)
    c = next(iter(h["data"]))
    own_t = float(tested["by_battery"].loc[c, "T_C tested"])
    other = [v for v in tested["values"]["T_C"] if v != own_t]
    if other:
        d, s = te.whatif_projection(h, c, 0.7, groups=groups, T_C=other[0])
        assert not np.allclose(d["physics SOH, own conditions"], d["physics SOH, changed conditions"])
        rec = d["measured SOH"].notna()
        eff = d.loc[rec, "measured, with the condition effect"] - d.loc[rec, "measured SOH"]
        assert np.allclose(eff, (d["physics SOH, changed conditions"] - d["physics SOH, own conditions"])[rec])


# ---- the hybrid can be cached (v5.29 bug: Streamlit's cache could not pickle it) ----
def test_hybrid_portable_pickles(demo):
    ct, imp, groups, h = demo
    hy = te.nasa_hybrid_fit(ct, imp, list(h["data"]), fit_with="Re", restarts=2, lam=0.3, version=2)
    with pytest.raises(Exception):
        pickle.dumps(hy)
    p = pickle.loads(pickle.dumps(te.hybrid_portable(hy)))
    tab, _ = te.hybrid_table(p)
    assert len(tab) == len(h["data"]) and "predict" not in p


def test_app_requires_this_engine():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    req = re.search(r'REQUIRED_ENGINE = "([^"]+)"', app).group(1)
    assert req == te.ENGINE_VERSION == "5.34.0"
    for f in ("integrated_snapshot", "integrated_timeline", "whatif_projection", "whatif_identifiability",
              "hybrid_portable"):
        assert f in app and hasattr(te, f)


# ---- v5.31.0: conservative end of life from the twin's own track record ----
def _optimistic_case(true_rate=-0.004, fc_rate=-0.002):
    """Measured SOH fades at true_rate per cycle; every forecast (re-anchored to today's measurement) fades at the
    slower fc_rate: systematically optimistic, like the frozen forecasters on NASA."""
    n = np.arange(1, 121, dtype=float)
    meas = pd.DataFrame({"n": n, "SOH": 1.0 + true_rate * n})
    frames = {}
    for m in range(36, 121):
        fut = np.arange(m + 1, 300, dtype=float)
        med = (1.0 + true_rate * m) + fc_rate * (fut - m)
        frames[m] = (fut, med, med - 0.01, med + 0.01)
    return meas, frames


def test_optimism_rate_is_learned_and_causal():
    meas, frames = _optimistic_case()
    r = te.forecast_optimism(frames, meas, 80)
    assert len(r) >= 10 and np.allclose(r, 0.002)                 # forecast 0.002 per cycle too slow
    meas2 = meas.copy()
    meas2.loc[meas2["n"] > 80, "SOH"] = 0.0
    assert np.allclose(te.forecast_optimism(frames, meas2, 80), r)  # nothing after today


def test_conservative_eol_corrects_optimism_and_sets_watch():
    meas, frames = _optimistic_case()
    true_eol = (1 - 0.7) / 0.004                                    # 75
    s = te.integrated_snapshot(60, 0.7, meas, frames, watch_cycles=20, watch_q=0.5)
    assert s["EOL forecast"] > true_eol + 5                         # median: late
    assert abs(s["EOL conservative q50"] - true_eol) <= 1.5         # corrected
    assert s["status"] == "Watch" and "conservative 50 %" in s["reason"]
    m = te.integrated_snapshot(60, 0.7, meas, frames, watch_cycles=20)
    assert m["status"] == "Normal"                                  # median alone would not warn yet


def test_conservative_never_more_optimistic_and_needs_history():
    meas, frames = _optimistic_case(true_rate=-0.002, fc_rate=-0.004)   # pessimistic forecasts
    s = te.integrated_snapshot(80, 0.7, meas, frames, watch_q=0.9)
    assert s["EOL conservative q90"] == s["EOL forecast"]
    meas, frames = _optimistic_case()
    early = te.integrated_snapshot(38, 0.7, meas, frames, watch_q=0.75)  # 2 cycles of history
    assert early["calibration pairs"] < 10 and "too little track record" in early["watch basis"]
    assert early["EOL for watch"] == early["EOL forecast"]
