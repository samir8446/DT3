"""Mission 3 anomaly detector (twin_engine section 51): finds known knees, stays quiet on healthy batteries, causal."""
import numpy as np
import twin_engine as te


def test_validation_finds_the_knees_without_false_alarms():
    v = te.anomaly_validation(restarts=3)
    knees = v[v["battery"].isin(te.HYBRID_KNEE_CELLS)]
    healthy = v[~v["battery"].isin(te.HYBRID_KNEE_CELLS)]
    assert knees["result"].str.startswith("detected").all()
    assert (healthy["result"] == "quiet (correct)").all()
    assert (knees["knee size at the alarm (SOH points)"] < 3).all()


def test_replay_is_causal():
    cds, re, _, _ = te.hybrid_validation_cohort(0)
    cd = cds["R2"]
    a = te.anomaly_replay(None, cd)
    bad = {**cd, "soh": cd["soh"].copy()}
    cut = 60
    bad["soh"][cut:] = bad["soh"][cut:] - 0.2                # a catastrophic future
    b = te.anomaly_replay(None, bad)
    assert np.allclose(a["cusum"].to_numpy()[:cut], b["cusum"].to_numpy()[:cut])


def test_two_indicators_with_regeneration():
    v = te.anomaly_validation2(restarts=3)
    res = dict(zip(v["battery"], v["result"]))
    for c in te.HYBRID_KNEE_CELLS:
        assert res[c].startswith("OWN detected")
    for c in te.ANOMALY2_FAST:
        assert res[c].startswith("FLEET detected") and "false" not in res[c].lower()
    healthy = [c for c in res if c not in te.HYBRID_KNEE_CELLS and c not in te.ANOMALY2_FAST]
    assert all(res[c] == "quiet (correct)" for c in healthy)


def test_two_indicators_are_causal():
    cds, re, _ = te.anomaly_cohort2(0)
    cd = cds["R1"]
    a = te.anomaly_replay2(None, cd)
    bad = {**cd, "soh": cd["soh"].copy()}
    bad["soh"][90:] -= 0.2
    b = te.anomaly_replay2(None, bad)
    za, zb = a["slope z"].to_numpy()[:90], b["slope z"].to_numpy()[:90]
    assert np.allclose(np.nan_to_num(za), np.nan_to_num(zb))


def test_common_cause_classification():
    v = te.anomaly_validation_common(restarts=3)
    assert v["correct"].all(), v.to_string()


def test_event_log_reports_regeneration_and_alarms():
    cds, re, truth = te.anomaly_cohort2(0)
    tr = te.anomaly_replay2(None, cds["R2"])
    ev = te.anomaly_events(tr)
    assert (ev["indicator"] == "DATA").any()                       # regeneration jumps are reported as normal
    assert ((ev["indicator"] == "OWN") & (ev["kind"] == "alarm")).any()   # the knee
    assert ev["cycle"].is_monotonic_increasing


def test_cycle_status_labels_the_knee_and_regeneration():
    cds, re, truth = te.anomaly_cohort2(0)
    tr = te.anomaly_replay2(None, cds["R2"])
    stt = te.anomaly_cycle_status(tr)
    assert len(stt) == len(tr) and set(stt) <= set(te.DETECTION_STATUS)
    assert "Regeneration (normal)" in stt and "OWN alarm: fade steeper than its own past" in stt
    first_alarm = tr.attrs["first_own_alarm"]
    assert all(s_ != "OWN alarm: fade steeper than its own past"
               for s_, n in zip(stt, tr["n"]) if n < first_alarm)          # causal: nothing before the alarm
