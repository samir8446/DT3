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
