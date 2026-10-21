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
