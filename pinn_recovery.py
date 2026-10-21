"""Mechanism-recovery test, step 1: does PINN v2 find the TRUE degradation split when the truth is known?

On NASA data nobody knows how much capacity was lost to SEI growth, lithium plating and loss of active material, so
the PINN's explanation cannot be checked there. Here batteries are simulated with KNOWN mechanisms, following the
same kind of laws PINN v2 assumes (the well-specified case):
    SEI      dLLI_SEI/da = k_sei * exp(Ea/R (1/T_ref - 1/T)) / (LLI_SEI + d0)           (self-limiting, Arrhenius)
    plating  dLLI_pl/da  = k_pl * exp(b_C * cold + a_c * ln(I_charge/1.5)) * (1 + c * 20 * LAM)
    LAM      dLAM/da     = k_lam * exp(a_I * ln(I_dis/2) + a_V * depth)
    SOH = (1 - LLI_SEI - LLI_pl)(1 - LAM),  R/R0 = 1 + g * LLI_SEI,  Re/Re0 = 1 + g_e * LLI_SEI   (+ noise)
over 4, 24 and 43 C, discharge currents 1-4 A, charge currents 0.75-1.5 A and cut-off voltages 2.0-2.7 V, so each
mechanism dominates somewhere (hot: SEI, cold with fast charge: plating, deep discharge at high current: LAM).
The PINN sees only what it sees on NASA (capacity, resistance, impedance, conditions), never the split. For each
target battery it is fitted with the others (several seeds), and its split at the last cycle is compared with the
truth: error in share points, and whether the dominant mechanism is right.

Reading the result: if PINN v2 cannot recover the split here, where the data follow its own kind of physics, the
mechanisms are not identifiable from these measurements and its NASA attribution cannot be trusted. If it can, the
next test (step 2) uses PyBaMM cells, whose degradation does not follow the PINN's equations.

Usage (next to twin_engine.py):   python scripts/pinn_recovery.py            (~5-15 min)
                                  python scripts/pinn_recovery.py --quick    (2 targets, 2 seeds, ~2-4 min)
Outputs: pinn_recovery_truth.csv, pinn_recovery_results.csv. Send both.
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twin_engine as te  # noqa: E402   (the test itself lives in the engine, section 47, shared with the app)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--physics", action="store_true", help="only the pure physics fit (shared parameters, 30 starts)")
    a = ap.parse_args()
    ct, imp, truth = te.recovery_cohort()
    ts = te.recovery_true_shares(truth)
    ts.to_csv("pinn_recovery_truth.csv")
    print("True split at the last cycle (% of the capacity loss):\n" + (100 * ts).round(1).to_string())
    fits, tc, _ = te.recovery_physics_fit(ct, imp, list(te.RECOVERY_CELLS), restarts=30)
    tab, summ = te.recovery_physics_table(fits, tc, ts)
    tab.to_csv("physics_fit.csv", index=False)
    print(f"\nPure physics: error {summ['mean error']:.1f} pts, dominant correct {summ['dominant correct']:.0%}, "
          f"{summ['near-best starts']}/{summ['starts']} starts agree, Ea {summ['learned']['Ea']:.1f} kJ/mol")
    if a.physics:
        return
    targets = a.targets or (["H1", "C1"] if a.quick else ["H1", "H2", "R1", "R2", "C1", "C3"])
    seeds = tuple(a.seeds or ((0, 1) if a.quick else (0, 1, 2)))
    parts = []
    for ver in (1, 2):
        t = te.recovery_pinn_table(ct, imp, ts, targets, ver, seeds, a.iterations)
        parts.append(t.assign(model=f"PINN v{ver}"))
        print(f"PINN v{ver}: error {t['error (share points)'].mean():.1f} pts, dominant correct "
              f"{int(t['dominant correct'].sum())}/{len(t)}", flush=True)
    pd.concat(parts + [tab.assign(model="pure physics")], ignore_index=True).to_csv("pinn_recovery_results.csv", index=False)
    print("Saved pinn_recovery_truth.csv, physics_fit.csv and pinn_recovery_results.csv.")


if __name__ == "__main__":
    main()
