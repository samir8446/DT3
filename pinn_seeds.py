"""PINN versus the same network without physics, over several random seeds.

Standalone study (no app change, feature freeze): it runs the accuracy assessment's own protocol for the two Level 4
models (pinn:PINN and pinn:PINN-ablation): each battery of a condition group held out in turn, the forecast made at
30 % of its life, scored on all later measured cycles. Only the random seed changes between repeats.

Numerical threads are pinned to one, so a run is exactly reproducible: in the app, the same seed gave different
ablation results between two runs (multithreaded floating point), which hid the seed effect itself.

Usage (in the project folder, next to twin_engine.py):
    python scripts/pinn_seeds.py --master battery_master_data.parquet --imp impedance_ground_truth.parquet
    python scripts/pinn_seeds.py --master ... --groups Hot Cold --seeds 0 1 2 3 4
Results are saved after every battery to pinn_seeds_runs.csv (a stopped run resumes where it stopped) and summarised
in pinn_seeds_summary.csv. Send both files.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                                   # before numpy is imported: reproducible arithmetic

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twin_engine as te  # noqa: E402

KEYS = ("pinn:PINN", "pinn:PINN-ablation")
LABEL = {"pinn:PINN": "PINN", "pinn:PINN-ablation": "Ablation (no physics)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="battery_master_data.parquet")
    ap.add_argument("--imp", default=None, help="impedance_ground_truth.parquet (optional)")
    ap.add_argument("--groups", nargs="+", default=["Hot", "Cold", "Mixed conditions"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--origin", type=float, default=0.3)
    ap.add_argument("--eol-ah", type=float, default=te.DEFAULT_EOL_AH)
    ap.add_argument("--out", default="pinn_seeds_runs.csv")
    a = ap.parse_args()

    print(f"twin_engine {te.ENGINE_VERSION} · groups {a.groups} · seeds {a.seeds} · origin {a.origin:.0%}")
    if te.ENGINE_VERSION != "5.15.1":
        print("WARNING: the Mission 2 results were produced with 5.15.1")
    store = te.ParquetStore(a.master)
    ct, _ = te.build_cycle_table_with_report(store)
    imp = te.load_impedance(a.imp) if a.imp and Path(a.imp).exists() else None
    done = pd.read_csv(a.out) if Path(a.out).exists() else pd.DataFrame()
    key_done = set(zip(done["group"], done["seed"], done["Cell_ID"])) if len(done) else set()

    base_train = te.train_pinn_model
    for grp in a.groups:
        cells = te.protocol_cells(ct, grp)                  # membership from the recorded conditions, as in v5.14+
        if len(cells) < 3:
            print(f"{grp}: {len(cells)} usable batteries, skipped")
            continue
        # as in the assessment: Mixed conditions at each battery's reference condition, every other group as recorded
        ct_run = te.reference_condition_ct(ct) if grp == "Mixed conditions" else ct
        for seed in a.seeds:
            te.train_pinn_model = (lambda *args, _s=seed, **kw: base_train(*args, seed=_s, **kw))
            for c in cells:
                if (grp, seed, c) in key_done:
                    continue
                t0 = time.time()
                res = te.run_protocol(store, ct_run, imp, cells, KEYS, (a.origin,), eol_ah=a.eol_ah, only_cells=[c])
                res = res.assign(group=grp, seed=seed, **{"engine": te.ENGINE_VERSION, "threads": 1})
                res.to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
                ok = res[res["status"] == "ok"].set_index("key")["RMSE (SOH pts)"]
                print(f"{grp:17s} seed {seed} {c}: PINN {ok.get(KEYS[0], np.nan):.2f} · ablation "
                      f"{ok.get(KEYS[1], np.nan):.2f}  ({time.time() - t0:.0f} s)", flush=True)
    te.train_pinn_model = base_train
    summarise(pd.read_csv(a.out))


def summarise(r: pd.DataFrame) -> None:
    ok = r[r["status"] == "ok"]
    rows = []
    for grp, d in ok.groupby("group", sort=False):
        piv = d.pivot_table(index=["Cell_ID", "seed"], columns="key", values="RMSE (SOH pts)").dropna()
        if piv.empty or not set(KEYS) <= set(piv.columns):
            continue
        diff = piv[KEYS[1]] - piv[KEYS[0]]                  # > 0: physics better
        per_batt = piv.groupby(level="Cell_ID").agg(["mean", "std"])
        pb = diff.groupby(level="Cell_ID").mean()
        rows.append({
            "group": grp, "batteries": piv.index.get_level_values(0).nunique(), "seeds": piv.index.get_level_values(1).nunique(),
            "PINN mean RMSE": piv[KEYS[0]].mean(), "Ablation mean RMSE": piv[KEYS[1]].mean(),
            "PINN better (battery x seed runs)": f"{int((diff > 0).sum())}/{len(diff)}",
            "PINN better on battery (seed-averaged)": f"{int((pb > 0).sum())}/{len(pb)}",
            "Mean physics gain (pts)": diff.mean(),
            "Seed spread PINN (mean SD)": per_batt[(KEYS[0], "std")].mean(),
            "Seed spread ablation (mean SD)": per_batt[(KEYS[1], "std")].mean()})
        print(f"\n{grp}: per battery, mean ± SD over seeds (SOH pts)")
        for c, row in per_batt.iterrows():
            print(f"  {c}: PINN {row[(KEYS[0], 'mean')]:.2f} ± {row[(KEYS[0], 'std')]:.2f} · ablation "
                  f"{row[(KEYS[1], 'mean')]:.2f} ± {row[(KEYS[1], 'std')]:.2f}")
    s = pd.DataFrame(rows)
    if len(s):
        s.to_csv("pinn_seeds_summary.csv", index=False)
        print("\n" + s.round(3).to_string(index=False))
        print("\nSaved pinn_seeds_summary.csv. Send it with pinn_seeds_runs.csv.")


if __name__ == "__main__":
    main()
