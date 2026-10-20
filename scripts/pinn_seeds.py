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
in pinn_seeds_summary.csv. The PINN's mechanism attribution (SEI growth, lithium plating, loss of active material at
the battery's last recorded cycle) and its learned physics parameters are saved per seed to pinn_seeds_mechanisms.csv
and summarised in pinn_seeds_mechanism_stability.csv: an attribution that changes with the seed is not physical.
Send all four files.
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
    captured = []                                           # PINN forecasts of the current battery (final fits only)

    def capture(*args, _s=0, **kw):
        out = base_train(*args, seed=_s, **kw)
        if getattr(out, "mechanisms", None) is not None and out.model.startswith("PINN · physics"):
            captured.append(out)
        return out

    mech_path = Path(a.out).with_name("pinn_seeds_mechanisms.csv")
    for grp in a.groups:
        cells = te.protocol_cells(ct, grp)                  # membership from the recorded conditions, as in v5.14+
        if len(cells) < 3:
            print(f"{grp}: {len(cells)} usable batteries, skipped")
            continue
        # as in the assessment: Mixed conditions at each battery's reference condition, every other group as recorded
        ct_run = te.reference_condition_ct(ct) if grp == "Mixed conditions" else ct
        for seed in a.seeds:
            te.train_pinn_model = (lambda *args, _s=seed, **kw: capture(*args, _s=_s, **kw))
            for c in cells:
                if (grp, seed, c) in key_done:
                    continue
                t0 = time.time()
                captured.clear()
                res = te.run_protocol(store, ct_run, imp, cells, KEYS, (a.origin,), eol_ah=a.eol_ah, only_cells=[c])
                res = res.assign(group=grp, seed=seed, **{"engine": te.ENGINE_VERSION, "threads": 1})
                mrow = mechanism_row(captured[-1], ct_run, c) if captured else {"status": "no PINN forecast"}
                pd.DataFrame([{"group": grp, "seed": seed, "Cell_ID": c, **mrow}]).to_csv(
                    mech_path, mode="a", header=not mech_path.exists(), index=False)
                res.to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
                ok = res[res["status"] == "ok"].set_index("key")["RMSE (SOH pts)"]
                print(f"{grp:17s} seed {seed} {c}: PINN {ok.get(KEYS[0], np.nan):.2f} · ablation "
                      f"{ok.get(KEYS[1], np.nan):.2f}  ({time.time() - t0:.0f} s)", flush=True)
    te.train_pinn_model = base_train
    summarise(pd.read_csv(a.out))
    if mech_path.exists():
        summarise_mechanisms(pd.read_csv(mech_path))


MECH = ("LLI by SEI growth", "LLI by lithium plating", "Loss of active material")


def mechanism_row(f, ct, cell) -> dict:
    """The PINN's attribution at the battery's last recorded cycle: the three mechanism states (fractions of the
    initial capacity), their shares of the total, and the learned physics parameters."""
    m = f.mechanisms
    n = np.asarray(m["n"], float)
    n_last = float(ct.loc[(ct["Cell_ID"] == cell) & ~ct["outlier"], "n"].max())
    i = int(np.argmin(np.abs(n - n_last)))
    vals = {k: float(np.asarray(m[k], float)[i]) for k in MECH}
    tot = sum(max(v, 0.0) for v in vals.values())
    row = {"status": "ok", "origin cycle": f.n0, "cycle": float(n[i])}
    row.update({f"{k} (fraction)": v for k, v in vals.items()})
    row.update({f"{k} (share)": (max(v, 0.0) / tot if tot > 0 else np.nan) for k, v in vals.items()})
    row.update({f"param: {k}": v for k, v in (f.inputs_at_origin or {}).items() if not k.startswith("At the last cycle")})
    return row


def summarise_mechanisms(m: pd.DataFrame) -> None:
    """Is the attribution stable across seeds? Per battery: mean and SD of each mechanism's share; per group: the mean
    SD and how often the dominant mechanism stays the same across seeds."""
    ok = m[m["status"] == "ok"]
    if ok.empty:
        return
    sh = [f"{k} (share)" for k in MECH]
    rows = []
    for (grp, c), d in ok.groupby(["group", "Cell_ID"], sort=False):
        dom = d[sh].idxmax(axis=1).str.replace(" (share)", "", regex=False)
        row = {"group": grp, "Cell_ID": c, "seeds": len(d),
               "dominant mechanism (most seeds)": dom.mode().iloc[0],
               "same dominant mechanism in": f"{int((dom == dom.mode().iloc[0]).sum())}/{len(d)} seeds"}
        for k in MECH:
            row[f"{k}: share mean"] = d[f"{k} (share)"].mean()
            row[f"{k}: share SD"] = d[f"{k} (share)"].std()
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv("pinn_seeds_mechanism_stability.csv", index=False)
    print("\nMechanism attribution at the last recorded cycle (share of the total; mean ± SD over seeds)")
    for _, r in s.iterrows():
        parts = " · ".join(f"{k.split(' by ')[-1] if ' by ' in k else 'LAM'} {r[f'{k}: share mean']:.0%} ± "
                           f"{r[f'{k}: share SD']:.0%}" for k in MECH)
        print(f"  {r['group']:17s} {r['Cell_ID']}: {parts} · dominant: {r['dominant mechanism (most seeds)']} "
              f"({r['same dominant mechanism in']})")
    print("Saved pinn_seeds_mechanisms.csv and pinn_seeds_mechanism_stability.csv.")


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
