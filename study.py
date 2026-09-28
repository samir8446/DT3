"""Offline cohort study: answers the Mission questions on a full dataset and writes the evidence.

    python study.py --master data/battery_master_data.parquet --imp data/impedance.parquet --out results/study
    python study.py --synthetic --out results/study_demo

Outputs: forecast scores (validation.csv), group x model summary, paired tests, mechanism shares and checks,
health-indicator ranking, stress-factor regression and a markdown summary (summary.md) with a manifest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import twin_engine as te


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:                     # tabulate not installed
        return "```\n" + df.to_string() + "\n```"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master")
    ap.add_argument("--imp")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--eol-ah", type=float, default=te.DEFAULT_EOL_AH)
    ap.add_argument("--origins", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--out", default="results/study")
    a = ap.parse_args()
    t0 = time.time()
    if a.synthetic:
        master, imp, _ = te.make_synthetic_master(n_cells=8, n_cycles=110, ambients=(24, 4, 43, 24, 24, 4, 43, 34),
                                                  currents=(2, 2, 2, 4, 2, 2, 2, 2), seed=0, noise_v=0.01)
        store = te.ParquetStore.from_dataframe(master)
    else:
        store = te.ParquetStore(a.master)
        imp = pd.read_parquet(a.imp) if a.imp else None
    ct = te.build_cycle_table(store)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = te.condition_groups(ct)
    val, mech = te.cohort_validation(store, ct, imp, a.eol_ah, a.origins, a.horizon,
                                     progress=lambda f, m: print(f"[{100 * f:5.1f}%] {m}", flush=True))
    summ = te.cohort_summary(val)
    paired = te.paired_model_test(val, "ens", "twin")
    checks = te.mechanism_checks(mech)
    hi = te.rank_health_indicators(ct, imp)
    sf = te.stress_factor_regression(ct)
    groups.to_csv(out / "groups.csv")
    val.to_csv(out / "validation.csv", index=False)
    summ.to_csv(out / "summary.csv")
    paired.to_csv(out / "paired_tests.csv")
    mech.to_csv(out / "mechanism_shares.csv", index=False)
    checks.to_csv(out / "mechanism_checks.csv")
    hi.to_csv(out / "health_indicators.csv")
    if sf.get("available"):
        sf["coefficients"].to_csv(out / "stress_factors.csv")
    md = ["# Study results", "", f"Engine {te.ENGINE_VERSION} · {len(groups)} batteries · EOL {a.eol_ah} Ah", "",
          "## Evaluation groups", _md(groups["Group"].value_counts().to_frame()), "",
          "## Forecast error by group and model (20-cycle horizon)", _md(summ.round(4)), "",
          "## Live ensemble vs twin (paired Wilcoxon)", _md(paired.round(4)), "",
          "## Mechanism physics checks", _md(checks.round(3)), "",
          "## Health indicators", _md(hi.head(8).round(3))]
    (out / "summary.md").write_text("\n".join(md))
    (out / "manifest.json").write_text(json.dumps(te.run_manifest(vars(a), getattr(store, "key", "synthetic"),
                                                                  {"seconds": time.time() - t0}), indent=2, default=str))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
