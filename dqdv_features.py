"""Incremental-capacity (dQ/dV) features for every cycle of every NASA battery: step 1 of the full-signal plan.

A capacity curve shows only the SUM of the degradation mechanisms. The shape of the voltage curve separates them:
  - loss of active material (LAM) shrinks the incremental-capacity curve:   Q_n(V) ~ a * Q_ref(V), a < 1
  - loss of lithium inventory (LLI: SEI growth, plating) shifts the electrodes' operating windows against each
    other, which moves the dQ/dV peaks along the voltage axis:             Q_n(V) ~ Q_ref(V - delta)
For every clean cycle (the discharge, and the constant-current part of the charge) this script
  1. integrates the current into Q, corrects the voltage for the ohmic drop (V + I*R, R = the cycle's measured
     load-step resistance), smooths Q(V) and differentiates it: dQ/dV on a 5 mV grid;
  2. records the main dQ/dV peaks: voltage position, height, area;
  3. fits Q_n(V) = a * Q_ref(V - delta) against the median of the battery's first 3 clean cycles:
     a = "scale" (LAM-like indicator), delta = "shift" in mV (LLI-like indicator), and the fit residual.

Honest limit: a rigorous split into LLI, LAM of the positive and LAM of the negative electrode needs half-cell
reference curves of both electrodes, which the NASA data do not include. a and delta are physically motivated
INDICATORS, used as independent evidence the physics-informed model must agree with, not exact quantities.
The voltage correction removes the ohmic drop only. Remaining polarisation (resistance growth) shifts the charge
curve UP and the discharge curve DOWN, while lost lithium shifts both the SAME way. So per cycle:
    LLI-like shift     = (charge shift + discharge shift) / 2     (polarisation cancels)
    polarisation shift = (charge shift - discharge shift) / 2     (resistance growth)
Checked on synthetic cells with a known pure-capacity fade and growing resistance: discharge scale a matched SOH
within 0.01, LLI-like shift 0-9 mV, polarisation 54-100 mV.

Usage (next to twin_engine.py, e.g. in Colab):
    python scripts/dqdv_features.py --master battery_master_data.parquet
    python scripts/dqdv_features.py --master ... --cells B0005 B0029 B0045 --plots dqdv_plots
Outputs: dqdv_features.csv (one row per cell, cycle and mode), dqdv_modes.csv (per cell and cycle: LAM-like scale,
LLI-like shift, polarisation), dqdv_summary.csv (change over life per cell) and one PNG per battery. Send the three
CSVs and a few PNGs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twin_engine as te  # noqa: E402   (the analysis itself lives in the engine, section 45, shared with the app)


def plot_cell(f: pd.DataFrame, curves: dict, cell: str, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for j, mode in enumerate(("discharge", "charge")):
        cs = sorted(curves.get(mode, []), key=lambda x: x[0])
        if cs:
            pick = cs[:: max(1, len(cs) // 12)]
            cmap = plt.get_cmap("viridis")
            for k, (n, g, y) in enumerate(pick):
                ax[0, j].plot(g, y, color=cmap(k / max(len(pick) - 1, 1)), lw=1.2,
                              label=f"cycle {n}" if k in (0, len(pick) - 1) else None)
            ax[0, j].legend(fontsize=8)
        ax[0, j].set(title=f"{cell} · dQ/dV, {mode} (IR-corrected)", xlabel="Voltage (V)", ylabel="dQ/dV (Ah/V)")
    md = te.dqdv_modes(f).sort_values("n")
    ax[1, 0].plot(md["n"], md["dq_scale"], color="#AA4499", label="scale a (LAM-like)")
    ax[1, 0].plot(md["n"], md["SOH"], color="#888888", ls="--", label="SOH")
    ax[1, 0].set(title="Curve shrinkage", xlabel="Cycle", ylabel="scale a / SOH")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(md["n"], md["dq_lli_mV"], color="#0072B2", label="LLI-like shift (mV)")
    ax[1, 1].plot(md["n"], md["dq_pol_mV"], color="#882255", ls="--", label="polarisation (mV)")
    ax[1, 1].axhline(0, color="#999999", lw=0.8)
    ax[1, 1].set(title="Charge + discharge combined", xlabel="Cycle", ylabel="mV")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"dqdv_{cell}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--cells", nargs="*", default=None, help="default: every battery")
    ap.add_argument("--plots", default="dqdv_plots")
    ap.add_argument("--out", default="dqdv_features.csv")
    a = ap.parse_args()
    store = te.ParquetStore(a.master)
    ct, _ = te.build_cycle_table_with_report(store)
    feats, modes = [], []
    for c in (a.cells or sorted(ct["Cell_ID"].unique())):
        try:
            f, curves = te.dqdv_cell(store, ct, c)
        except Exception as exc:                            # one battery never stops the others
            print(f"{c}: failed ({type(exc).__name__}: {exc})")
            continue
        if f.empty:
            print(f"{c}: no usable constant-current curves")
            continue
        plot_cell(f, curves, c, Path(a.plots))
        md = te.dqdv_modes(f).sort_values("n")
        feats.append(f)
        modes.append(md)
        e = md.tail(3).median(numeric_only=True)
        print(f"{c}: {len(f)} curves · end: SOH {e['SOH']:.3f}, scale {e['dq_scale']:.3f}, LLI-like "
              f"{e['dq_lli_mV']:+.0f} mV, polarisation {e['dq_pol_mV']:+.0f} mV", flush=True)
    if not feats:
        return
    pd.concat(feats, ignore_index=True).assign(engine=te.ENGINE_VERSION).to_csv(a.out, index=False)
    allm = pd.concat(modes, ignore_index=True)
    allm.to_csv("dqdv_modes.csv", index=False)
    end = allm.sort_values("n").groupby("Cell_ID").tail(3).groupby("Cell_ID")[
        ["SOH", "dq_scale", "dq_lli_mV", "dq_pol_mV"]].median()
    end.to_csv("dqdv_summary.csv")
    print("\nEnd of life per battery (median of the last 3 cycles)\n" + end.round(3).to_string())
    print(f"\nSaved {a.out}, dqdv_modes.csv, dqdv_summary.csv and plots in {a.plots}/.")


if __name__ == "__main__":
    main()
