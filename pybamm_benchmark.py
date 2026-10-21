"""Which degradation modes can dQ/dV identify at NASA's currents and temperatures? A simulator benchmark.

Simulated LiCoO2/graphite cells (PyBaMM, SPMe, Ramadass2004 parameters) with KNOWN amounts of the three classic
degradation modes:
    LLI      loss of lithium inventory (SEI growth and plating both consume lithium)
    LAM_NE   loss of active material in the graphite electrode (its lithium is kept)
    LAM_PE   loss of active material in the LiCoO2 electrode (its lithium is kept)
Every cell is first emptied slowly and rested, then follows the NASA protocol: constant-current charge at 0.75C to 4.2 V, constant voltage until C/100, 30 min
rest, discharge at 0.5C, 1C or 2C (NASA's 1, 2 and 4 A on 2 Ah cells) to 2.7 V, at 4, 24 or 43 C. On each cell the
SAME dQ/dV features as in the app (twin_engine section 45) are computed, against the fresh cell of the same current
and temperature. Then, per condition, each mode is predicted from the features (cross-validated linear model) and
from the capacity alone: the cross-validated R^2 says whether the features can identify that mode at that rate and
temperature (R^2 near 1) or not (near 0), and how much they add over the capacity.

Assumption, stated: the Ramadass2004 set has no temperature dependence. Arrhenius factors are added with typical
literature activation energies (solid diffusion 30 kJ/mol, reaction kinetics 35 kJ/mol, electrolyte transport
17 kJ/mol); the cold results depend on these values.
Limit: LLI covers SEI and plating together; the split between them comes from operating conditions, not from dQ/dV.

STATUS (2026-10-21): WORK IN PROGRESS, results not yet trustworthy.
  - Ramadass2004's LiCoO2 voltage table limits lithium loss to ~6 % (10 % leaves the table: nonsense voltages);
    levels used: LLI 0-6 %, LAM 0-15 %.
  - 4 C is far too pessimistic (fresh cell at 1C: 0.12 Ah): the activation energies must be calibrated so a fresh
    cold cell keeps the capacity fraction of the fresh NASA cold cells (no self-heating in SPMe).
  - 15 % LiCoO2 LAM raises capacity slightly (1.876 vs 1.837 Ah): model LAM of delithiated AND lithiated material
    separately and check the electrode balance before using LAM_PE results.

Usage (next to twin_engine.py, e.g. in Colab):  pip install pybamm   then
    python scripts/pybamm_benchmark.py                 # full design: 9 conditions x 37 cells (~15-30 min)
    python scripts/pybamm_benchmark.py --quick         # 2 conditions x 10 cells, a check (~1-2 min)
Outputs (resumable): pybamm_benchmark_runs.csv, pybamm_identifiability.csv, pybamm_dqdv_modes.png. Send all three.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twin_engine as te  # noqa: E402

warnings.filterwarnings("ignore")
R = 8.314462618
T_REF = 298.15
EA = {"diff": 30e3, "kin": 35e3, "elyte": 17e3}             # J/mol, typical literature values (assumption)
MODES = ("LLI", "LAM_NE", "LAM_PE")
RATES = (0.5, 1.0, 2.0)                                     # C-rates = NASA 1, 2, 4 A on 2 Ah
TEMPS = (4.0, 24.0, 43.0)


def _arrh(f, ea):
    """Wrap a PyBaMM parameter function so it gains an Arrhenius factor in its last argument (temperature)."""
    import pybamm

    def g(*args):
        T = args[-1]
        return f(*args) * pybamm.exp(ea / R * (1 / T_REF - 1 / T))
    return g


def base_params():
    import pybamm
    p = pybamm.ParameterValues("Ramadass2004")
    for key, kind in (("Negative particle diffusivity [m2.s-1]", "diff"), ("Positive particle diffusivity [m2.s-1]", "diff"),
                      ("Negative electrode exchange-current density [A.m-2]", "kin"),
                      ("Positive electrode exchange-current density [A.m-2]", "kin"),
                      ("Electrolyte conductivity [S.m-1]", "elyte"), ("Electrolyte diffusivity [m2.s-1]", "elyte")):
        if key in p.keys() and callable(p[key]):
            p.update({key: _arrh(p[key], EA[kind])})
    p.update({"Lower voltage cut-off [V]": 2.7, "Upper voltage cut-off [V]": 4.2, "Nominal cell capacity [A.h]": 1.5})
    return p


def cell_params(base, lli: float, lam_ne: float, lam_pe: float, T_c: float):
    """Impose the modes on the electrodes' sizes (LAM) and the cell's lithium (LLI). Lithium is kept in the fresh
    split between the electrodes; PyBaMM's electrode balancing (initial_soc in simulate) then sets physically
    consistent states. LAM here keeps the lost material's lithium (LAM of delithiated material)."""
    p = base.copy()
    Ln, Lp = p["Negative electrode thickness [m]"], p["Positive electrode thickness [m]"]
    en, ep = p["Negative electrode active material volume fraction"], p["Positive electrode active material volume fraction"]
    cn, cp = p["Initial concentration in negative electrode [mol.m-3]"], p["Initial concentration in positive electrode [mol.m-3]"]
    tot = cn * en * Ln + cp * ep * Lp
    li, fn = (1 - lli) * tot, cn * en * Ln / tot
    en2, ep2 = en * (1 - lam_ne), ep * (1 - lam_pe)
    p.update({"Negative electrode active material volume fraction": en2,
              "Positive electrode active material volume fraction": ep2,
              "Initial concentration in negative electrode [mol.m-3]": li * fn / (en2 * Ln),
              "Initial concentration in positive electrode [mol.m-3]": li * (1 - fn) / (ep2 * Lp),
              "Ambient temperature [K]": T_c + 273.15, "Initial temperature [K]": T_c + 273.15})
    return p


def simulate(p, rate: float):
    """Empty and rested (electrode balancing at 0 % state of charge), then the NASA protocol. Returns (charge CC
    segment, discharge) as (t, V, I) arrays, or None."""
    import pybamm
    exp = pybamm.Experiment(["Rest for 30 minutes", "Charge at 0.75C for 4 hours or until 4.2 V",
                             "Hold at 4.2 V until C/100", "Rest for 30 minutes",
                             f"Discharge at {rate}C for 3 hours or until 2.7 V"], period="5 seconds")
    sol = pybamm.Simulation(pybamm.lithium_ion.SPMe(), parameter_values=p, experiment=exp).solve(initial_soc=0.0)
    st = [step for cyc in sol.cycles if cyc is not None for step in cyc.steps]
    if len(st) < 5:
        return None
    seg = lambda s: (s["Time [s]"].entries, s["Terminal voltage [V]"].entries, np.abs(s["Current [A]"].entries))
    return seg(st[1]), seg(st[4])


def width_fwhm(grid, y) -> float:
    k = int(np.argmax(y))
    half = y[k] / 2
    left = np.where(y[:k] < half)[0]
    right = np.where(y[k:] < half)[0]
    if not len(left) or not len(right):
        return np.nan
    return float(grid[k + right[0]] - grid[left[-1]])


def features(ch, dis, ref):
    """Same dQ/dV pipeline as the app (no ohmic correction: the charge/discharge combination removes polarisation)."""
    out = {}
    curves = {}
    for name, (t, v, i), charge in (("charge", ch, True), ("discharge", dis, False)):
        r = te._dq_qv(t, v, i, np.nan, charge)
        if r is None:
            return None, None
        grid, q, y = r
        curves[name] = (grid, q, y)
        pk = te._dq_peaks(grid, y)
        out[f"{name} capacity (Ah)"] = float(q.max() - q.min())
        out[f"{name} main peak V"] = float(grid[int(np.argmax(y))])
        out[f"{name} main peak height"] = float(y.max())
        out[f"{name} main peak width (V)"] = width_fwhm(grid, y)
        out[f"{name} peak count"] = sum(1 for k in pk if k.endswith("_V"))
        if ref is not None:
            g0, q0, _ = ref[name]
            out.update({f"{name} {k}": v_ for k, v_ in te._dq_shift_scale(grid, q, g0, q0).items()})
    if ref is not None:
        out["LLI-like shift (mV)"] = (out["charge shift_mV"] + out["discharge shift_mV"]) / 2
        out["polarisation (mV)"] = (out["charge shift_mV"] - out["discharge shift_mV"]) / 2
    return out, curves


def design(quick: bool) -> pd.DataFrame:
    rows = [(0.0, 0.0, 0.0)]
    levels = {0: (0.04,), 1: (0.10,), 2: (0.10,)} if quick else {0: (0.02, 0.04, 0.06), 1: (0.05, 0.10, 0.15),
                                                                  2: (0.05, 0.10, 0.15)}
    for m in range(3):
        for lv in levels[m]:
            r = [0.0, 0.0, 0.0]
            r[m] = lv
            rows.append(tuple(r))
    rng = np.random.default_rng(7)
    rows += [(round(rng.uniform(0, 0.06), 3), round(rng.uniform(0, 0.15), 3), round(rng.uniform(0, 0.15), 3))
             for _ in range(6 if quick else 24)]
    return pd.DataFrame(rows, columns=list(MODES))


FEATS = ["discharge capacity (Ah)", "discharge main peak V", "discharge main peak height", "discharge main peak width (V)",
         "charge main peak V", "charge main peak height", "charge main peak width (V)", "discharge scale_a",
         "LLI-like shift (mV)", "polarisation (mV)"]


def identifiability(runs: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for (rate, T), d in runs[runs["status"] == "ok"].groupby(["C-rate", "T (C)"]):
        if len(d) < 8:
            continue
        cv = KFold(min(5, len(d)), shuffle=True, random_state=0)
        X = d[[f for f in FEATS if f in d]].fillna(d[[f for f in FEATS if f in d]].median())
        Xc = d[["discharge capacity (Ah)"]]
        for m in MODES:
            y = d[m].to_numpy()
            if np.ptp(y) == 0:
                continue
            model = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 2, 13)))
            r2 = lambda pred: 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            rows.append({"C-rate": rate, "T (C)": T, "mode": m, "cells": len(d),
                         "R2 from dQ/dV features": r2(cross_val_predict(model(), X, y, cv=cv)),
                         "R2 from capacity only": r2(cross_val_predict(model(), Xc, y, cv=cv))})
    return pd.DataFrame(rows)


def plot_modes(curves_by: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    col = {"fresh": "#222222", "LLI": "#0072B2", "LAM_NE": "#56B4E9", "LAM_PE": "#AA4499"}
    for lab, cv in curves_by.items():
        for j, name in enumerate(("discharge", "charge")):
            g, _, y = cv[name]
            ax[j].plot(g, y, color=col.get(lab, "#999999"), lw=1.8, label=lab)
    for j, name in enumerate(("discharge", "charge")):
        ax[j].set(title=f"Simulated LiCoO2/graphite, 1C, 24 C · {name}", xlabel="Voltage (V)", ylabel="dQ/dV (Ah/V)")
        ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="pybamm_benchmark_runs.csv")
    a = ap.parse_args()
    base = base_params()
    cells = design(a.quick)
    conds = [(1.0, 24.0), (1.0, 4.0)] if a.quick else [(r, T) for T in TEMPS for r in RATES]
    done = pd.read_csv(a.out) if Path(a.out).exists() else pd.DataFrame()
    key = lambda r, T, c: (round(r, 3), round(T, 1), round(c.LLI, 3), round(c.LAM_NE, 3), round(c.LAM_PE, 3))
    seen = {key(x["C-rate"], x["T (C)"], x) for _, x in done.iterrows()} if len(done) else set()
    plot_curves = {}
    for rate, T in conds:
        ref = None
        for _, c in cells.iterrows():
            is_fresh = c.sum() == 0
            k = key(rate, T, c)
            if k in seen and not is_fresh:
                continue
            row = {"C-rate": rate, "T (C)": T, **c.to_dict()}
            try:
                res = simulate(cell_params(base, c.LLI, c.LAM_NE, c.LAM_PE, T), rate)
                f, curves = features(*res, ref) if res is not None else (None, None)
            except Exception as exc:
                f, curves, row["status"] = None, None, f"failed: {type(exc).__name__}: {str(exc)[:80]}"
            if is_fresh:
                ref = curves
                if f is None:
                    print(f"{rate}C {T} C: fresh cell failed, condition skipped")
                    break
                f, _ = features(*res, ref)
            if f is not None:
                row.update({"status": "ok", **f})
            row.setdefault("status", "failed: no curve")
            if (rate, T) == (1.0, 24.0) and curves is not None:
                lab = "fresh" if is_fresh else next((m for m in MODES if c[m] > 0 and c[m] == c.sum()), None)
                if lab:
                    plot_curves[lab] = curves
            if k not in seen:
                pd.DataFrame([row]).to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
                seen.add(k)
        print(f"{rate}C at {T:.0f} C done", flush=True)
    runs = pd.read_csv(a.out)
    if plot_curves:
        plot_modes(plot_curves, "pybamm_dqdv_modes.png")
    idf = identifiability(runs)
    idf.to_csv("pybamm_identifiability.csv", index=False)
    if len(idf):
        print("\nCan each mode be recovered? Cross-validated R^2 (1 = fully, 0 = not at all)")
        print(idf.pivot_table(index=["C-rate", "T (C)"], columns="mode",
                              values=["R2 from dQ/dV features", "R2 from capacity only"]).round(2).to_string())
    print(f"\nSaved {a.out}, pybamm_identifiability.csv" + (", pybamm_dqdv_modes.png" if plot_curves else ""))


if __name__ == "__main__":
    main()
