"""
app.py - Battery Digital Twin & Operando Diagnostics Platform (Streamlit front-end, v4)
=====================================================================================

UI layer only; all computation lives in ``twin_engine.py``.

Design rules enforced in this file
  * Theme resiliency - custom CSS never hard-codes a text colour. Body text inherits the
    active Streamlit theme; surfaces and borders are translucent neutrals. Plotly figures
    get an explicit light, dark or publication palette resolved from the active theme
    (``st.context.theme``, Streamlit >= 1.46) with a manual override in the sidebar.
  * Colour - Okabe-Ito colour-blind-safe palette; every paradigm is double-encoded with a
    line dash / marker symbol so figures survive greyscale printing.
  * Figures - one ``style_fig``: title in a reserved top band, legend centred in a reserved
    bottom band below the x-axis title (bordered, theme-matched), unified cross-hair hover.
    Forecasts always show their uncertainty band. Every key figure has HTML / SVG / CSV export.
  * Layout - global cell selector at the top; three views (data & diagnostics, models &
    forecasting, operations & control). Only the active view is executed (lazy navigation),
    and interactive sub-sections run as fragments so their widgets do not rerun the page.
  * Provenance - every result can be downloaded with a JSON run manifest.
"""

from __future__ import annotations

import html
import itertools
import json
import math
import io
import time
import os
import re
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import twin_engine as te

# ---- engine / app version handshake -------------------------------------------------------------
# Streamlit can keep an old copy of twin_engine in memory after a redeploy (it reruns app.py but does
# not always re-import changed modules), and app.py and twin_engine.py must come from the same release.
REQUIRED_ENGINE = "5.45.1"
if str(getattr(te, "ENGINE_VERSION", "0")) != REQUIRED_ENGINE:
    import importlib
    te = importlib.reload(te)

st.set_page_config(
    page_title="Battery Digital Twin · Operando Diagnostics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

_REQUIRED_FUNCS = ("fleet_shape_forecast", "fleet_shape", "shape_drop", "run_protocol", "protocol_scoreboard",
                   "train_ml_v2", "baseline_forecast_v2", "train_trajectory_model", "train_deep_trajectory",
                   "tune_level_model", "run_protocol_task", "train_pinn_model", "training_influence",
                   "ensemble_trajectories", "safe_origin", "reference_condition_ct", "dqdv_cell", "dqdv_modes", "dqdv_indicators", "attach_dqdv", "pinn_explain", "recovery_cohort", "recovery_physics_fit", "recovery_pinn_table", "nasa_physics_fit", "nasa_physics_table", "nasa_physics_trajectory", "nasa_hybrid_fit", "hybrid_table", "hybrid_trajectory", "hybrid_validation", "physics_forecast", "nasa_anomaly_scan", "anomaly_summary", "anomaly_validation", "nasa_anomaly_scan2", "anomaly_summary2", "anomaly_validation2", "anomaly_common_cause", "anomaly_events", "anomaly_cycle_status", "eol_cycle_from_replay", "pinn_state_raw", "hybrid_portable", "integrated_snapshot", "integrated_timeline", "whatif_identifiability", "whatif_tested", "whatif_projection", "forecast_optimism", "conservative_eol", "m3_depth_table", "m3_operation_study", "m3_maintenance_study", "m3_lifecycle_study", "frozen_live_window")
_missing = [f for f in _REQUIRED_FUNCS if not hasattr(te, f)]
if str(getattr(te, "ENGINE_VERSION", "0")) != REQUIRED_ENGINE or _missing:
    _here = Path(__file__).resolve().parent
    _copies = sorted(p.name for p in _here.glob("twin_engine*.py")) + sorted(p.name for p in _here.glob("app*.py"))
    st.error(f"**Version mismatch.** This app.py needs twin_engine.py version **{REQUIRED_ENGINE}**, but the engine it "
             f"loaded is version **{getattr(te, 'ENGINE_VERSION', 'unknown')}** "
             f"(file: `{Path(getattr(te, '__file__', '?')).name}`)"
             + (f", missing: {', '.join(_missing)}" if _missing else "") + ".\n\n"
             f"Python files found next to the app: {', '.join(f'`{c}`' for c in _copies) or 'none'}. "
             "If you see names like `twin_engine (1).py`, the browser renamed a second download: rename it to exactly "
             "`twin_engine.py` (replacing the old one) and delete the copies. Then reboot the app (Manage app → ⋮ → Reboot).")
    st.stop()

# =============================================================================
# Constants & version gates
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"
MASTER_NAME, IMP_NAME = "battery_master_data.parquet", "impedance_ground_truth.parquet"
POLICIES = ["Twin-Aware", "Fixed 1 A", "Fixed 2 A", "Fixed 4 A"]
V_HOME, V_DATA = ":material/home: Home", ":material/monitoring: 1 · Data & diagnostics"
V_TWIN = ":material/hub: 2 · GA-PINN twin"
V_MODELS = ":material/psychology: 3 · Model comparison"
V_OPS = ":material/tune: 4 · Mission 3 · Operation & maintenance"
V_STUDY = ":material/fact_check: 5 · Study results"
V_LIVE, V_UPDATE = ":material/play_circle: Appendix · EKF live twin", ":material/update: Appendix · Model updating"
# v5.46 (meeting build): one story, data -> diagnostics -> the GA-PINN twin -> model comparison -> Mission 3 ->
# results; the EKF twin and the update studies are kept, in an appendix
VIEWS = [V_HOME, V_DATA, V_TWIN, V_MODELS, V_OPS, V_STUDY, V_LIVE, V_UPDATE]
SANS = "IBM Plex Sans, Segoe UI, Helvetica Neue, Arial, sans-serif"
SERIF = "'STIX Two Text', 'Times New Roman', Times, serif"


def _version_tuple(v: str) -> Tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", v)[:2]]
    nums += [0] * (2 - len(nums))
    return nums[0], nums[1]


ST_VERSION = _version_tuple(st.__version__)
PLOTLY_VERSION = _version_tuple(plotly.__version__)
LEGEND_CONTAINER_REF = PLOTLY_VERSION >= (5, 15)
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# st.fragment (>= 1.37): widgets inside a fragment rerun only that fragment.
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


# =============================================================================
# Theme-aware, colour-blind-safe palettes (Okabe & Ito, 2008)
# =============================================================================
@dataclass(frozen=True)
class Palette:
    mode: str
    template: str
    font: str
    text: str
    muted: str
    grid: str
    axis: str
    paper_bg: str
    plot_bg: str
    hover_bg: str
    legend_bg: str
    legend_border: str
    accent: str
    measured: str
    cohort: str
    ekf: str
    pinn: str
    eis: str
    eol: str
    r_int: str
    r_ct: str
    currents: Tuple[Tuple[float, str], ...]
    series: Tuple[str, ...]
    ica_range: Tuple[float, float]
    semi: str = "#332288"
    pf: str = "#F0E442"
    mode_colors: Tuple[str, ...] = ("#56B4E9", "#CC79A7", "#332288", "#AA4499", "#6699CC")

    def current_color(self, amps: float) -> str:
        return dict(self.currents).get(float(amps), self.muted)


# Double encoding: each paradigm / series also gets its own dash and marker.
DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "star")
# 24 distinguishable, colour-blind-considerate hues (Okabe-Ito + Paul Tol bright/muted/vibrant)
CELL_COLORS = ("#0072B2", "#CC79A7", "#56B4E9", "#882255", "#332288", "#AA4499", "#6699CC", "#7F3C8D",
               "#3969AC", "#B07AA1", "#4E79A7", "#9C755F", "#76B7B2", "#BAB0AC", "#5D69B1", "#A5AA99",
               "#8E6C8A", "#2F8AC4", "#99C945", "#764E9F", "#ED645A", "#CC61B0", "#24796C", "#DAA51B")[:20]
# risk colours (green / yellow / orange / red) are reserved for risk; categories use the hues above
CELL_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "triangle-down", "star", "hexagon", "pentagon",
                "cross", "x", "star-triangle-up", "hourglass")


def cell_style(cell_id: str, all_cells: Sequence[str]) -> Tuple[str, str]:
    """Stable colour + marker per battery across every chart (sorted cohort order)."""
    order = sorted(all_cells)
    i = order.index(cell_id) if cell_id in order else hash(cell_id)
    return CELL_COLORS[i % len(CELL_COLORS)], CELL_SYMBOLS[(i // len(CELL_COLORS) + i) % len(CELL_SYMBOLS)]


def cell_label(cell_id: str, meta: pd.DataFrame) -> str:
    if cell_id not in meta.index:
        return cell_id
    m = meta.loc[cell_id]
    return f"{cell_id} · {m['Ambient_C']:.0f} °C · {m['I_dis_A']:.1f} A · {m['V_cut_V']:.1f} V"
CURRENT_SYMBOL = {1.0: "circle", 2.0: "square", 4.0: "diamond"}
PARADIGM_DASH = {"twin": "dash", "pinn": "solid", "ml": "dashdot"}

DARK = Palette(
    mode="dark", template="plotly_dark", font=SANS,
    text="#f1f5f9", muted="#cbd5e1", grid="rgba(148,163,184,0.16)", axis="#64748b",
    paper_bg="rgba(0,0,0,0)", plot_bg="rgba(148,163,184,0.05)", hover_bg="#0b1220",
    legend_bg="rgba(11,18,32,0.90)", legend_border="#64748b",
    accent="#56B4E9", measured="#f8fafc", cohort="rgba(148,163,184,0.38)",
    ekf="#56B4E9", pinn="#CC79A7", eis="#E69F00", eol="#D55E00",
    r_int="#CC79A7", r_ct="#56B4E9",
    currents=((1.0, "#56B4E9"), (2.0, "#CC79A7"), (4.0, "#AA4499")),
    series=("#E69F00", "#009E73", "#F0E442", "#D55E00", "#CC79A7"),
    ica_range=(0.15, 0.95),
)

LIGHT = Palette(
    mode="light", template="plotly_white", font=SANS,
    text="#0f172a", muted="#334155", grid="rgba(15,23,42,0.09)", axis="#94a3b8",
    paper_bg="rgba(0,0,0,0)", plot_bg="rgba(15,23,42,0.02)", hover_bg="#ffffff",
    legend_bg="rgba(255,255,255,0.95)", legend_border="#94a3b8",
    accent="#0072B2", measured="#111827", cohort="rgba(100,116,139,0.35)",
    ekf="#0072B2", pinn="#CC79A7", eis="#E69F00", eol="#D55E00",
    r_int="#AA4499", r_ct="#0072B2",
    currents=((1.0, "#0072B2"), (2.0, "#AA4499"), (4.0, "#882255")),
    # yellow (#F0E442) has too little contrast on white; use black-ish grey instead.
    series=("#E69F00", "#009E73", "#D55E00", "#56B4E9", "#555555"),
    ica_range=(0.0, 0.82),
    pf="#555555", mode_colors=("#0072B2", "#CC79A7", "#332288", "#AA4499", "#6699CC"),
)

# Publication style: opaque white background, serif type, black axes (journal figures).
PUBLICATION = replace(
    LIGHT, mode="publication", font=SERIF, text="#000000", muted="#222222",
    grid="rgba(0,0,0,0.08)", axis="#000000", paper_bg="#ffffff", plot_bg="#ffffff",
    legend_bg="#ffffff", legend_border="#000000", cohort="rgba(0,0,0,0.25)")


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def detect_base_theme() -> str:
    """Active Streamlit theme: st.context.theme (>= 1.46), then config, then light."""
    try:
        t = getattr(getattr(st.context, "theme", None), "type", None)
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


def resolve_palette(choice: str, publication: bool) -> Palette:
    if publication:
        return PUBLICATION
    mode = choice.lower() if choice in ("Light", "Dark") else detect_base_theme()
    return DARK if mode == "dark" else LIGHT


# Design layer on top of the original colours: one typeface with tabular figures, sentence-case labels, visible
# keyboard focus and reduced motion when requested (the original gradients and palette are kept).
DESIGN_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, input, textarea, select, button, h1, h2, h3, h4, h5,
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }
div[data-testid="stMetricValue"], div[data-testid="stDataFrame"], div[data-testid="stTable"], .bt-statusbar {
  font-variant-numeric: tabular-nums; }
.bt-hero .bt-kicker { text-transform: none !important; letter-spacing: 0 !important; font-weight: 500 !important;
  font-size: 0.92rem !important; }
.bt-hero .bt-title { font-weight: 650 !important; letter-spacing: -0.02em !important; }
.stButton > button:focus-visible { outline: 2px solid var(--bt-accent) !important; outline-offset: 2px; }
.bt-statusbar { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important; letter-spacing: 0 !important;
  font-size: 0.86rem !important; }
@media (prefers-reduced-motion: reduce) { .bt-hero, .bt-dot, .bt-pkt, .bt-scan { animation: none !important; } }
/* ---------- application look: full-width workspace, title bar, tab strip, panels, denser controls ---------- */
html { font-size: 15px; }
.block-container { max-width: 100% !important; padding: 0.4rem 1.4rem 2rem 1.4rem !important; }
div[data-testid="stDecoration"], footer, #MainMenu, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {
  display: none !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: flex !important; visibility: visible !important; opacity: 1 !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 2.4rem !important; }
.bt-appbar { display: flex; align-items: center; gap: 14px; padding: 9px 16px; margin: 0;
  background: #0E1726; color: #E6EEF7; border: 1px solid #1F2E45; border-bottom: 0; border-radius: 8px 8px 0 0;
  position: relative; }
.bt-appbar::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px; background: var(--bt-grad); }
.bt-appicon { display: inline-flex; width: 28px; height: 28px; border-radius: 6px; background: var(--bt-grad);
  align-items: center; justify-content: center; flex: none; }
.bt-appname { font-weight: 650; font-size: 1.02rem; letter-spacing: -0.01em; }
.bt-appmeta { color: #8EA3BA; font-size: 0.86rem; }
.bt-appright { margin-left: auto; color: #8EA3BA; font-size: 0.82rem; }
.bt-statusbar { margin: 10px 0 10px 0 !important; padding: 6px 16px !important; }
div[data-testid="stButtonGroup"] { border-bottom: 1px solid var(--bt-border); gap: 0 !important; margin-bottom: 6px; }
div[data-testid="stButtonGroup"] button { border-radius: 6px 6px 0 0 !important; border: 1px solid transparent !important;
  border-bottom: 0 !important; background: transparent !important; padding: 6px 16px !important; font-weight: 500 !important;
  margin-bottom: -1px; }
div[data-testid="stButtonGroup"] button[kind$="Active"], div[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--bt-surface) !important; border-color: var(--bt-border) !important; font-weight: 650 !important;
  box-shadow: inset 0 3px 0 0 #0072B2 !important; }
.bt-section { font-size: 1.02rem !important; font-weight: 650 !important; margin: 1.4rem 0 0.6rem 0 !important;
  padding: 7px 12px !important; border: 1px solid var(--bt-border) !important; border-radius: 6px !important;
  background: var(--bt-surface); animation: none !important; }
.bt-section .bt-num { width: 24px !important; height: 24px !important; border-radius: 5px !important; font-size: 0.78rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 6px !important; }
div[data-testid="stExpander"] details { border-radius: 6px !important; }
div[data-testid="stWidgetLabel"] p { font-size: 0.84rem !important; font-weight: 500 !important; }
.stButton > button, .stDownloadButton > button { border-radius: 6px !important; padding: 0.3rem 0.9rem !important; }
.bt-context { border-radius: 6px !important; }
</style>"""


def inject_css(P: Palette) -> None:
    """Text colours are inherited from the Streamlit theme; only accents use the palette."""
    css = f"""<style>
:root {{
  --bt-accent: {P.accent};
  --bt-accent-soft: {rgba(P.accent, 0.12)};
  --bt-accent-line: {rgba(P.accent, 0.45)};
  --bt-surface: rgba(128,128,128,0.07);
  --bt-border: rgba(128,128,128,0.28);
  --bt-grad: linear-gradient(120deg, #0B3D91 0%, #0072B2 38%, #009E73 72%, #56B4E9 100%);
  --bt-grad-warm: linear-gradient(90deg, #0072B2, #009E73, #E69F00);
  --bt-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 6px 18px rgba(0,0,0,0.06);
}}
.block-container {{padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1500px;}}
/* ---------- hero: fixed dark gradient, so white text is safe in both themes ---------- */
.bt-hero {{
  position: relative; overflow: hidden; border-radius: 18px; padding: 28px 34px 22px 34px; margin-bottom: 1.1rem;
  background: var(--bt-grad); box-shadow: 0 10px 30px rgba(0,60,120,0.25);
}}
.bt-hero::before {{
  content: ""; position: absolute; inset: 0; opacity: 0.18; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,0.35) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.35) 1px, transparent 1px);
  background-size: 28px 28px; mask-image: linear-gradient(90deg, transparent 0%, black 60%);
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 60%);
}}
.bt-hero::after {{
  content: ""; position: absolute; right: -60px; top: -80px; width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.28), rgba(255,255,255,0) 70%); pointer-events: none;
}}
.bt-hero .bt-title {{position: relative; font-size: 1.9rem; font-weight: 850; letter-spacing: -0.025em;
  line-height: 1.2; color: #FFFFFF;}}
.bt-hero .bt-sub {{position: relative; margin-top: 8px; font-size: 1.0rem; line-height: 1.55; color: rgba(255,255,255,0.92);
  max-width: 92ch;}}
.bt-hero .bt-kicker {{position: relative; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: rgba(255,255,255,0.80); margin-bottom: 6px;}}
.bt-pill {{
  position: relative; display: inline-block; padding: 4px 12px; margin: 12px 8px 0 0; border-radius: 999px;
  font-size: 0.8rem; font-weight: 650; color: #FFFFFF; background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.35); backdrop-filter: blur(6px);
}}
.bt-chip {{
  display: inline-block; padding: 3px 11px; margin: 4px 6px 0 0; border-radius: 999px;
  font-size: 0.82rem; background: var(--bt-surface); border: 1px solid var(--bt-border); color: inherit;
}}
.bt-status {{font-size: 0.95rem; line-height: 1.6; color: inherit;}}
/* ---------- numbered section headers ---------- */
.bt-section {{
  display: flex; align-items: center; gap: 12px; font-size: 1.22rem; font-weight: 800;
  margin: 2.2rem 0 0.8rem 0; color: inherit; letter-spacing: -0.01em;
  padding-bottom: 8px; border-bottom: 1px solid var(--bt-border);
}}
.bt-section .bt-num {{
  flex: none; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
  border-radius: 9px; background: var(--bt-grad); color: #FFFFFF; font-size: 0.9rem; font-weight: 800;
  box-shadow: 0 3px 10px rgba(0,114,178,0.35);
}}
/* ---------- cards & KPI tiles ---------- */
.bt-card {{
  background: var(--bt-surface); border: 1px solid var(--bt-border); border-left: 4px solid var(--bt-accent);
  border-radius: 12px; padding: 16px 20px; margin: 10px 0 16px 0; box-shadow: var(--bt-shadow);
}}
.bt-card h4 {{margin: 0 0 8px 0; font-size: 0.98rem; font-weight: 800; color: var(--bt-accent);}}
.bt-card li {{font-size: 0.93rem; line-height: 1.55; margin: 3px 0; color: inherit;}}
div[data-testid="stMetric"] {{
  position: relative; overflow: hidden; background: var(--bt-surface); border: 1px solid var(--bt-border);
  border-radius: 14px; padding: 16px 18px 14px 18px; box-shadow: var(--bt-shadow);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div[data-testid="stMetric"]::before {{
  content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px; background: var(--bt-grad-warm);
}}
div[data-testid="stMetric"]:hover {{transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,0.10);}}
div[data-testid="stMetricLabel"] p {{font-size: 0.8rem; font-weight: 700; opacity: 0.75; letter-spacing: 0.01em;}}
div[data-testid="stMetricValue"] {{font-weight: 800; letter-spacing: -0.01em;}}
/* ---------- plots, expanders, tabs, buttons ---------- */
div[data-testid="stPlotlyChart"] {{
  border: 1px solid var(--bt-border); border-radius: 14px; padding: 6px 4px 2px 4px; background: var(--bt-surface);
  box-shadow: var(--bt-shadow); margin-bottom: 0.4rem;
}}
div[data-testid="stExpander"] details {{border-radius: 12px; border: 1px solid var(--bt-border);}}
div[data-testid="stExpander"] summary {{font-weight: 650;}}
button[data-baseweb="tab"] {{font-weight: 700; font-size: 0.98rem;}}
div[data-baseweb="tab-highlight"] {{background: var(--bt-grad-warm) !important; height: 3px !important;}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: var(--bt-grad); border: 0; color: #FFFFFF; font-weight: 700; border-radius: 10px;
  box-shadow: 0 4px 14px rgba(0,114,178,0.35);
}}
.stButton > button[kind="primary"]:hover {{filter: brightness(1.08); box-shadow: 0 6px 18px rgba(0,114,178,0.45);}}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {{border-radius: 12px; overflow: hidden;}}
section[data-testid="stSidebar"] {{border-right: 1px solid var(--bt-border);}}
/* ---------- explanations, recommendations, ladder, guide ---------- */
.bt-explain {{margin: -2px 0 14px 0; padding: 9px 14px; border-radius: 10px; font-size: 0.88rem; line-height: 1.5;
  background: var(--bt-accent-soft); border: 1px solid var(--bt-accent-line); color: inherit;}}
.bt-explain-tag {{font-weight: 800; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--bt-accent); margin-right: 6px;}}
.bt-reco {{margin: 4px 0 14px 0; padding: 12px 16px; border-radius: 14px; border: 1px solid var(--bt-border);
  background: linear-gradient(135deg, rgba(0,114,178,0.10), rgba(0,158,115,0.08)); box-shadow: var(--bt-shadow);
  animation: bt-rise .5s ease both;}}
.bt-reco-h {{font-weight: 800; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.8;
  margin-bottom: 6px;}}
.bt-reco-item {{display: flex; gap: 10px; align-items: baseline; font-size: 0.93rem; line-height: 1.5; margin: 3px 0;}}
.bt-reco-dot {{flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--bt-grad-warm);
  box-shadow: 0 0 0 3px rgba(0,158,115,0.15); transform: translateY(-1px);}}
.bt-ladder {{display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin: 6px 0 4px 0;}}
.bt-step {{position: relative; padding: 10px 10px 9px 10px; border-radius: 12px; border: 1px solid var(--bt-border);
  background: var(--bt-surface); transition: transform .15s ease, box-shadow .15s ease;}}
.bt-step:hover {{transform: translateY(-2px); box-shadow: var(--bt-shadow);}}
.bt-step-n {{display: inline-flex; width: 24px; height: 24px; border-radius: 7px; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.8rem; color: #fff; background: #8C8C8C;}}
.bt-step-done .bt-step-n {{background: var(--bt-grad);}}
.bt-step-done::after {{content: "✓"; position: absolute; right: 10px; top: 8px; color: #009E73; font-weight: 900;}}
.bt-step-t {{margin-top: 6px; font-size: 0.8rem; font-weight: 700; line-height: 1.25;}}
.bt-guide {{padding: 10px 12px; border-radius: 12px; border: 1px solid var(--bt-border); background: var(--bt-surface);
  margin-bottom: 10px;}}
.bt-guide-h {{font-weight: 800; font-size: 0.82rem; margin-bottom: 6px;}}
.bt-guide-bar {{height: 6px; border-radius: 99px; background: rgba(128,128,128,0.2); overflow: hidden; margin-bottom: 8px;}}
.bt-guide-bar > div {{height: 100%; background: var(--bt-grad-warm); transition: width .6s ease;}}
.bt-guide-row {{display: flex; gap: 8px; font-size: 0.82rem; opacity: 0.6; margin: 2px 0;}}
.bt-guide-row.ok {{opacity: 1; font-weight: 650;}}
.bt-guide-row.ok span {{color: #009E73;}}
@keyframes bt-rise {{from {{opacity: 0; transform: translateY(6px);}} to {{opacity: 1; transform: none;}}}}
div[data-testid="stPlotlyChart"], div[data-testid="stMetric"], .bt-card {{animation: bt-rise .45s ease both;}}
.bt-section {{animation: bt-rise .4s ease both;}}
@media (max-width: 900px) {{.bt-ladder {{grid-template-columns: repeat(3, 1fr);}}}}
@media (prefers-reduced-motion: reduce) {{.bt-reco, .bt-section, div[data-testid="stPlotlyChart"], div[data-testid="stMetric"] {{animation: none;}}}}
/* ---------- context bar, compact explanations, empty states, home ---------- */
.bt-context {{position: sticky; top: 2.9rem; z-index: 50; display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center;
  padding: 8px 14px; margin: 6px 0 12px 0; border-radius: 12px; border: 1px solid var(--bt-border);
  background: color-mix(in srgb, var(--bt-surface) 70%, transparent); backdrop-filter: blur(8px); font-size: 0.85rem;}}
.bt-context i {{font-style: normal; opacity: 0.6; margin-right: 4px; font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.06em;}}
.bt-context-hint {{margin-left: auto; font-size: 0.72rem; opacity: 0.55;}}
.bt-explain-mini {{display: inline-block; margin: -4px 0 10px 0; font-size: 0.75rem; opacity: 0.6; cursor: help;
  border-bottom: 1px dotted currentColor;}}
.bt-empty {{display: flex; gap: 10px; align-items: center; padding: 14px 16px; margin: 6px 0 14px 0; border-radius: 12px;
  border: 1px dashed var(--bt-border); font-size: 0.9rem; opacity: 0.85; background: rgba(128,128,128,0.04);}}
.bt-empty-i {{font-size: 1.1rem; color: var(--bt-accent);}}
.bt-qs {{min-height: 118px; padding: 14px; border-radius: 14px; border: 1px solid var(--bt-border); background: var(--bt-surface);
  box-shadow: var(--bt-shadow); margin-bottom: 6px; animation: bt-rise .5s ease both; transition: transform .15s ease;}}
.bt-qs:hover {{transform: translateY(-3px);}}
.bt-qs-t {{font-weight: 850; margin-bottom: 6px;}} .bt-qs-d {{font-size: 0.83rem; opacity: 0.8; line-height: 1.45;}}
@media (max-width: 1100px) {{.bt-context-hint {{display: none;}} .bt-qs {{min-height: 0;}}}}
@media (max-width: 760px) {{.bt-podium {{grid-template-columns: 1fr;}} .bt-pod {{order: 0 !important;}}}}
/* ---------- podium & mission strip ---------- */
.bt-podium {{display: grid; grid-template-columns: 1fr 1.15fr 1fr; gap: 10px; align-items: end; margin: 8px 0 14px 0;}}
.bt-pod {{text-align: center; padding: 14px 10px; border-radius: 14px; border: 1px solid var(--bt-border);
  background: var(--bt-surface); box-shadow: var(--bt-shadow); animation: bt-rise .5s ease both;}}
.bt-pod-0 {{order: 2; padding-top: 26px; background: linear-gradient(160deg, rgba(240,228,66,0.18), var(--bt-surface));}}
.bt-pod-1 {{order: 1;}} .bt-pod-2 {{order: 3;}}
.bt-pod-m {{font-size: 1.8rem;}} .bt-pod-n {{font-weight: 850; margin-top: 4px;}}
.bt-pod-l {{font-size: 0.78rem; opacity: 0.7;}} .bt-pod-v {{font-size: 0.82rem; font-weight: 700; margin-top: 4px;}}
.bt-mstrip {{display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 12px 0;}}
.bt-mchip {{padding: 7px 14px; border-radius: 999px; font-size: 0.85rem; border: 1px solid var(--bt-border);}}
.bt-m-answered {{background: rgba(0,158,115,0.15); border-color: rgba(0,158,115,0.45);}}
.bt-m-partly {{background: rgba(230,159,0,0.15); border-color: rgba(230,159,0,0.45);}}
.bt-m-open {{background: rgba(213,94,0,0.12); border-color: rgba(213,94,0,0.4);}}
/* ---------- 3D battery résumé ---------- */
.bt-resume {{padding: 18px 20px; border-radius: 16px; border: 1px solid var(--bt-border); background: var(--bt-surface);
  box-shadow: var(--bt-shadow); animation: bt-rise .5s ease both;}}
.bt-res-head {{display: flex; align-items: center; gap: 10px;}}
.bt-res-badge {{padding: 3px 10px; border-radius: 8px; color: #fff; font-weight: 800; font-size: 0.75rem; letter-spacing: 0.06em;}}
.bt-res-name {{font-size: 1.5rem; font-weight: 850; letter-spacing: -0.02em;}}
.bt-res-sub {{font-size: 0.85rem; opacity: 0.75; margin: 4px 0 12px 0;}}
.bt-res-row {{display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; margin: 9px 0;}}
.bt-res-l {{font-size: 0.82rem; font-weight: 700; opacity: 0.85;}}
.bt-res-v {{font-size: 0.82rem; font-weight: 800; text-align: right;}}
.bt-res-bar {{grid-column: 1 / span 2; height: 7px; border-radius: 99px; background: rgba(128,128,128,0.18); overflow: hidden;}}
.bt-res-bar > div {{height: 100%; border-radius: 99px; animation: bt-grow 1.1s cubic-bezier(.2,.8,.2,1) both;
  transform-origin: left;}}
@keyframes bt-grow {{from {{transform: scaleX(0);}} to {{transform: scaleX(1);}}}}
.bt-grade {{flex: none; width: 58px; height: 58px; border-radius: 50%; border: 4px solid; display: flex; align-items: center;
  justify-content: center; font-size: 1.7rem; font-weight: 900; animation: bt-pop .6s cubic-bezier(.2,1.6,.4,1) both;}}
@keyframes bt-pop {{from {{transform: scale(0.4); opacity: 0;}} to {{transform: scale(1); opacity: 1;}}}}
.bt-res-head {{gap: 14px;}}
.bt-res-top {{display: flex; align-items: center; gap: 10px;}}
.bt-res-sum {{font-size: 0.9rem; line-height: 1.5; margin: 10px 0 4px 0; padding: 8px 12px; border-radius: 10px;
  background: var(--bt-accent-soft); border: 1px solid var(--bt-accent-line);}}
.bt-tiles {{display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px;}}
.bt-tile {{padding: 8px 6px; border-radius: 10px; text-align: center; border: 1px solid var(--bt-border);
  background: rgba(128,128,128,0.06); animation: bt-rise .5s ease both;}}
.bt-tile-v {{font-weight: 850; font-size: 0.98rem;}}
.bt-tile-l {{font-size: 0.72rem; opacity: 0.7; margin-top: 2px;}}
.bt-res-ev {{margin-top: 10px; font-size: 0.84rem; display: flex; flex-wrap: wrap; gap: 6px; align-items: center;}}
.bt-sev-chip {{padding: 1px 8px; border-radius: 999px; border: 1px solid; font-size: 0.76rem; font-weight: 700;}}
@media (max-width: 1100px) {{.bt-tiles {{grid-template-columns: repeat(2, 1fr);}}}}
.bt-res-alerts {{margin-top: 12px; font-size: 0.82rem; line-height: 1.45; padding: 8px 10px; border-radius: 10px;
  background: rgba(213,94,0,0.08); border: 1px solid rgba(213,94,0,0.25);}}
/* ---------- animated hero gradient ---------- */
.bt-hero {{background-size: 220% 220%; animation: bt-flow 18s ease-in-out infinite;}}
@keyframes bt-flow {{0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}}}}
/* ---------- industrial status bar ---------- */
.bt-statusbar {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 18px; margin: 4px 0 14px 0; padding: 9px 16px;
  border-radius: 12px; font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace; font-size: 0.78rem;
  letter-spacing: 0.06em; background: #0E1726; color: #9FB3C8; border: 1px solid #1F2E45;
  box-shadow: inset 0 0 0 1px rgba(86,180,233,0.08);
}}
.bt-statusbar b {{color: #E6EEF7; font-weight: 700;}}
.bt-statusbar .bt-clock {{margin-left: auto; color: #6B819A;}}
.bt-live {{display: inline-flex; align-items: center; gap: 8px; color: #3DDC97; font-weight: 800;}}
.bt-dot {{width: 9px; height: 9px; border-radius: 50%; background: #3DDC97; box-shadow: 0 0 0 0 rgba(61,220,151,0.7);
  animation: bt-pulse 1.8s infinite;}}
@keyframes bt-pulse {{0% {{box-shadow: 0 0 0 0 rgba(61,220,151,0.65);}} 70% {{box-shadow: 0 0 0 9px rgba(61,220,151,0);}}
  100% {{box-shadow: 0 0 0 0 rgba(61,220,151,0);}}}}
.bt-sev {{padding: 2px 9px; border-radius: 6px; font-weight: 800;}}
.bt-sev-crit {{background: rgba(213,94,0,0.18); color: #FF8A4C; border: 1px solid rgba(213,94,0,0.45);}}
.bt-sev-warn {{background: rgba(230,159,0,0.16); color: #F5C04A; border: 1px solid rgba(230,159,0,0.40);}}
/* ---------- alert banner ---------- */
.bt-alert {{display: flex; gap: 12px; align-items: flex-start; padding: 12px 16px; border-radius: 12px; margin: 6px 0 12px 0;
  border: 1px solid var(--bt-border); background: var(--bt-surface); box-shadow: var(--bt-shadow);}}
.bt-alert .bt-badge {{flex: none; padding: 3px 10px; border-radius: 8px; color: #fff; font-weight: 800; font-size: 0.8rem;}}
.bt-alert .bt-msg {{font-size: 0.93rem; line-height: 1.5; color: inherit;}}
</style>"""
    st.markdown(css, unsafe_allow_html=True)
    # a separate call on one line: a <style> HTML block ends at the line containing </style>, so appending a second
    # block after it would make Markdown print the rules as text
    st.markdown(" ".join(line.strip() for line in DESIGN_CSS.splitlines()), unsafe_allow_html=True)


# =============================================================================
# Small UI helpers
# =============================================================================
def section(title: str, icon: Optional[str] = None) -> None:
    """Numbered section header (numbering restarts in every view)."""
    n = st.session_state.get("_sec_n", 0) + 1
    st.session_state["_sec_n"] = n
    st.markdown(f'<div class="bt-section"><span class="bt-num">{n}</span><span>{html.escape(title)}</span></div>',
                unsafe_allow_html=True)


def card(title: str, lines: Sequence[str]) -> None:
    body = "".join(f"<li>{html.escape(ln)}</li>" for ln in lines)
    st.markdown(f'<div class="bt-card"><h4>{html.escape(title)}</h4>'
                f'<ul style="margin:0;padding-left:20px">{body}</ul></div>', unsafe_allow_html=True)


def columns(spec: Any, **kw: Any):
    """st.columns with graceful fallback for kwargs unsupported by older Streamlit."""
    try:
        return st.columns(spec, **kw)
    except TypeError:
        return st.columns(spec)


def fmt(value: Any, spec: str = ".3f", unit: str = "") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:{spec}}{(' ' + unit) if unit else ''}"


def report_error(where: str, exc: BaseException, verbose: bool) -> None:
    st.error(f"{where}: {exc}")
    if verbose:
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")) or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")


def nav(options: Sequence[str], key: str) -> str:
    """Lazy view switcher: only the returned view is executed on each run."""
    choice, rendered = None, False
    seg = getattr(st, "segmented_control", None)
    if seg is not None:
        try:
            choice = seg("View", list(options), default=options[0], key=key, label_visibility="collapsed")
            rendered = True
        except TypeError:
            rendered = False
    if not rendered:
        choice = st.radio("View", list(options), horizontal=True, key=key, label_visibility="collapsed")
    if choice is None:                     # segmented control deselected: keep the last view
        choice = st.session_state.get(f"{key}_last", options[0])
    st.session_state[f"{key}_last"] = choice
    return choice


def download(label: str, data: Any, file_name: str, mime: str, key: str) -> None:
    """Download button that does not rerun the app when supported (on_click='ignore')."""
    try:
        st.download_button(label, data, file_name=file_name, mime=mime, key=key, on_click="ignore")
    except TypeError:
        st.download_button(label, data, file_name=file_name, mime=mime, key=key)


def manifest_button(config: Dict[str, Any], key: str, data_key: str, extra: Optional[Dict[str, Any]] = None) -> None:
    man = te.run_manifest(config, data_key, extra)
    download("Run manifest (JSON)", json.dumps(man, indent=2), f"{key}_manifest.json", "application/json",
             key=f"man_{key}")


PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,                          # always visible: the camera saves the chart as a named PNG
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def image_name(fig: go.Figure, key: str) -> str:
    """File name of a chart image: chart id, battery (or condition group for group-level charts), chart title, date,
    engine version. Lets a saved image be identified without opening it, e.g.
    acc_board__Reference__prediction-accuracy-assessment__2026-10-14__v5.15.0."""
    def slug(x: Any, n: int = 60, lower: bool = True) -> str:
        x = re.sub(r"<[^>]+>", " ", str(x or ""))
        x = re.sub(r"[^A-Za-z0-9]+", "-", x).strip("-")
        return (x.lower() if lower else x)[:n].strip("-")
    title = getattr(getattr(fig.layout, "title", None), "text", "") if fig is not None else ""
    ss = st.session_state
    where = ss.get("acc_group") if key.startswith("acc") else ss.get("cell")
    parts = [slug(key, 40), slug(where, 20, lower=False), slug(title), date.today().isoformat(), f"v{te.ENGINE_VERSION}"]
    return "__".join(p for p in parts if p)


def plot_config(fig: go.Figure, key: str, **extra: Any) -> Dict[str, Any]:
    return dict(PLOT_CONFIG, toImageButtonOptions={**PLOT_CONFIG["toImageButtonOptions"],
                                                   "filename": image_name(fig, key)}, **extra)


EXPLAIN: Dict[str, str] = {
    # diagnostics
    "bat3d": "Animated battery: the electrolyte level is the state of health, its colour the risk level (green → red), the empty space above it the capacity already lost; glowing dots are lithium ions moving between the electrodes.",
    "gauges": "Instrument cluster: the battery's state of health, quick remaining-life estimate, resistance growth and peak temperature, coloured by risk. Use it to judge at a glance whether this battery needs attention.",
    "risk_matrix": "Each bubble is a battery: remaining life (x) against how fast it is degrading (y). Top-left means little life left and fading fast: act on those first.",
    "fleet_map": "Treemap of the fleet grouped by ambient temperature; tile size = cycles run, colour = remaining health margin. It shows whether problems cluster under one operating condition.",
    "fade_multi": "Capacity (or SOH) against cycle number for the selected batteries. It shows how fast each battery ages, where fade accelerates (★ knee) and where capacity recovers after rest (▲).",
    "trace_multi": "Raw voltage, current and temperature inside the chosen discharges. Late-life cycles reach the cut-off voltage sooner and heat more: the physical signature of ageing.",
    "cohort_": "SOH of the chosen batteries that share one ambient temperature. Comparing panels shows how temperature changes the ageing speed.",
    "hi_rank": "Candidate health indicators scored on four criteria (link to SOH, monotonicity, same trend in every cell, similar end values). The top one is the best single measure of health (Mission 1).",
    "hi_traj": "The best indicators of this battery normalised to their starting value, showing how each drifts as the battery ages.",
    "hi_pca": "Principal component analysis of all indicators. If one component explains most of the variance, one parameter is enough to describe health; otherwise ageing is multi-dimensional.",
    "ica": "Incremental capacity curves (dQ/dV). Peaks are electrode phase transitions; their shrinking and shifting reveal which ageing mechanism is active.",
    "ica_peaks": "Position and height of the main ICA peak over life: height loss points to loss of active material, a downward shift to growing resistance.",
    "dva": "Differential voltage curves (dV/dQ): the distance between peaks tracks the electrode that owns them, so shrinking spacing indicates active-material loss.",
    "modes": "Indicative split of the capacity loss into lithium-inventory loss (LLI), active-material loss (LAM) and conductivity loss (CL) over life.",
    "hc_modes": "Quantitative degradation modes from fitting electrode potentials to the discharge curve: lithium inventory lost and active material lost on each electrode.",
    "hc_fits": "Measured pseudo-OCV curves against the fitted half-cell model, and the electrode potentials behind them: how well the physics explains the data.",
    "stress": "Peak temperature, minimum voltage and cold-charging risk per cycle, with the literature thresholds. Exposure to these stressors explains faster ageing.",
    # live twin
    "rp_3dfig": "The battery as the twin sees it now: the fill drops with the estimated state of health and turns orange or red as the remaining life shortens.",
    "rp_main": "The live twin sees the battery one discharge at a time. Each model forecasts the rest of life from what it has seen so far; grey circles (if revealed) are the future it has not seen.",
    "rp_track": "Top: each model's predicted end of life converging as data arrive. Middle: how much the live ensemble trusts each model. Bottom: which degradation mechanism is consuming the capacity.",
    "abl_fig": "The twin re-run with different measurement sets: the gap between open loop and voltage-only is the information the sensors add (Mission 2).",
    "uf_fig": "Tracking accuracy when the twin is updated every m cycles: it shows how rarely the twin can be updated without losing accuracy (Mission 2).",
    # models
    "acc_board": "Every model with the same held-out batteries and origins. Bars: mean error ± 95% CI; dashed line: the best baseline. A model is only worth using if it beats that line clearly.",
    "pinn_forecast_fig": "Level 4: forecasts of the physics-informed network and of the same network without physics (the ablation), from the training | test line.",
    "pinn_mech": "The PINN's decomposition of the predicted capacity loss into its three mechanisms, from the origin to the last cycle. A late rise of plating is the mechanism behind a knee.",
    "tr_forecast_fig": "Level 2 in forecast mode: cycle-by-cycle predictions from what is known at the training | test line plus the operating conditions.",
    "tr_measured_fig": "Level 2 in measured mode: each test cycle's SOH estimated from that cycle's measured parameters.",
    "dl_forecast_fig": "Level 3 in forecast mode: neural networks, same question and data as Level 2.",
    "dl_measured_fig": "Level 3 in measured mode: neural networks estimating each test cycle's SOH from its measurements (GRU / Transformer: from the last 10 cycles).",
    "tr_forecast_imp": "What the best Level-2 forecast model relies on (held-out batteries).",
    "tr_measured_imp": "What the best Level-2 measured-mode model relies on (held-out batteries).",
    "dl_forecast_imp": "What the MLP relies on (held-out batteries).",
    "dl_measured_imp": "What the MLP relies on (held-out batteries).",
    "tr_fig": "Cycle-by-cycle predictions of models trained on every cycle of the training batteries. In forecast mode they only know the battery's state at the training | test line plus the operating conditions; in measured mode they also see each test cycle's measurements.",
    "tr_imp": "What the best trajectory model relies on: how much its error grows on held-out batteries when one input is shuffled.",
    "lad_base": "Level 1 references: 'nothing changes' (persistence) and 'the recent straight line continues' (linear trend). A model is only useful if it beats these.",
    "lad_deep": "Level 4 deep sequence models (GRU, Transformer) read a window of past SOH values and predict the next cycle, repeatedly, to build the forecast.",
    "lad_hybrid": "Level 5 hybrid models combine physics equations with learning: the mechanistic PINN obeys the degradation kinetics; hierarchical Bayes combines a physics law with fleet knowledge.",
    "lad_spm": "Level 6 first-principles model: capacity follows from simulated discharges of a single-particle electrochemical model while SEI growth removes lithium.",
    "lad_spm_curves": "Simulated discharge curves of the single-particle model: the effect of lost lithium, higher current and cold on voltage and delivered capacity.",
    "lad_board": "All models run so far on this battery and forecast origin, from the simplest to the most advanced, scored on the same future cycles. The best is highlighted.",
    "ml_fig": "Forecasts of the selected ML models from the forecast origin (dotted line), with their uncertainty bands, against the measured SOH.",
    "ml_v2_fig": "ML v2 forecasts: each battery's own recent trend, corrected by what the models learned from the training batteries about acceleration and knees.",
    "ml_v2_imp": "Which measurements drive the v2 forecast: how much the error grows when one feature is shuffled (on held-out batteries, 40-cycle horizon). This answers which variables are most informative.",
    "ml_board": "Leaderboard of the ML models on the held-out cycles (fade skill: 1 = perfect, 0 = no better than assuming no further fade).",
    "ml_parity": "Predicted against measured SOH on the test cycles: points on the diagonal are perfect predictions.",
    "est_board": "SOH estimation from operando indicators: R² on the test set per model.",
    "est_parity": "Estimated against measured SOH for the test cycles, coloured by battery.",
    "est_traj": "Measured SOH (points) and the estimated SOH (line) of the test batteries over life.",
    "est_imp": "How much each indicator matters: the error increase when its values are shuffled.",
    "el_dq": "Change of the discharge curve between an early and a later cycle (ΔQ(V)), coloured by eventual life: large changes early predict a short life.",
    "el_parity": "Cycle life predicted from the first cycles only against the actual life (leave-one-battery-out).",
    "bench_": "Cross-battery benchmark: forecasts from several origins on several batteries, summarised with prognostic metrics.",
    # operations
    "ops_fig": "Simulated life under the chosen operating policy: currents chosen each cycle, SOH and cumulative profit.",
    "mm_fig": "Robustness: the twin-aware policy's advantage when the real battery differs from the model.",
    "om_heat": "Long-run profit rate for each operating policy and replacement threshold; the star is the best compliant combination (Mission 3).",
    "om_trade": "Replacing early costs money; replacing late raises the risk of sudden failure. The optimum balances the two.",
    "dp_fig": "Dynamic-programming policy: the best action (current or replace) for every state of health and season.",
    "scen_fig": "What-if planner: projected fade under each operating scenario, from the stress law fitted on this cohort.",
    # study
    "st_hi": "Health-indicator ranking across the whole cohort (Mission 1).",
    "st_mech": "Mechanism shares per condition group, one point per battery: plating should dominate in the cold, SEI when hot, LAM under high current.",
    "st_models": "Median forecast error per condition group and model on real data, with interquartile bars (Mission 2).",
}


def _explain_html(text: str) -> str:
    if st.session_state.get("ui_explain", True):
        return (f'<div class="bt-explain"><span class="bt-explain-tag">What this shows</span> '
                f'{html.escape(text)}</div>')
    return f'<div class="bt-explain-mini" title="{html.escape(text, quote=True)}">ⓘ What this shows</div>'


def note(text: str) -> None:
    st.markdown(_explain_html(text), unsafe_allow_html=True)


def placeholder(text: str, icon: str = "hourglass_empty") -> None:
    """Consistent empty state for sections that have not been run yet."""
    st.markdown(f'<div class="bt-empty"><span class="bt-empty-i">▷</span>{html.escape(text)}</div>',
                unsafe_allow_html=True)


def explain(key: str) -> None:
    """Small 'what this shows' paragraph under a chart or table (catalogue lookup, exact key or prefix)."""
    text = EXPLAIN.get(key) or next((v for k, v in EXPLAIN.items() if k.endswith("_") and key.startswith(k)), None)
    if text:
        st.markdown(_explain_html(text), unsafe_allow_html=True)


def show(fig: go.Figure, key: str, data: Optional[pd.DataFrame] = None, export: bool = True) -> None:
    """Render with Plotly styling intact (theme=None) across Streamlit versions, plus an
    export row: interactive HTML, SVG / PDF (when kaleido is installed) and the plotted data."""
    base = dict(theme=None, config=plot_config(fig, key))
    attempts: List[Dict[str, Any]] = []
    if ST_VERSION >= (1, 50):
        attempts.append(dict(width="stretch", key=key))
    attempts += [dict(use_container_width=True, key=key), dict(use_container_width=True)]
    for extra in attempts:
        try:
            st.plotly_chart(fig, **base, **extra)
            break
        except TypeError:
            continue
    explain(key)
    if export and st.session_state.get("ui_export", False):
        export_row(fig, key, data)


def show_selectable(fig: go.Figure, key: str) -> Optional[Any]:
    """Chart with point selection (Streamlit >= 1.35). Returns the selection event or None."""
    cfg = plot_config(fig, key, modeBarButtonsToRemove=["lasso2d"])
    for extra in ([dict(width="stretch")] if ST_VERSION >= (1, 50) else []) + [dict(use_container_width=True)]:
        try:
            ev = st.plotly_chart(fig, theme=None, config=cfg, key=key, on_select="rerun",
                                 selection_mode="points", **extra)
            explain(key)
            return ev
        except TypeError:
            continue
    show(fig, key, export=False)
    return None


def export_row(fig: go.Figure, key: str, data: Optional[pd.DataFrame]) -> None:
    holder = getattr(st, "popover", None)
    ctx = holder("⋯ Export", use_container_width=False) if holder else st.expander("⋯ Export")
    with ctx:
        name = image_name(fig, key)
        download("Interactive HTML", fig.to_html(include_plotlyjs="cdn", full_html=True), f"{name}.html",
                 "text/html", key=f"dl_html_{key}")
        if HAS_KALEIDO:
            try:
                download("PNG (image)", fig.to_image(format="png", scale=2), f"{name}.png", "image/png",
                         key=f"dl_png_{key}")
                download("SVG (vector)", fig.to_image(format="svg"), f"{name}.svg", "image/svg+xml",
                         key=f"dl_svg_{key}")
                download("PDF (vector)", fig.to_image(format="pdf"), f"{name}.pdf", "application/pdf",
                         key=f"dl_pdf_{key}")
            except Exception as exc:                     # kaleido needs a browser runtime
                st.caption(f"Static export unavailable: {exc}")
        else:
            st.caption("Install `kaleido` for SVG / PDF export here; the camera icon on the chart always saves a PNG.")
        if data is not None and len(data):
            download("Plotted data (CSV)", data.to_csv(index=False), f"{name}.csv", "text/csv",
                     key=f"dl_csv_{key}")


def show_table(data: Any, note: Optional[str] = None) -> None:
    _show_table(data)
    if note:
        st.markdown(_explain_html(note), unsafe_allow_html=True)


def _show_table(data: Any) -> None:
    if ST_VERSION >= (1, 50):
        try:
            st.dataframe(data, width="stretch")
            return
        except TypeError:
            pass
    st.dataframe(data, use_container_width=True)


def paradigm_style(name: str, P: Palette, i: int = 0) -> Tuple[str, str, str]:
    """(colour, dash, marker) for a forecast series - colour AND dash encode the paradigm."""
    if name.startswith("ECM twin"):
        return P.ekf, PARADIGM_DASH["twin"], "square"
    if name == te.PINN_NAME:
        return P.pinn, PARADIGM_DASH["pinn"], "diamond"
    if name == te.HB_NAME:
        return "#882255", "dashdot", "hexagon"
    return P.series[i % len(P.series)], DASHES[(i + 3) % len(DASHES)], SYMBOLS[i % len(SYMBOLS)]


def _legend_entries(fig: go.Figure) -> List[str]:
    names, groups = [], set()
    for tr in fig.data:
        if tr.showlegend is False or not tr.name:
            continue
        if tr.legendgroup:
            if tr.legendgroup in groups:
                continue
            groups.add(tr.legendgroup)
        names.append(str(tr.name))
    return names


def _legend_rows(names: Sequence[str], width_px: int) -> int:
    needed = sum(7.2 * len(n) + 48 for n in names)
    return max(1, math.ceil(needed / max(width_px - 80, 200)))


def _style_subplot_titles(fig: go.Figure, P: Palette) -> None:
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=P.text, family=P.font)


AXIS_BLOCK_PX = 62      # tick labels + x-axis title below the plot area
LEGEND_ROW_PX = 24
TITLE_BAND_PX = 92


def style_fig(fig: go.Figure, P: Palette, height: int = 460, title: Optional[str] = None,
              width_hint: int = 1100, hovermode: str = "x unified") -> go.Figure:
    """Publication layout: reserved title band on top, reserved legend band at the bottom
    (below the x-axis title), unified cross-hair hover."""
    names = _legend_entries(fig)
    rows = _legend_rows(names, width_hint) if names else 0
    legend_px = rows * LEGEND_ROW_PX + 14 if rows else 0
    top = TITLE_BAND_PX if title else 40
    bottom = AXIS_BLOCK_PX + (legend_px + 18 if rows else 0)
    H = int(height + max(rows - 1, 0) * LEGEND_ROW_PX)

    if LEGEND_CONTAINER_REF:
        legend_pos = dict(yref="container", y=8 / H, yanchor="bottom")
    else:
        plot_h = max(H - top - bottom, 60)
        legend_pos = dict(y=-(AXIS_BLOCK_PX + 8) / plot_h, yanchor="top")

    fig.update_layout(
        template=P.template, height=H, showlegend=bool(names),
        paper_bgcolor=P.paper_bg, plot_bgcolor=P.plot_bg,
        font=dict(family=P.font, size=13, color=P.text),
        margin=dict(l=76, r=30, t=top, b=bottom, pad=4),
        legend=dict(orientation="h", xanchor="center", x=0.5,
                    bgcolor=P.legend_bg, bordercolor=P.legend_border, borderwidth=1,
                    font=dict(family=P.font, size=12, color=P.text),
                    itemsizing="constant", **legend_pos),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=P.hover_bg, bordercolor=P.legend_border, namelength=-1,
                        font=dict(family=P.font, size=12, color=P.text)),
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{html.escape(title)}</b>", x=0.012, xanchor="left",
                                     y=1 - 16 / H, yanchor="top",
                                     font=dict(family=P.font, size=16, color=P.text)))
    axis = dict(gridcolor=P.grid, zerolinecolor=P.grid, linecolor=P.axis, showline=True, mirror=True,
                ticks="outside", tickcolor=P.axis, title_standoff=10,
                title_font=dict(family=P.font, size=13, color=P.text),
                tickfont=dict(family=P.font, size=12, color=P.muted))
    fig.update_xaxes(**axis, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor=P.muted)
    fig.update_yaxes(**axis)
    return fig


def _forecast_frame(fig: go.Figure, n0: int, n_max: float, soh_eol: float, P: Palette,
                    row: Optional[int] = None) -> None:
    kw = dict(row=row, col=1) if row else {}
    fig.add_vrect(x0=n0, x1=n_max, fillcolor=rgba(P.accent, 0.06), line_width=0, layer="below", **kw)
    fig.add_vline(x=n0, line_dash="dot", line_color=P.accent, line_width=1.5, **kw)
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, line_width=1.5,
                  annotation_text="EOL", annotation_position="bottom left",
                  annotation_font=dict(color=P.eol, size=12), **kw)


def _origin_label(fig: go.Figure, n0: int, P: Palette) -> None:
    fig.add_annotation(x=n0, xref="x", y=1.0, yref="paper", xanchor="left", yanchor="bottom",
                       text=f"<b> forecast origin n₀ = {n0}</b>", showarrow=False,
                       font=dict(color=P.accent, size=12, family=P.font))


def add_band(fig: go.Figure, n: np.ndarray, lo: np.ndarray, hi: np.ndarray, color: str, name: str,
             n_from: Optional[float] = None, group: Optional[str] = None, showlegend: bool = True,
             row: Optional[int] = None, col: Optional[int] = None, alpha: float = 0.16) -> None:
    n, lo, hi = np.asarray(n, float), np.asarray(lo, float), np.asarray(hi, float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if n_from is not None:
        m &= n >= n_from
    if not m.any():
        return
    x = np.r_[n[m], n[m][::-1]]
    y = np.r_[hi[m], lo[m][::-1]]
    fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=rgba(color, alpha), line=dict(width=0),
                             name=name, legendgroup=group, showlegend=showlegend, hoverinfo="skip"),
                  row=row, col=col)


# =============================================================================
# Figures: data & diagnostics
# =============================================================================


def _break_gaps(x: np.ndarray, y: np.ndarray, factor: float = 5.0, min_gap: float = 8.0) -> Tuple[List, List]:
    """Insert None where consecutive cycles are far apart, so missing data are shown as gaps
    instead of long straight lines."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return list(x), list(y)
    dx = np.diff(x)
    step = float(np.median(dx)) if len(dx) else 1.0
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if dx[i - 1] > max(factor * step, min_gap):
            xs.append(None)
            ys.append(None)
        xs.append(x[i])
        ys.append(y[i])
    return xs, ys


def fig_cohort_group(ct_all: pd.DataFrame, meta: pd.DataFrame, group_cells: Sequence[str], highlight: Sequence[str],
                     P: Palette, title: str, normalise_x: bool = False) -> go.Figure:
    """One ambient-temperature group with its own legend (one entry per battery)."""
    good = ct_all[~ct_all["outlier"]]
    allc = list(meta.index)
    fig = go.Figure()
    for cid in sorted(group_cells):
        d = good[good["Cell_ID"] == cid].sort_values("n")
        if d.empty:
            continue
        col, sym = cell_style(cid, allc)
        is_t = cid in highlight
        x = (d["n"] / d["n"].max()).to_numpy() if normalise_x else d["n"].to_numpy()
        xs, ys = _break_gaps(d["n"].to_numpy(), d["SOH"].to_numpy())
        if normalise_x:
            nmax = float(d["n"].max())
            xs = [v / nmax if v is not None else None for v in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=cell_label(cid, meta) + ("  ★" if is_t else ""),
            line=dict(color=col, width=3.6 if is_t else 1.8), connectgaps=False,
            marker=dict(symbol=sym, size=7 if is_t else 5, maxdisplayed=16, line=dict(color=P.plot_bg, width=0.5)),
            opacity=1.0 if (is_t or not highlight) else 0.75,
            hovertemplate=f"<b>{cid}</b> n=%{{x}}: SOH %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title_text="Fraction of recorded life" if normalise_x else "Discharge cycle n")
    fig.update_yaxes(title_text="SOH (%)", tickformat=".0%")
    return style_fig(fig, P, 470, title, hovermode="closest")


TRACE_MODES = ("Single cycle", "Choose cycles", "Every k-th cycle", "Range of cycles", "All cycles")


def fig_cycle_traces(prep: pd.DataFrame, picks: List[Tuple[int, int]], P: Palette, x_axis: str = "time",
                     max_pts: int = 500) -> go.Figure:
    """Overlay raw telemetry of several discharges, coloured along life (Viridis, early = dark).
    x_axis = "time" (minutes into the cycle) or "capacity" (discharged Ah: aligns curves of
    different length and shows capacity fade and polarisation directly)."""
    picks = sorted(picks, key=lambda p: p[1])
    many = len(picks) > 10
    cols = sample_colorscale("Viridis", list(np.linspace(*P.ica_range, max(len(picks), 2))))[: len(picks)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Terminal voltage", "Current", "Cell temperature"))
    _style_subplot_titles(fig, P)
    ns = [n for _, n in picks]
    for (ci, n), col in zip(picks, cols):
        d = prep[prep["Cycle_Index"] == ci]
        if d.empty:
            continue
        if len(d) > max_pts:
            d = d.iloc[np.linspace(0, len(d) - 1, max_pts).astype(int)]
        if x_axis == "capacity":
            x = np.cumsum(np.clip(-d["Current_A"].to_numpy(), 0, None) * d["dt"].to_numpy()) / 3600.0
        else:
            x = d["Time_s"].to_numpy() / 60.0
        for row, key, unit in ((1, "Voltage_V", "V"), (2, "Current_A", "A"), (3, "Temp_C", "°C")):
            fig.add_trace(go.Scatter(
                x=x, y=d[key], mode="lines", name=f"n = {n}", legendgroup=f"n{n}",
                showlegend=(row == 1 and not many), line=dict(color=col, width=1.4 if many else 2.2),
                hovertemplate=f"n = {n}: %{{y:.3f}} {unit}<extra></extra>"), row=row, col=1)
    if many:                                      # colour bar instead of a long legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=[[0, cols[0]], [1, cols[-1]]], cmin=min(ns), cmax=max(ns),
                                             color=[min(ns)], showscale=True,
                                             colorbar=dict(title=dict(text="cycle n", font=dict(color=P.text)),
                                                           tickfont=dict(color=P.muted), thickness=12, len=0.9))),
                      row=1, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="A", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_xaxes(title_text="Discharged capacity (Ah)" if x_axis == "capacity" else "Time in cycle (min)",
                     row=3, col=1)
    title = (f"Raw telemetry, discharge n = {ns[0]}" if len(ns) == 1 else
             f"Raw telemetry of {len(ns)} discharges (n = {min(ns)} to {max(ns)})")
    return style_fig(fig, P, 720, title)


def fig_fade_compare(ct_all: pd.DataFrame, meta: pd.DataFrame, cells: Sequence[str], y: str,
                     eol_line: Optional[float], P: Palette, knees: Dict[str, Dict[str, Any]],
                     show_cohort: bool = True, x_mode: str = "n") -> go.Figure:
    """Fade trajectories of 1 - 8 selected batteries (own colour + marker each) over the muted cohort;
    regeneration (▲), outliers (×) and knee points (★) per selected cell."""
    fig = go.Figure()
    allc = list(meta.index)
    xcol = {"n": "n", "Ah": "cum_Ah", "frac": "_frac"}[x_mode]
    d_all = ct_all.copy()
    d_all["_frac"] = d_all["n"] / d_all.groupby("Cell_ID")["n"].transform("max")
    if show_cohort:
        bg = d_all[~d_all["outlier"] & ~d_all["Cell_ID"].isin(cells)].sort_values(["Cell_ID", xcol])
        xs, ys = [], []
        for _, d in bg.groupby("Cell_ID"):
            xs += d[xcol].tolist() + [None]
            ys += d[y].tolist() + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Other cells", line=dict(color=P.cohort, width=1),
                                 hoverinfo="skip", opacity=0.7))
    for k_, cid in enumerate(cells):
        d = d_all[d_all["Cell_ID"] == cid].sort_values("n")
        good = d[~d["outlier"]]
        col, sym = cell_style(cid, allc)
        fig.add_trace(go.Scatter(x=good[xcol], y=good[y], mode="lines+markers", name=cell_label(cid, meta),
                                 legendgroup=cid, showlegend=k_ < 8 or len(cells) <= 10, customdata=np.column_stack([good["n"], [cid] * len(good)]),
                                 line=dict(color=col, width=2.8), marker=dict(symbol=sym, size=6, color=col),
                                 hovertemplate=f"<b>{cid}</b> n=%{{customdata[0]}}: %{{y:.4f}}<extra></extra>"))
        rg = good[good["regen"]]
        if len(rg):
            fig.add_trace(go.Scatter(x=rg[xcol], y=rg[y], mode="markers", name=f"{cid} regeneration",
                                     legendgroup=cid, showlegend=False,
                                     marker=dict(symbol="triangle-up", size=11, color=col,
                                                 line=dict(color=P.text, width=1)),
                                     hovertemplate=f"{cid} regeneration n=%{{x}}<extra></extra>"))
        ol = d[d["outlier"]]
        if len(ol):
            fig.add_trace(go.Scatter(x=ol[xcol], y=ol[y].clip(upper=1.2 if y == "SOH" else None), mode="markers",
                                     name=f"{cid} excluded", legendgroup=cid, showlegend=False,
                                     marker=dict(symbol="x", size=8, color=col, opacity=0.6),
                                     hovertemplate=f"{cid} excluded point<extra></extra>"))
        k = knees.get(cid, {})
        if k.get("found"):
            kr = good.iloc[(good["n"] - k["knee_n"]).abs().argsort()[:1]]
            fig.add_trace(go.Scatter(x=kr[xcol], y=kr[y], mode="markers", name=f"{cid} knee", legendgroup=cid,
                                     showlegend=False, marker=dict(symbol="star", size=18, color=col,
                                                                   line=dict(color=P.text, width=1.5)),
                                     hovertemplate=f"{cid} knee at n = {k['knee_n']}<extra></extra>"))
    if eol_line is not None:
        fig.add_hline(y=eol_line, line_dash="dash", line_color=P.eol, line_width=1.5,
                      annotation_text="End of life", annotation_position="bottom right",
                      annotation_font=dict(color=P.eol, size=12))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="▲ regeneration · × excluded · ★ knee",
                             marker=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"))
    fig.update_xaxes(title_text={"n": "Discharge cycle n", "Ah": "Cumulative throughput (Ah)",
                                 "frac": "Fraction of recorded life"}[x_mode])
    fig.update_yaxes(title_text="State of health (%)", tickformat=".0%" if y == "SOH" else "Capacity (Ah)")
    ttl = (f"Capacity fade, {cells[0]}" if len(cells) == 1 else f"Capacity fade: {len(cells)} batteries compared")
    return style_fig(fig, P, 560, ttl, hovermode="closest")


def fig_ica(curves: List[te.ICACurve], P: Palette) -> go.Figure:
    fig = go.Figure()
    lo, hi = P.ica_range
    cols = sample_colorscale("Viridis", list(np.linspace(lo, hi, max(len(curves), 2))))
    for c, col in zip(curves, cols):
        fig.add_trace(go.Scatter(x=c.voltage, y=c.dqdv, mode="lines", name=f"n = {c.n}",
                                 line=dict(color=col, width=2.4),
                                 hovertemplate=f"n = {c.n}<br>%{{x:.3f}} V · %{{y:.2f}} Ah/V<extra></extra>"))
        fig.add_trace(go.Scatter(x=[c.peak_V], y=[c.peak_height], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=10, line=dict(color=P.text, width=1.2)),
                                 hovertemplate=f"peak, n = {c.n}<br>%{{x:.3f}} V<extra></extra>"))
    fig.update_xaxes(title_text="Cell voltage (V)", autorange="reversed")
    fig.update_yaxes(title_text="dQ/dV (Ah V⁻¹)")
    return style_fig(fig, P, 470, "Incremental capacity (dQ/dV) across life", width_hint=900,
                     hovermode="closest")


def fig_peaks(curves: List[te.ICACurve], P: Palette) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = [c.n for c in curves]
    fig.add_trace(go.Scatter(x=n, y=[c.peak_V for c in curves], mode="lines+markers", name="Peak voltage",
                             line=dict(color=P.accent, width=2.5), marker=dict(size=7, symbol="circle"),
                             hovertemplate="%{y:.3f} V<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=n, y=[c.peak_height for c in curves], mode="lines+markers", name="Peak height",
                             line=dict(color=P.pinn, width=2.5, dash="dash"), marker=dict(size=7, symbol="diamond"),
                             hovertemplate="%{y:.2f} Ah/V<extra></extra>"), secondary_y=True)
    fig.update_xaxes(title_text="Discharge cycle n")
    style_fig(fig, P, 340, "Main peak evolution", width_hint=620)
    fig.update_yaxes(title_text="Peak voltage (V)", title_font_color=P.accent, secondary_y=False)
    fig.update_yaxes(title_text="Peak height (Ah V⁻¹)", title_font_color=P.pinn, showgrid=False,
                     mirror=False, secondary_y=True)
    return fig


# =============================================================================
# Figures: models & forecasting
# =============================================================================
def fig_ml(results: List[te.MLForecast], ct_cell: pd.DataFrame, n0: int, soh_eol: float,
           P: Palette) -> go.Figure:
    fig = go.Figure()
    good = ct_cell[~ct_cell["outlier"]]
    for i, r in enumerate(results):
        if r.soh_lo is not None:
            col, _ = model_style(r.model)
            add_band(fig, r.n_grid, r.soh_lo, r.soh_hi, col, f"{r.model} {int(100 * (r.band_level or 0.9))}% band",
                     n_from=n0, group=r.model, alpha=0.12)
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.2%}<extra></extra>"))
    for i, r in enumerate(results):
        col, sym = model_style(r.model)
        dash = DASHES[i % len(DASHES)]
        fig.add_trace(go.Scatter(x=r.n_grid, y=r.soh_pred, mode="lines+markers", name=r.model, legendgroup=r.model,
                                 line=dict(color=col, width=2.6, dash=dash),
                                 marker=dict(symbol=sym, size=7, maxdisplayed=10),
                                 hovertemplate="%{y:.2%}<extra></extra>"))
    n_max = max(float(r.n_grid.max()) for r in results)
    _forecast_frame(fig, n0, n_max, soh_eol, P)
    _origin_label(fig, n0, P)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of health (%)", tickformat=".0%")
    return style_fig(fig, P, 520, "ML surrogate SOH forecasts with cross-cell conformal bands")


def model_style(name: str) -> Tuple[str, str]:
    models = list(te.ML_MODELS)
    i = models.index(name) if name in models else 0
    return CELL_COLORS[(i * 5) % len(CELL_COLORS)], CELL_SYMBOLS[i % len(CELL_SYMBOLS)]


def fig_leaderboard(tbl: pd.DataFrame, metric: str, P: Palette, title: str, higher_better: bool = True) -> go.Figure:
    t = tbl.dropna(subset=[metric]).sort_values(metric, ascending=not higher_better)
    fig = go.Figure(go.Bar(
        y=t.index, x=t[metric], orientation="h", text=[f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}" for v in t[metric]],
        textposition="outside", textfont=dict(color=P.text),
        marker=dict(color=[model_style(m)[0] for m in t.index], line=dict(color=P.text, width=0.5)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text=metric)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return style_fig(fig, P, 140 + 34 * len(t), title, hovermode="closest")


def fig_parity(df: pd.DataFrame, color_by: str, P: Palette, title: str, all_cells: Sequence[str] = ()) -> go.Figure:
    """Measured vs predicted SOH (test points), one colour per model or per battery."""
    fig = go.Figure()
    lo = float(min(df["SOH"].min(), df["SOH_pred"].min())) - 0.01
    hi = float(max(df["SOH"].max(), df["SOH_pred"].max())) + 0.01
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=P.muted, dash="dash", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo], y=[lo - 0.02, hi - 0.02, hi + 0.02, lo + 0.02], fill="toself",
                             fillcolor=rgba(P.muted, 0.10), line=dict(width=0), name="±0.02 SOH", hoverinfo="skip"))
    for key, d in df.groupby(color_by, sort=False):
        col, sym = model_style(key) if color_by == "model" else cell_style(key, all_cells or df[color_by].unique())
        fig.add_trace(go.Scatter(x=d["SOH"], y=d["SOH_pred"], mode="markers", name=str(key),
                                 marker=dict(color=col, symbol=sym, size=7, opacity=0.8,
                                             line=dict(color=P.plot_bg, width=0.5)),
                                 hovertemplate=f"{key}<br>measured %{{x:.4f}}<br>predicted %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="Measured SOH", range=[lo, hi])
    fig.update_yaxes(title_text="Predicted SOH", range=[lo, hi], scaleanchor="x", scaleratio=1)
    return style_fig(fig, P, 620, title, hovermode="closest")


def fig_importance(imp: pd.DataFrame, P: Palette, model: str) -> go.Figure:
    t = imp.sort_values("Importance (ΔRMSE)")
    fig = go.Figure(go.Bar(y=t["Indicator"], x=t["Importance (ΔRMSE)"], orientation="h",
                           error_x=dict(type="data", array=t["std"], color=P.muted),
                           marker=dict(color=P.accent), showlegend=False,
                           hovertemplate="%{y}: +%{x:.4f} RMSE when shuffled<extra></extra>"))
    fig.update_xaxes(title_text="RMSE increase when the indicator is shuffled (test set)")
    fig.update_yaxes(automargin=True)
    return style_fig(fig, P, 160 + 36 * len(t), f"Which indicators does {model} rely on? Permutation importance",
                     hovermode="closest")


def fig_est_traj(pred: pd.DataFrame, P: Palette, model: str, all_cells: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    for cid, d in pred.groupby("Cell_ID"):
        col, sym = cell_style(cid, all_cells)
        d = d.sort_values("n")
        fig.add_trace(go.Scatter(x=d["n"], y=d["SOH"], mode="markers", name=f"{cid} measured", legendgroup=cid,
                                 marker=dict(color=col, symbol=sym, size=6, opacity=0.55),
                                 hovertemplate=f"{cid} measured %{{y:.4f}}<extra></extra>"))
        te_ = d[d["set"] == "test"]
        fig.add_trace(go.Scatter(x=te_["n"], y=te_["SOH_pred"], mode="lines", name=f"{cid} estimated (test)",
                                 legendgroup=cid, line=dict(color=col, width=2.6),
                                 hovertemplate=f"{cid} estimated %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="SOH (%)", tickformat=".0%")
    return style_fig(fig, P, 520, f"{model}: SOH estimated from operando indicators on the test cycles")


def fig_compare(res: te.ComparisonResult, P: Palette) -> go.Figure:
    fig = go.Figure()
    for i, (name, (n_g, lo, hi)) in enumerate(res.bands.items()):
        col, _, _ = paradigm_style(name, P, i)
        add_band(fig, n_g, lo, hi, col, f"{name} {int(100 * res.band_level)}% band", n_from=res.n0,
                 group=name, alpha=0.13)
    m = res.measured
    fig.add_trace(go.Scatter(x=m["n"], y=m["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.2%}<extra></extra>"))
    if res.ekf is not None:
        e = res.ekf.per_cycle.dropna(subset=["SOH"])
        add_band(fig, e["n"], e["SOH"] - 2 * e["SOH_std"], e["SOH"] + 2 * e["SOH_std"], P.ekf,
                 "Observer ±2σ", group="obs", alpha=0.10)
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name="Observer causal estimate",
                                 legendgroup="obs", line=dict(color=P.ekf, width=1.8, dash="dot"),
                                 hovertemplate="%{y:.2%}<extra></extra>"))
    for i, (name, (n_g, p_g)) in enumerate(res.predictions.items()):
        col, dash, sym = paradigm_style(name, P, i)
        mask = n_g >= res.n0
        fig.add_trace(go.Scatter(x=n_g[mask], y=p_g[mask], mode="lines+markers", name=f"{name} forecast",
                                 legendgroup=name, line=dict(color=col, width=3, dash=dash),
                                 marker=dict(symbol=sym, size=8, maxdisplayed=8),
                                 hovertemplate="%{y:.2%}<extra></extra>"))
    n_max = max(float(n.max()) for n, _ in res.predictions.values()) if res.predictions else float(m["n"].max())
    _forecast_frame(fig, res.n0, n_max, res.soh_eol, P)
    _origin_label(fig, res.n0, P)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of health (%)", tickformat=".0%")
    return style_fig(fig, P, 560, f"Forecast benchmark on {res.cell_id} with uncertainty bands")


def fig_ablation(results: Dict[str, te.EKFResult], ct_cell: pd.DataFrame, P: Palette) -> go.Figure:
    good = ct_cell[~ct_cell["outlier"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=5, opacity=0.8),
                             hovertemplate="%{y:.2%}<extra></extra>"))
    cols = (P.eol, P.eis, P.r_ct, P.ekf)
    for i, (name, r) in enumerate(results.items()):
        e = r.per_cycle
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name=name,
                                 line=dict(color=cols[i % len(cols)], width=2.4, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.2%}<extra></extra>"))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Causal SOH estimate (–)")
    return style_fig(fig, P, 460, "Measurement ablation: what each signal adds to the observer")


# ---------------------------------------------------------------- benchmark --
def _bench_ok(bench: pd.DataFrame) -> pd.DataFrame:
    return bench[bench["error"].isna()] if "error" in bench.columns else bench


def fig_bench_parity(bench: pd.DataFrame, alpha: float, P: Palette) -> go.Figure:
    b = _bench_ok(bench).dropna(subset=["rul_true", "rul_pred"])
    fig = go.Figure()
    top = float(max(b["rul_true"].max(), b["rul_pred"].max(), 10)) if len(b) else 100.0
    x = np.array([0, top])
    add_band(fig, x, x * (1 - alpha), x * (1 + alpha), P.muted, f"±{int(alpha * 100)}% accuracy cone",
             alpha=0.12)
    fig.add_trace(go.Scatter(x=x, y=x, mode="lines", line=dict(color=P.muted, dash="dash", width=1.2),
                             name="Perfect prediction", hoverinfo="skip"))
    for i, (par, d) in enumerate(b.groupby("paradigm")):
        col, _, sym = paradigm_style(par, P, i)
        fig.add_trace(go.Scatter(x=d["rul_true"], y=d["rul_pred"], mode="markers", name=par,
                                 marker=dict(color=col, symbol=sym, size=9, line=dict(color=P.text, width=0.6)),
                                 customdata=d[["cell", "n0"]].to_numpy(),
                                 hovertemplate="%{customdata[0]} · n₀ %{customdata[1]}<br>true %{x}, pred %{y}"
                                               "<extra></extra>"))
    fig.update_xaxes(title_text="True RUL after n₀ (cycles)")
    fig.update_yaxes(title_text="Predicted RUL (cycles)")
    return style_fig(fig, P, 480, "RUL parity (uncensored forecasts)", width_hint=720, hovermode="closest")


def fig_alpha_lambda(bench: pd.DataFrame, alpha: float, P: Palette) -> go.Figure:
    """Normalised alpha-lambda plot: RUL_pred / RUL_true against the fraction of life elapsed
    lambda = n0 / EOL_true; the alpha cone is 1 +/- alpha."""
    b = _bench_ok(bench).dropna(subset=["rul_true", "rul_pred", "eol_true"])
    b = b[b["rul_true"] > 0]
    fig = go.Figure()
    add_band(fig, np.array([0.0, 1.0]), np.array([1 - alpha] * 2), np.array([1 + alpha] * 2), P.muted,
             f"α = {alpha:.2f} cone", alpha=0.12)
    fig.add_hline(y=1.0, line_dash="dash", line_color=P.muted, line_width=1)
    for i, (par, d) in enumerate(b.groupby("paradigm")):
        col, dash, sym = paradigm_style(par, P, i)
        lam = d["n0"] / d["eol_true"]
        ratio = d["rul_pred"] / d["rul_true"]
        fig.add_trace(go.Scatter(x=lam, y=ratio, mode="markers", name=par,
                                 marker=dict(color=col, symbol=sym, size=9, line=dict(color=P.text, width=0.6)),
                                 customdata=d[["cell"]].to_numpy(),
                                 hovertemplate="%{customdata[0]}: λ %{x:.2f}, ratio %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Fraction of life elapsed λ = n₀ / EOL", range=[0, 1])
    fig.update_yaxes(title_text="RUL_pred / RUL_true", type="log")
    return style_fig(fig, P, 480, "α-λ accuracy across cells", width_hint=720, hovermode="closest")


def fig_horizon(cov: pd.DataFrame, level: float, P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=("Band coverage vs horizon", "RMSE vs horizon"))
    _style_subplot_titles(fig, P)
    for i, (par, d) in enumerate(cov.groupby("paradigm")):
        col, dash, sym = paradigm_style(par, P, i)
        d = d.sort_values("h_bin")
        if d["coverage"].notna().any():
            fig.add_trace(go.Scatter(x=d["h_bin"], y=d["coverage"], mode="lines+markers", name=par, legendgroup=par,
                                     line=dict(color=col, dash=dash, width=2.4), marker=dict(symbol=sym, size=8),
                                     hovertemplate="%{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["h_bin"], y=d["rmse"], mode="lines+markers", name=par, legendgroup=par,
                                 showlegend=not d["coverage"].notna().any(),
                                 line=dict(color=col, dash=dash, width=2.4), marker=dict(symbol=sym, size=8),
                                 hovertemplate="%{y:.2%}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=level, line_dash="dash", line_color=P.eol, row=1, col=1,
                  annotation_text=f"nominal {int(level * 100)}%", annotation_font=dict(color=P.eol, size=11))
    fig.update_yaxes(title_text="Empirical coverage", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="SOH RMSE", row=2, col=1)
    fig.update_xaxes(title_text="Forecast horizon h (cycles)")
    return style_fig(fig, P, 770, "Calibration and error growth over the forecast horizon")


# =============================================================================
# Figures: operations
# =============================================================================




# =============================================================================
# Figures: health indicators, degradation modes, stress (Mission 1)
# =============================================================================
HI_CRITERIA = ("|ρ| with SOH", "Monotonicity", "Trendability", "Prognosability")


def fig_hi_rank(tab: pd.DataFrame, P: Palette) -> go.Figure:
    """Grouped horizontal bars: the four prognostic-parameter criteria per indicator."""
    t = tab.sort_values("Fitness")
    fig = go.Figure()
    for i, crit in enumerate(HI_CRITERIA):
        fig.add_trace(go.Bar(y=t["Indicator"], x=t[crit], orientation="h", name=crit,
                             marker=dict(color=P.mode_colors[i % len(P.mode_colors)],
                                         pattern=dict(shape=("", "/", ".", "x")[i], fgcolor=P.text, size=5,
                                                      solidity=0.15)),
                             hovertemplate="%{x:.2f}<extra>" + crit + "</extra>"))
    fig.add_trace(go.Scatter(y=t["Indicator"], x=t["Fitness"], mode="markers", name="Fitness (mean)",
                             marker=dict(symbol="diamond", size=12, color=P.text, line=dict(color=P.plot_bg, width=1)),
                             hovertemplate="fitness %{x:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(title_text="Criterion score (0 – 1, higher is better)", range=[0, 1.05])
    fig.update_yaxes(title_text=None, automargin=True)
    return style_fig(fig, P, 120 + 42 * len(t), "Which parameter best represents health? Prognostic-parameter criteria",
                     hovermode="closest")


def fig_hi_traj(traj: pd.DataFrame, P: Palette, cell_id: str) -> go.Figure:
    fig = go.Figure()
    for i, (k, d) in enumerate(traj.groupby("key", sort=False)):
        hi = te.HI_CATALOG[k]
        fig.add_trace(go.Scatter(x=d["n"], y=d["rel"], mode="lines", name=f"{hi.label} ({hi.mode})",
                                 line=dict(color=P.series[i % len(P.series)], width=2.4, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.3f}<extra>" + html.escape(hi.label) + "</extra>"))
    fig.add_hline(y=1.0, line_color=P.muted, line_dash="dot", line_width=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Value / beginning-of-life value")
    return style_fig(fig, P, 440, f"Top health indicators on {cell_id}, normalised to beginning of life")


def fig_pca(pca: Dict[str, Any], P: Palette) -> go.Figure:
    ev = pca["explained"][: min(6, len(pca["explained"]))]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=("Variance explained by each component", "Loadings of PC1 and PC2"))
    _style_subplot_titles(fig, P)
    fig.add_trace(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=100 * ev, name="Explained variance",
                         marker=dict(color=P.accent), hovertemplate="%{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=100 * np.cumsum(ev), mode="lines+markers",
                             name="Cumulative", line=dict(color=P.eol, dash="dash"), marker=dict(symbol="square"),
                             hovertemplate="%{y:.1f}%<extra></extra>"), row=1, col=1)
    L = pca["loadings"]
    labels = [te.HI_CATALOG[k].label for k in L.index]
    for j, pc in enumerate([c for c in L.columns[:2]]):
        fig.add_trace(go.Bar(y=labels, x=L[pc], orientation="h", name=f"{pc} loading",
                             marker=dict(color=P.mode_colors[(j + 1) % len(P.mode_colors)],
                                         pattern=dict(shape=("", "/")[j], fgcolor=P.text, solidity=0.15)),
                             hovertemplate="%{x:.2f}<extra>" + pc + "</extra>"), row=2, col=1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="%", range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="Loading", row=2, col=1)
    fig.update_yaxes(automargin=True, row=2, col=1)
    return style_fig(fig, P, 805, "Can degradation be represented by one parameter? PCA of the health indicators",
                     hovermode="closest")


def fig_dva(curves: List[te.DVACurve], P: Palette) -> go.Figure:
    fig = go.Figure()
    ns = [c.n for c in curves]
    cols = sample_colorscale("Viridis", list(np.linspace(*P.ica_range, len(curves))))
    for c, col in zip(curves, cols):
        fig.add_trace(go.Scatter(x=c.q, y=c.dvdq, mode="lines", name=f"n = {c.n}", line=dict(color=col, width=2.2),
                                 hovertemplate="%{x:.2f} Ah: %{y:.2f} V/Ah<extra>n = " + str(c.n) + "</extra>"))
    fig.update_xaxes(title_text="Discharged capacity Q (Ah)")
    fig.update_yaxes(title_text="|dV/dQ| (V/Ah)", type="log")
    return style_fig(fig, P, 440, f"Differential voltage analysis across life (n = {min(ns)} to {max(ns)})")


def fig_modes(modes: pd.DataFrame, P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=("Capacity-loss modes (% of initial capacity)", "Conductivity loss (% growth)"))
    _style_subplot_titles(fig, P)
    for i, col in enumerate(["Capacity loss", "LLI (proxy)", "LAM (proxy)"]):
        if col in modes:
            fig.add_trace(go.Scatter(x=modes["n"], y=modes[col], mode="lines+markers", name=col,
                                     line=dict(color=P.mode_colors[i], width=2.6, dash=DASHES[i]),
                                     marker=dict(symbol=SYMBOLS[i], size=8),
                                     hovertemplate="%{y:.1f}%<extra>" + col + "</extra>"), row=1, col=1)
    for j, col in enumerate([c for c in modes.columns if c.startswith("CL")]):
        fig.add_trace(go.Scatter(x=modes["n"], y=modes[col], mode="lines+markers", name=col,
                                 line=dict(color=P.mode_colors[3 + j % 2], width=2.6, dash=DASHES[3 + j % 2]),
                                 marker=dict(symbol=SYMBOLS[3 + j % 2], size=8),
                                 hovertemplate="%{y:.1f}%<extra>" + col + "</extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return style_fig(fig, P, 770, "Degradation-mode trajectories: LLI · LAM · conductivity loss (CL)")


def fig_stress(sx: pd.DataFrame, limits: te.SafetyLimits, P: Palette, cell_id: str) -> go.Figure:
    d = sx[(sx["Cell_ID"] == cell_id) & ~sx["outlier"]].sort_values("n")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Peak cell temperature per discharge", "Minimum discharge voltage",
                                        "Plating-risk index during charge"))
    _style_subplot_titles(fig, P)
    fig.add_trace(go.Scatter(x=d["n"], y=d["T_max_C"], mode="lines+markers", name="T_max",
                             line=dict(color=P.eol, width=2), marker=dict(size=5),
                             hovertemplate="%{y:.1f} °C<extra></extra>"), row=1, col=1)
    fig.add_hline(y=limits.T_warn_C, line_dash="dot", line_color=P.eis, row=1, col=1,
                  annotation_text=f"accelerated SEI growth ≥ {limits.T_warn_C:.0f} °C",
                  annotation_font=dict(color=P.eis, size=11))
    fig.add_hline(y=limits.T_crit_C, line_dash="dash", line_color=P.eol, row=1, col=1,
                  annotation_text=f"SEI decomposition onset ≈ {limits.T_crit_C:.0f} °C",
                  annotation_font=dict(color=P.eol, size=11))
    fig.add_trace(go.Scatter(x=d["n"], y=d["V_min_V"], mode="lines+markers", name="V_min",
                             line=dict(color=P.accent, width=2), marker=dict(size=5, symbol="square"),
                             hovertemplate="%{y:.3f} V<extra></extra>"), row=2, col=1)
    fig.add_hline(y=limits.V_deep_V, line_dash="dash", line_color=P.eol, row=2, col=1,
                  annotation_text=f"deep discharge < {limits.V_deep_V:.1f} V",
                  annotation_font=dict(color=P.eol, size=11))
    fig.add_trace(go.Bar(x=d["n"], y=d["PRI"], name="Plating-risk index", marker=dict(color=P.r_ct),
                         hovertemplate="%{y:.2f}<extra></extra>"), row=3, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="V", row=2, col=1)
    fig.update_yaxes(title_text="PRI", row=3, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=3, col=1)
    return style_fig(fig, P, 720, f"Operating-stress and safety exposure, {cell_id}", hovermode="x unified")


# =============================================================================
# Figures: Mission 2 and Mission 3 studies
# =============================================================================
def fig_update_freq(tab: pd.DataFrame, results: Dict[int, te.EKFResult], ct_cell: pd.DataFrame, P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=("Causal SOH estimate by update interval", "Accuracy versus measurement cost"))
    _style_subplot_titles(fig, P)
    good = ct_cell[~ct_cell["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=5, opacity=0.8),
                             hovertemplate="%{y:.2%}<extra></extra>"), row=1, col=1)
    for i, (m, r) in enumerate(results.items()):
        e = r.per_cycle
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name=f"every {m} cycle(s)",
                                 line=dict(color=P.series[i % len(P.series)], width=2, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.2%}<extra>every " + str(m) + "</extra>"), row=1, col=1)
    x = tab["Updates per 100 cycles"]
    fig.add_trace(go.Scatter(x=x, y=tab["Tracking RMSE"], mode="lines+markers+text", name="Tracking RMSE",
                             text=[f"m={m}" for m in tab.index], textposition="top center",
                             textfont=dict(color=P.muted, size=11),
                             line=dict(color=P.ekf, width=2.4), marker=dict(size=9),
                             hovertemplate="%{y:.2%}<extra>tracking RMSE</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=tab["Forecast RMSE"], mode="lines+markers", name="Forecast RMSE from n₀",
                             line=dict(color=P.pinn, width=2.4, dash="dash"), marker=dict(symbol="diamond", size=9),
                             hovertemplate="%{y:.2%}<extra>forecast RMSE</extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=1, col=1)
    fig.update_xaxes(title_text="Measurement updates per 100 cycles", type="log", row=2, col=1)
    fig.update_yaxes(title_text="SOH (%)", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="SOH RMSE", row=2, col=1)
    return style_fig(fig, P, 840, "How often should the twin update?")






# =============================================================================
# Figures: operations centre (fleet monitoring)
# =============================================================================
RISK_COLORS = {"Healthy": "#009E73", "Watch": "#E6B800", "Warning": "#E69F00", "Critical": "#D55E00"}
RISK_ICON = {"Healthy": "🟢", "Watch": "🟡", "Warning": "🟠", "Critical": "🔴"}


def fig_gauges(row: pd.Series, P: Palette, limits: te.SafetyLimits) -> go.Figure:
    """Instrument cluster for one battery: SOH, quick RUL, resistance growth, peak temperature."""
    fig = go.Figure()
    eol = float(row["SOH_EOL"])
    common = dict(bgcolor=rgba(P.muted, 0.08), borderwidth=0)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=100 * float(row["SOH"]), number=dict(suffix=" %", font=dict(size=34, color=P.text)),
        title=dict(text="State of health", font=dict(size=14, color=P.muted)), domain=dict(x=[0.0, 0.22], y=[0, 1]),
        gauge=dict(axis=dict(range=[50, 105], tickcolor=P.muted, tickfont=dict(color=P.muted)),
                   bar=dict(color=P.accent, thickness=0.28),
                   steps=[dict(range=[50, 100 * eol], color=rgba("#D55E00", 0.35)),
                          dict(range=[100 * eol, 100 * eol + 10], color=rgba("#E69F00", 0.30)),
                          dict(range=[100 * eol + 10, 105], color=rgba("#009E73", 0.25))],
                   threshold=dict(line=dict(color=P.eol, width=4), thickness=0.85, value=100 * eol), **common)))
    rul = float(row["Quick RUL"])
    fig.add_trace(go.Indicator(
        mode="number", value=rul if np.isfinite(rul) else 999,
        number=dict(suffix=" cyc" if np.isfinite(rul) else "+ cyc", font=dict(size=40, color=RISK_COLORS[row["Risk"]]),
                    valueformat=".0f"),
        title=dict(text=f"Quick RUL<br><span style='font-size:12px'>{RISK_ICON[row['Risk']]} {row['Risk']}</span>",
                   font=dict(size=14, color=P.muted)), domain=dict(x=[0.27, 0.47], y=[0.1, 0.9])))
    rg = float(row["R growth (%)"]) if np.isfinite(row["R growth (%)"]) else 0.0
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=rg, number=dict(suffix=" %", font=dict(size=30, color=P.text), valueformat="+.0f"),
        title=dict(text="Resistance growth", font=dict(size=14, color=P.muted)), domain=dict(x=[0.52, 0.74], y=[0, 1]),
        gauge=dict(axis=dict(range=[min(-20, rg), max(150, rg)], tickcolor=P.muted, tickfont=dict(color=P.muted)),
                   bar=dict(color=P.r_ct, thickness=0.28),
                   steps=[dict(range=[min(-20, rg), 50], color=rgba("#009E73", 0.22)),
                          dict(range=[50, 100], color=rgba("#E69F00", 0.28)),
                          dict(range=[100, max(150, rg)], color=rgba("#D55E00", 0.30))], **common)))
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=float(row["Peak T (°C)"]),
        number=dict(suffix=" °C", font=dict(size=30, color=P.text), valueformat=".1f"),
        title=dict(text="Peak cell temperature", font=dict(size=14, color=P.muted)), domain=dict(x=[0.79, 1.0], y=[0, 1]),
        gauge=dict(axis=dict(range=[0, 80], tickcolor=P.muted, tickfont=dict(color=P.muted)),
                   bar=dict(color=P.eis, thickness=0.28),
                   steps=[dict(range=[0, limits.T_warn_C], color=rgba("#009E73", 0.22)),
                          dict(range=[limits.T_warn_C, limits.T_crit_C], color=rgba("#E69F00", 0.30)),
                          dict(range=[limits.T_crit_C, 80], color=rgba("#D55E00", 0.32))],
                   threshold=dict(line=dict(color=P.eol, width=3), thickness=0.8, value=limits.T_crit_C), **common)))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=P.text))
    return fig


def fig_fleet_map(fs: pd.DataFrame, P: Palette) -> go.Figure:
    """Treemap: fleet → ambient group → battery. Tile area = cycles run, colour = health margin
    (1 = new, 0 = at end of life, < 0 = past it)."""
    d = fs.reset_index()
    d["group"] = d["Ambient_C"].round(0).map(lambda t: f"{t:.0f} °C")
    ids, labels, parents, values, colors, text = ["fleet"], ["Fleet"], [""], [0], [float(d["Health margin"].mean())], [""]
    for g, gg in d.groupby("group"):
        ids.append(f"g/{g}"); labels.append(f"Ambient {g}"); parents.append("fleet"); values.append(0)
        colors.append(float(gg["Health margin"].mean())); text.append(f"{len(gg)} cells")
        for _, r in gg.iterrows():
            ids.append(f"c/{r['Cell_ID']}"); labels.append(r["Cell_ID"]); parents.append(f"g/{g}")
            values.append(max(int(r["Cycles"]), 1)); colors.append(float(r["Health margin"]))
            text.append(f"SOH {100 * r['SOH']:.1f}%<br>{RISK_ICON[r['Risk']]} {r['Risk']}")
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, text=text, branchvalues="remainder",
        texttemplate="<b>%{label}</b><br>%{text}", textfont=dict(size=14),
        marker=dict(colors=colors, colorscale=[[0, "#D55E00"], [0.35, "#E69F00"], [0.6, "#F0E442"], [1, "#009E73"]],
                    cmin=0, cmax=1, line=dict(color=P.plot_bg, width=2),
                    colorbar=dict(title=dict(text="Health margin", font=dict(color=P.text)), tickformat=".0%",
                                  tickfont=dict(color=P.muted), thickness=12)),
        hovertemplate="<b>%{label}</b><br>%{text}<br>health margin %{color:.0%}<extra></extra>",
        pathbar=dict(visible=True)))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=P.text),
                      title=dict(text="Fleet health map (area = cycles run, colour = remaining health margin)",
                                 font=dict(size=16, color=P.text), x=0.01))
    return fig


def fig_risk_matrix(fs: pd.DataFrame, P: Palette, cell_id: str, highlight: Optional[Sequence[str]] = None) -> go.Figure:
    """Risk matrix: remaining life (x) against degradation speed (y). With ``highlight`` the selected
    batteries are drawn in colour and the rest of the fleet as grey context."""
    d_all = fs.reset_index()
    d = d_all[d_all["Cell_ID"].isin(highlight)] if highlight else d_all
    finite = d_all["Quick RUL"].replace(np.inf, np.nan)
    cap = float(np.nanmax(finite)) if np.isfinite(finite).any() else 300
    cap = max(cap * 1.2, 50)
    x = d["Quick RUL"].replace(np.inf, cap).clip(lower=1)
    fig = go.Figure()
    if highlight:
        ctx = d_all[~d_all["Cell_ID"].isin(highlight)]
        if len(ctx):
            fig.add_trace(go.Scatter(
                x=ctx["Quick RUL"].replace(np.inf, cap).clip(lower=1), y=ctx["Fade per 100 cycles (%)"],
                mode="markers", name="Rest of fleet", marker=dict(size=8, color=P.cohort, opacity=0.55),
                text=ctx["Cell_ID"], hovertemplate="%{text}<extra>fleet</extra>"))
    fig.add_vrect(x0=1, x1=15, fillcolor=rgba("#D55E00", 0.10), line_width=0)
    fig.add_vrect(x0=15, x1=50, fillcolor=rgba("#E69F00", 0.08), line_width=0)
    for risk in te.RISK_LEVELS:
        m = d["Risk"] == risk
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[m], y=d.loc[m, "Fade per 100 cycles (%)"], mode="markers+text", name=f"{RISK_ICON[risk]} {risk}",
            text=d.loc[m, "Cell_ID"], textposition="top center", textfont=dict(size=11, color=P.text),
            marker=dict(size=10 + 16 * np.sqrt(d.loc[m, "Cycles"] / d_all["Cycles"].max()), color=RISK_COLORS[risk],
                        opacity=0.85, line=dict(width=[3 if c == cell_id else 1 for c in d.loc[m, "Cell_ID"]],
                                                color=P.text)),
            customdata=np.column_stack([d.loc[m, "SOH"], d.loc[m, "Alerts"]]),
            hovertemplate="<b>%{text}</b><br>quick RUL %{x:.0f} cycles<br>fade %{y:.2f} %/100 cycles<br>"
                          "SOH %{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title_text="Quick RUL (cycles, log scale; right edge = not declining)", type="log")
    fig.update_yaxes(title_text="Recent fade (SOH % per 100 cycles)")
    return style_fig(fig, P, 520, "Risk matrix: remaining life versus degradation speed (size = cycles run)",
                     hovermode="closest")




def build_report_html(title: str, sections: List[Tuple[str, Any]]) -> str:
    """Self-contained HTML report: each section is (heading, plotly figure | DataFrame | text)."""
    parts, first = [], True
    for head, obj in sections:
        parts.append(f"<h2>{html.escape(head)}</h2>")
        if isinstance(obj, go.Figure):
            parts.append(obj.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
            first = False
        elif isinstance(obj, pd.DataFrame):
            parts.append(obj.to_html(classes="tbl", float_format=lambda v: f"{v:.4g}", border=0, na_rep="—"))
        else:
            parts.append(f"<p>{html.escape(str(obj))}</p>")
    css = ("body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1180px;margin:32px auto;color:#1f2933;}"
           "header{background:linear-gradient(120deg,#0B3D91,#0072B2 40%,#009E73);color:#fff;padding:26px 32px;"
           "border-radius:16px}h1{margin:0;font-size:26px}h2{margin-top:34px;border-bottom:2px solid #0072B2;"
           "padding-bottom:6px;font-size:19px}table.tbl{border-collapse:collapse;font-size:13px;width:100%}"
           ".tbl th{background:#eef4fa;text-align:left}.tbl td,.tbl th{padding:6px 10px;border-bottom:1px solid #dde3ea}")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body><header><h1>{html.escape(title)}</h1>"
            f"<div>Battery digital twin · engine {te.ENGINE_VERSION} · generated {stamp}</div></header>"
            + "".join(parts) + "</body></html>")


# =============================================================================
# Figures: live replay, DP policy, half-cell fitting
# =============================================================================


LIVE_COLORS = {"twin": "#0072B2", "mech": "#882255", "pf": "#CC79A7", "trend": "#332288",
               "hb": "#CC79A7", "ens": None, "frozen": "#D55E00"}
LIVE_DASH = {"twin": "dash", "mech": "solid", "pf": "dot", "trend": "dashdot", "hb": "longdash",
             "ens": "solid", "frozen": "solid"}
MECH_COLORS = {"SEI": "#0072B2", "plating": "#56B4E9", "LAM": "#AA4499"}


def _live_color(m: str, P: Palette) -> str:
    return LIVE_COLORS.get(m) or P.text


def fig_live_main(ct_cell: pd.DataFrame, pc: pd.DataFrame, fr: Any, n: int, soh_eol: float, P: Palette,
                  reveal: bool, n_max: int, show_models: Sequence[str],
                  acc: Optional[Dict[str, float]] = None) -> go.Figure:
    good = ct_cell[~ct_cell["outlier"]]
    seen, future = good[good["n"] <= n], good[good["n"] > n]
    est = pc[pc["n"] <= n]
    fig = go.Figure()
    add_band(fig, est["n"].to_numpy(), (est["SOH"] - 2 * est["SOH_std"]).to_numpy(),
             (est["SOH"] + 2 * est["SOH_std"]).to_numpy(), P.ekf, "Twin estimate ±2σ", group="est", alpha=0.15)
    if fr is not None and "ens" in fr.forecasts and "ens" in show_models:
        med, lo, hi = fr.forecasts["ens"]
        add_band(fig, fr.n_grid, lo, hi, P.accent, "Live ensemble 90% band", group="ens", alpha=0.14)
    if reveal and len(future):
        fig.add_trace(go.Scatter(x=future["n"], y=future["SOH"], mode="markers", name="Future (hidden from models)",
                                 marker=dict(color=P.muted, size=5, opacity=0.35, symbol="circle-open"),
                                 hovertemplate="%{y:.2%}<extra>future</extra>"))
    fig.add_trace(go.Scatter(x=seen["n"], y=seen["SOH"], mode="markers", name="Measured so far",
                             marker=dict(color=P.measured, size=6), hovertemplate="%{y:.2%}<extra>measured</extra>"))
    fig.add_trace(go.Scatter(x=est["n"], y=est["SOH"], mode="lines", name="Twin estimate (causal)", legendgroup="est",
                             line=dict(color=P.ekf, width=2.6), hovertemplate="%{y:.2%}<extra>twin</extra>"))
    if fr is not None:
        for m in show_models:
            if m not in fr.forecasts:
                continue
            med = fr.forecasts[m][0]
            w = fr.weights.get(m)
            a = (acc or {}).get(m)
            label = te.LIVE_MODELS[m] + (f" · acc {a:.1f}%" if a is not None and np.isfinite(a) else "") \
                + (f" · w = {w:.2f}" if w is not None and m != "ens" else "")
            fig.add_trace(go.Scatter(x=fr.n_grid, y=med, mode="lines", name=label, legendgroup=m,
                                     line=dict(color=_live_color(m, P), width=4 if m == "ens" else 2.2, dash=LIVE_DASH[m]),
                                     hovertemplate="%{y:.2%}<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"))
    fig.add_vline(x=n, line_color=P.ekf, line_width=1.5, line_dash="dot",
                  annotation_text=f"now: n = {n}", annotation_font=dict(color=P.ekf, size=12))
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, annotation_text="End of life",
                  annotation_font=dict(color=P.eol))
    lo_y = min(float(good["SOH"].min()), soh_eol) - 0.04
    fig.update_xaxes(title_text="Discharge cycle n", range=[0, n_max])
    fig.update_yaxes(title_text="State of health (%)", tickformat=".0%", range=[lo_y, 1.04])
    return style_fig(fig, P, 560, "Live multi-model twin: every model assimilates one discharge at a time",
                     hovermode="x unified")


def fig_live_track(track: pd.DataFrame, n: int, eol_true: Optional[int], P: Palette, n_max: int,
                   show_models: Sequence[str]) -> go.Figure:
    has_mech = any(c.startswith("share_") for c in track.columns)
    nrow = 3 if has_mech else 2
    fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.42, 0.29, 0.29] if has_mech else [0.55, 0.45],
                        subplot_titles=("Predicted end-of-life cycle converging as data arrive",
                                        "Live ensemble weights (recent 5-cycle-ahead skill)")
                        + (("Which mechanism consumes the capacity? (mechanistic PF, share of loss so far)",)
                           if has_mech else ()))
    _style_subplot_titles(fig, P)
    t = track[track["n"] <= n]
    for m in show_models:
        c = f"eol_{m}"
        if c in t:
            fig.add_trace(go.Scatter(x=t["n"], y=t[c], mode="lines", name=te.LIVE_MODELS[m], legendgroup=m,
                                     line=dict(color=_live_color(m, P), width=3.4 if m == "ens" else 1.8,
                                               dash=LIVE_DASH[m]), connectgaps=False,
                                     hovertemplate="%{y:.0f}<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"),
                          row=1, col=1)
    if eol_true is not None:
        fig.add_hline(y=eol_true, line_dash="dash", line_color=P.eol, row=1, col=1,
                      annotation_text=f"actual EOL n = {eol_true}", annotation_font=dict(color=P.eol))
    if has_mech:
        for k_ in te.MECH_NAMES:
            c = f"share_{k_}"
            if c in t:
                fig.add_trace(go.Scatter(x=t["n"], y=t[c], mode="lines", stackgroup="mech", name=f"{k_} share",
                                         line=dict(color=MECH_COLORS[k_], width=0.8),
                                         fillcolor=rgba(MECH_COLORS[k_], 0.65),
                                         hovertemplate="%{y:.0%}<extra>" + k_ + "</extra>"), row=3, col=1)
        fig.update_yaxes(title_text="share", range=[0, 1], tickformat=".0%", row=3, col=1)
    for m in [m for m in ("twin", "mech", "pf", "trend", "hb") if f"w_{m}" in t]:
        fig.add_trace(go.Scatter(x=t["n"], y=t[f"w_{m}"], mode="lines", stackgroup="w", name=f"weight · {te.LIVE_MODELS[m]}",
                                 legendgroup=m, showlegend=False, line=dict(color=_live_color(m, P), width=0.8),
                                 fillcolor=rgba(_live_color(m, P), 0.6),
                                 hovertemplate="%{y:.2f}<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"),
                      row=2, col=1)
    fig.update_xaxes(range=[0, n_max])
    fig.update_xaxes(title_text="Discharge cycle n (data assimilated so far)", row=nrow, col=1)
    fig.update_yaxes(title_text="EOL cycle", row=1, col=1)
    fig.update_yaxes(title_text="weight", range=[0, 1], row=2, col=1)
    return style_fig(fig, P, 880 if has_mech else 680, None)




def fig_half_cell_fits(fits: List[te.HalfCellFit], P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=("Pseudo-OCV (IR-compensated) and half-cell model fit",
                                        "Electrode potentials at beginning of life (vs Li/Li⁺)"))
    _style_subplot_titles(fig, P)
    cols = sample_colorscale("Viridis", list(np.linspace(*P.ica_range, max(len(fits), 2))))
    for f, c in zip(fits, cols):
        fig.add_trace(go.Scatter(x=f.q, y=f.v, mode="markers", name=f"n = {f.n} data", legendgroup=f"n{f.n}",
                                 marker=dict(color=c, size=4, opacity=0.55),
                                 hovertemplate="%{y:.3f} V<extra>data</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=f.q, y=f.v_fit, mode="lines", name=f"n = {f.n} fit ({f.rmse_mV:.1f} mV)",
                                 legendgroup=f"n{f.n}", line=dict(color=c, width=2.4),
                                 hovertemplate="%{y:.3f} V<extra>fit</extra>"), row=1, col=1)
    f0 = fits[0]
    q = f0.q
    y = f0.params["y0"] + q / f0.params["Cp_Ah"]
    x = f0.params["x0"] - q / f0.params["Cn_Ah"]
    fig.add_trace(go.Scatter(x=q, y=te.ocp_lco(y), mode="lines", name="Positive: LiCoO₂",
                             line=dict(color=P.accent, width=2.6), hovertemplate="%{y:.3f} V<extra>LCO</extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=q, y=te.ocp_graphite(x), mode="lines", name="Negative: graphite",
                             line=dict(color=P.r_ct, width=2.6), hovertemplate="%{y:.3f} V<extra>graphite</extra>"),
                  row=2, col=1)
    fig.update_xaxes(title_text="Discharged capacity (Ah)", row=2, col=1)
    fig.update_yaxes(title_text="Cell voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Potential vs Li/Li⁺ (V)", row=2, col=1)
    return style_fig(fig, P, 760, None, hovermode="closest")


def fig_half_cell_modes(tab: pd.DataFrame, P: Palette) -> go.Figure:
    fig = go.Figure()
    for i, (col, name) in enumerate((("Capacity loss (%)", "Capacity loss"), ("LLI (%)", "Loss of lithium inventory"),
                                     ("LAM_PE (%)", "LAM positive (LiCoO₂)"), ("LAM_NE (%)", "LAM negative (graphite)"))):
        fig.add_trace(go.Scatter(x=tab["n"], y=tab[col], mode="lines+markers", name=name,
                                 line=dict(color=P.mode_colors[i], width=3 if i else 2, dash=DASHES[i]),
                                 marker=dict(symbol=SYMBOLS[i], size=9),
                                 hovertemplate="%{y:.2f}%<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=P.muted, line_width=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="% of beginning-of-life value")
    return style_fig(fig, P, 500, "Quantitative degradation modes from half-cell fitting")


def fig_dq_curves(ct_all: pd.DataFrame, el: Dict[str, Any], P: Palette) -> go.Figure:
    """Delta-Q(V) = Q_nb(V) - Q_na(V) per cell, coloured by log10 cycle life (Severson et al. 2019, Fig. 2)."""
    tab = el["table"].set_index("Cell_ID")
    lives = tab["life"].dropna()
    lo, hi = (np.log10(lives.min()), np.log10(lives.max())) if len(lives) else (1, 3)
    fig = go.Figure()
    for cid, d in ct_all[~ct_all["outlier"]].groupby("Cell_ID"):
        dq = te.delta_q_curve(d, el["n_a"], el["n_b"])
        if dq is None:
            continue
        life = tab.loc[cid, "life"] if cid in tab.index else np.nan
        frac = (np.log10(life) - lo) / max(hi - lo, 1e-9) if np.isfinite(life) else None
        col = sample_colorscale("Viridis", [float(np.clip(frac, 0, 1))])[0] if frac is not None else P.cohort
        fig.add_trace(go.Scatter(x=1000 * dq, y=te.QV_GRID, mode="lines", name=cid, showlegend=False,
                                 line=dict(color=col, width=2 if frac is not None else 1, dash="solid" if frac is not None else "dot"),
                                 hovertemplate=f"<b>{cid}</b> life {life if np.isfinite(life) else 'censored'}"
                                               "<br>ΔQ %{x:.1f} mAh at %{y:.2f} V<extra></extra>"))
    if len(lives):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale="Viridis", cmin=lo, cmax=hi, color=[lo], showscale=True,
                                             colorbar=dict(title=dict(text="log₁₀ life", font=dict(color=P.text)),
                                                           tickfont=dict(color=P.muted), thickness=12))))
    fig.add_vline(x=0, line_color=P.muted, line_width=1)
    fig.update_xaxes(title_text=f"ΔQ(V) = Q(n={el['n_b']}) − Q(n={el['n_a']})  (mAh)")
    fig.update_yaxes(title_text="Voltage (V)")
    return style_fig(fig, P, 520, "Early-life ΔQ(V) curves, coloured by eventual cycle life (dotted = censored)",
                     hovermode="closest")


def fig_lifetime_parity(el: Dict[str, Any], P: Palette, all_cells: Sequence[str]) -> go.Figure:
    d = el["loco"]
    fig = go.Figure()
    lo, hi = float(min(d["life"].min(), d["life_pred"].min())) * 0.85, float(max(d["life"].max(), d["life_pred"].max())) * 1.15
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect", line=dict(color=P.muted, dash="dash")))
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo], y=[lo * 0.8, hi * 0.8, hi * 1.2, lo * 1.2], fill="toself",
                             fillcolor=rgba(P.muted, 0.1), line=dict(width=0), name="±20 %", hoverinfo="skip"))
    for _, r in d.iterrows():
        col, sym = cell_style(r["Cell_ID"], all_cells)
        fig.add_trace(go.Scatter(x=[r["life"]], y=[r["life_pred"]], mode="markers+text", text=[r["Cell_ID"]],
                                 textposition="top center", textfont=dict(size=10, color=P.muted), showlegend=False,
                                 marker=dict(color=col, symbol=sym, size=11, line=dict(color=P.text, width=0.8)),
                                 hovertemplate=f"<b>{r['Cell_ID']}</b><br>actual %{{x:.0f}} · predicted %{{y:.0f}} cycles"
                                               "<extra></extra>"))
    fig.update_xaxes(title_text="Actual cycles to end of life", type="log")
    fig.update_yaxes(title_text="Predicted from early cycles (leave-one-cell-out)", type="log")
    return style_fig(fig, P, 520, f"Early-life lifetime prediction · MAPE {el['mape_pct']:.1f}% "
                                  f"(cohort-mean baseline {el['baseline_mape_pct']:.1f}%)", hovermode="closest")


# =============================================================================
# Figures: study results (cohort-wide evidence)
# =============================================================================
def fig_cohort_models(summary: pd.DataFrame, P: Palette) -> go.Figure:
    """Median 20-cycle forecast RMSE per condition group and live model, with interquartile bars."""
    d = summary.reset_index()
    groups = [g for g in list(te.GROUP_ORDER) + ["All batteries"] if g in set(d["Group"])]
    fig = go.Figure()
    for m in [m for m in ("twin", "mech", "pf", "trend", "hb", "ens") if m in set(d["model"])]:
        dm = d[d["model"] == m].set_index("Group").reindex(groups)
        col = LIVE_COLORS.get(m) or P.text
        fig.add_trace(go.Bar(
            x=groups, y=dm["median RMSE"], name=te.LIVE_MODELS[m], marker=dict(color=col, line=dict(color=P.text, width=0.4)),
            error_y=dict(type="data", symmetric=False, array=(dm["IQR high"] - dm["median RMSE"]).clip(lower=0),
                         arrayminus=(dm["median RMSE"] - dm["IQR low"]).clip(lower=0), color=P.muted, thickness=1),
            customdata=np.column_stack([dm["cells"].fillna(0), dm["coverage"]]),
            hovertemplate="%{x}<br>median RMSE %{y:.2%}<br>%{customdata[0]:.0f} cells · coverage %{customdata[1]:.0%}"
                          "<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(title_text="Median 20-cycle forecast RMSE (SOH points)", tickformat=".1%")
    fig.update_xaxes(title_text=None)
    return style_fig(fig, P, 520, "Forecast error by operating condition and model (bars: interquartile range)",
                     hovermode="closest")


def fig_mech_groups(mech: pd.DataFrame, P: Palette) -> go.Figure:
    """Mechanism shares attributed by the mechanistic particle filter, per condition group."""
    fig = go.Figure()
    groups = [g for g in te.GROUP_ORDER if g in set(mech["Group"])]
    for k in te.MECH_NAMES:
        fig.add_trace(go.Box(x=mech["Group"], y=mech[f"share_{k}"], name=k, marker_color=MECH_COLORS[k],
                             boxpoints="all", jitter=0.4, pointpos=0, line=dict(width=1.5),
                             hovertemplate="%{x}: %{y:.0%}<extra>" + k + "</extra>"))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=groups)
    fig.update_yaxes(title_text="Share of capacity lost", tickformat=".0%", range=[0, 1])
    return style_fig(fig, P, 480, "Which mechanism dominates under which conditions? (mechanistic PF, one point per battery)",
                     hovermode="closest")


# =============================================================================
# Animated 3D battery (pure CSS 3D, runs by itself: no interaction, no external library)
# =============================================================================
def _hex_rgb(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    return tuple(int(c[k:k + 2], 16) for k in (0, 2, 4))  # type: ignore[return-value]


def _battery_css_block(uid: str, soh: float, color: str, R: int, H: int, n_faces: int = 32, ions: int = 10,
                       spin_s: float = 16.0) -> str:
    """HTML/CSS for one rotating 18650 cell: metal can (shaded panels), electrolyte fill at the SOH level in
    the risk colour with a rippling surface, lithium ions moving between the electrodes, terminals."""
    soh = float(np.clip(soh, 0.0, 1.0))
    r, g, b = _hex_rgb(color)
    w = 2 * R * math.tan(math.pi / n_faces) + 1.2
    R2 = int(R * 0.86)
    w2 = 2 * R2 * math.tan(math.pi / n_faces) + 1.2
    hf = max(int(H * soh), 4)
    faces, fill = [], []
    for k in range(n_faces):
        th = 360.0 * k / n_faces
        light = 0.5 + 0.5 * math.cos(math.radians(th) - 0.7)
        faces.append(f'<i class="f" style="width:{w:.1f}px;height:{H}px;margin-left:{-w / 2:.1f}px;'
                     f'transform:rotateY({th:.2f}deg) translateZ({R}px);'
                     f'background:linear-gradient(90deg,rgba(210,222,235,{0.10 + 0.22 * light:.3f}),'
                     f'rgba(255,255,255,{0.06 + 0.30 * light ** 4:.3f}) 50%,rgba(150,165,180,{0.10 + 0.18 * light:.3f}))"></i>')
        fill.append(f'<i class="f" style="width:{w2:.1f}px;height:{hf}px;top:{H - hf}px;margin-left:{-w2 / 2:.1f}px;'
                    f'transform:rotateY({th:.2f}deg) translateZ({R2}px);'
                    f'background:linear-gradient(180deg,rgba({r},{g},{b},{0.55 + 0.35 * light:.3f}),'
                    f'rgba({int(r * 0.45)},{int(g * 0.45)},{int(b * 0.45)},{0.75 + 0.2 * light:.3f}))"></i>')
    rng = np.random.default_rng(abs(hash(uid)) % 2 ** 32)
    ion = []
    for k in range(ions):
        ang = rng.uniform(0, 360)
        rad = rng.uniform(0.15, 0.75) * R2
        delay = rng.uniform(0, 3.5)
        dur = rng.uniform(2.4, 4.2)
        ion.append(f'<b class="ion" style="--a:{ang:.0f}deg;--r:{rad:.0f}px;--top:{H - 6}px;--bot:{H - hf + 8}px;'
                   f'animation-delay:-{delay:.2f}s;animation-duration:{dur:.2f}s"></b>')
    return (f'<div class="cell" style="width:{2 * R + 40}px;height:{H + 70}px">'
            f'<div class="halo" style="width:{2 * R + 60}px;background:radial-gradient(ellipse at center,'
            f'rgba({r},{g},{b},0.55),rgba({r},{g},{b},0) 70%)"></div>'
            f'<div class="cyl" style="width:{2 * R}px;height:{H}px;top:36px;animation-duration:{spin_s}s">'
            + "".join(fill)
            + f'<i class="disc liquid" style="width:{2 * R2}px;height:{2 * R2}px;left:{R - R2}px;top:{H - hf - R2}px;'
              f'background:radial-gradient(circle at 40% 40%,rgba(255,255,255,0.55),rgba({r},{g},{b},0.9) 45%,'
              f'rgba({int(r * 0.6)},{int(g * 0.6)},{int(b * 0.6)},0.95))"></i>'
            + "".join(ion) + "".join(faces)
            + f'<i class="disc" style="width:{2 * R}px;height:{2 * R}px;left:0;top:{-R}px;'
              f'background:radial-gradient(circle,#eef3f8,#a9b8c6 70%,#7d8b99)"></i>'
              f'<i class="disc term" style="width:{int(0.76 * R)}px;height:{int(0.76 * R)}px;left:{int(0.62 * R)}px;'
              f'top:{-int(0.38 * R)}px"></i>'
              f'<i class="disc" style="width:{2 * R}px;height:{2 * R}px;left:0;top:{H - R}px;'
              f'background:radial-gradient(circle,#9aa8b6,#5f6d7b)"></i>'
            + '</div></div>')


BATTERY_CSS = """
.wrap{display:flex;gap:18px;justify-content:center;align-items:flex-end;flex-wrap:wrap;font-family:Inter,'Segoe UI',sans-serif;}
.unit{display:flex;flex-direction:column;align-items:center;}
.cell{position:relative;perspective:900px;}
.cyl{position:absolute;left:20px;transform-style:preserve-3d;animation:spin linear infinite;}
.cyl .f{position:absolute;left:50%;top:0;display:block;backface-visibility:visible;}
.disc{position:absolute;display:block;border-radius:50%;transform:rotateX(90deg);}
.term{background:radial-gradient(circle,#ffffff,#cfd8e1 60%,#9aa8b6);box-shadow:0 0 8px rgba(255,255,255,0.6);}
.liquid{animation:ripple 2.6s ease-in-out infinite;}
.ion{position:absolute;left:50%;width:6px;height:6px;margin-left:-3px;border-radius:50%;background:#bff6ff;
  box-shadow:0 0 8px 3px rgba(127,227,255,0.9);animation:ion linear infinite;display:block;}
.halo{position:absolute;left:-10px;bottom:0;height:46px;border-radius:50%;animation:pulse 2.4s ease-in-out infinite;}
@keyframes spin{from{transform:rotateX(-16deg) rotateY(0deg)}to{transform:rotateX(-16deg) rotateY(360deg)}}
@keyframes ripple{0%,100%{transform:rotateX(90deg) scale(1)}50%{transform:rotateX(90deg) scale(0.94)}}
@keyframes ion{0%{transform:rotateY(var(--a)) translateZ(var(--r)) translateY(var(--top));opacity:0}
  15%{opacity:1}85%{opacity:1}100%{transform:rotateY(var(--a)) translateZ(var(--r)) translateY(var(--bot));opacity:0}}
@keyframes pulse{0%,100%{opacity:0.55;transform:scaleX(0.92)}50%{opacity:1;transform:scaleX(1.05)}}
.name{font-weight:850;font-size:var(--fs);letter-spacing:-0.02em;margin-top:4px;}
.soh{font-weight:800;font-size:calc(var(--fs) * 0.75);}
.sub{font-size:12px;opacity:0.75;margin-top:2px;text-align:center;}
@media (prefers-reduced-motion: reduce){.cyl,.liquid,.ion,.halo{animation:none}}
"""


def battery_anim_html(items: Sequence[Dict[str, Any]], text_color: str, big: bool) -> Tuple[str, int]:
    R, H = (72, 240) if big else (40, 132)
    units = []
    for d in items:
        col = RISK_COLORS.get(d.get("risk", "Healthy"), "#009E73")
        units.append(f'<div class="unit">{_battery_css_block(d["id"], d["soh"], col, R, H)}'
                     f'<div class="name" style="--fs:{24 if big else 15}px;color:{text_color}">{html.escape(d["id"])}</div>'
                     f'<div class="soh" style="--fs:{24 if big else 15}px;color:{col}">SOH {100 * d["soh"]:.0f}%</div>'
                     + (f'<div class="sub" style="color:{text_color}">{html.escape(d.get("sub", ""))}</div>' if d.get("sub") else "")
                     + '</div>')
    height = H + 70 + (92 if big else 70)
    doc = (f'<html><head><style>body{{margin:0;background:transparent;overflow:hidden}}{BATTERY_CSS}</style></head>'
           f'<body><div class="wrap">{"".join(units)}</div></body></html>')
    return doc, height


def render_battery_anim(items: Sequence[Dict[str, Any]], big: bool = True, key: str = "") -> None:
    doc, height = battery_anim_html(items, P.text, big)
    try:
        import streamlit.components.v1 as components
        components.html(doc, height=height, scrolling=False)
    except Exception:                                   # fallback: same markup inline
        st.markdown(doc.split("<body>")[1].split("</body>")[0].replace('<div class="wrap">', f'<style>{BATTERY_CSS}</style><div class="wrap">', 1),
                    unsafe_allow_html=True)
    explain("bat3d")


def battery_info(c: str, fs: pd.DataFrame, groups: pd.DataFrame) -> Dict[str, Any]:
    m = meta.loc[c]
    row = fs.loc[c] if c in fs.index else None
    soh = float(row["SOH"]) if row is not None else float(ct[(ct["Cell_ID"] == c) & ~ct["outlier"]]["SOH"].tail(5).median())
    risk = str(row["Risk"]) if row is not None else "Healthy"
    grp = groups.loc[c, "Group"] if c in groups.index else "—"
    cond = [f"Ambient {m['Ambient_C']:.0f} °C", f"Discharge {m['I_dis_A']:.1f} A", f"Cut-off {m['V_cut_V']:.2f} V",
            f"Group: {grp}"]
    hover = (f"<b>{c}</b><br>SOH {100 * soh:.1f}% · {risk}<br>{m['Ambient_C']:.0f} °C · {m['I_dis_A']:.1f} A · "
             f"{int(m['cycles'])} cycles")
    return {"soh": soh, "risk": risk, "conditions": cond, "hover": hover, "group": grp, "row": row}


def _grade(row: Optional[pd.Series]) -> Tuple[str, str]:
    if row is None:
        return "–", "#8C8C8C"
    margin, risk = float(row["Health margin"]), str(row["Risk"])
    if risk == "Critical":
        return "E", RISK_COLORS["Critical"]
    if risk == "Warning":
        return "D", RISK_COLORS["Warning"]
    if risk == "Watch":
        return "C", RISK_COLORS["Watch"]
    return ("A", RISK_COLORS["Healthy"]) if margin > 0.5 else ("B", "#56B4E9")


def resume_card(c: str, inf: Dict[str, Any], ev: Optional[pd.DataFrame] = None) -> None:
    """General performance overview of one battery: grade, health, life, speed of ageing, stress, events, data."""
    m = meta.loc[c]
    row = inf["row"]
    d = ct[ct["Cell_ID"] == c].sort_values("n")
    good = d[~d["outlier"]]

    def bar(label: str, frac: float, color: str, text: str) -> str:
        w = 100 * float(np.clip(frac, 0, 1))
        return (f'<div class="bt-res-row"><div class="bt-res-l">{label}</div><div class="bt-res-v">{text}</div>'
                f'<div class="bt-res-bar"><div style="width:{w:.0f}%;background:{color}"></div></div></div>')

    def tile(label: str, value: str) -> str:
        return f'<div class="bt-tile"><div class="bt-tile-v">{value}</div><div class="bt-tile-l">{label}</div></div>'

    soh = inf["soh"]
    col = RISK_COLORS.get(inf["risk"], "#009E73")
    grade, gcol = _grade(row)
    rul = float(row["Quick RUL"]) if row is not None else float("nan")
    rg = float(row["R growth (%)"]) if row is not None and np.isfinite(row["R growth (%)"]) else 0.0
    peak = float(row["Peak T (°C)"]) if row is not None else float("nan")
    fade = float(row["Fade per 100 cycles (%)"]) if row is not None else float("nan")
    thr = float(good["cum_Ah"].max() - good["cum_Ah"].min()) if len(good) else float("nan")
    eff = float(good["eff_energy"].median()) if "eff_energy" in good and good["eff_energy"].notna().any() else float("nan")
    tmin = float(good["T_mean_C"].min()) if len(good) else float("nan")
    n_regen = int(good["regen"].sum()) if "regen" in good else 0
    n_out = int(d["outlier"].sum())
    k = knee_cached(d, DATA_KEY, c)
    evc = ev[ev["Cell_ID"] == c]["Severity"].value_counts() if ev is not None and len(ev) else pd.Series(dtype=int)
    n_ev = int(evc.sum())
    sev_chips = "".join(f'<span class="bt-sev-chip" style="border-color:{RISK_COLORS.get(sv, "#56B4E9")}">'
                        f'{int(evc.get(sv, 0))} {sv.lower()}</span>' for sv in ("Critical", "Warning", "Watch", "Info"))
    life_txt = ("past end of life" if np.isfinite(rul) and rul == 0 else f"{rul:.0f} cycles left" if np.isfinite(rul)
                else "not declining")
    summary = (f"{c} has delivered {thr:.0f} Ah over {int(m['cycles'])} cycles and keeps {100 * soh:.0f}% of its "
               f"capacity ({life_txt}). " if np.isfinite(thr) else "")
    summary += (f"Fade accelerated at cycle {k['knee_n']}. " if k.get("found") else "") + \
               (f"{n_ev} logged events." if n_ev else "No logged events.")
    rows = [bar("State of health", soh, col, f"{100 * soh:.1f}%"),
            bar("Capacity now / new", soh, "#0072B2", f"{m['C_bol_Ah'] * soh:.3f} / {m['C_bol_Ah']:.3f} Ah"),
            bar("Quick remaining life", min(rul / 200, 1) if np.isfinite(rul) else 1, col,
                f"{rul:.0f} cycles" if np.isfinite(rul) else "not declining"),
            bar("Fade speed", min(fade / 10, 1) if np.isfinite(fade) else 0, "#E69F00",
                f"{fade:.2f} % / 100 cycles" if np.isfinite(fade) else "—"),
            bar("Resistance growth", min(max(rg, 0) / 150, 1), "#CC79A7", f"{rg:+.0f}%"),
            bar("Peak temperature", min(peak / 70, 1) if np.isfinite(peak) else 0, "#D55E00",
                f"{peak:.1f} °C" if np.isfinite(peak) else "—")]
    tiles = [tile("Cycles", f"{int(m['cycles'])}"), tile("Throughput", f"{thr:.0f} Ah" if np.isfinite(thr) else "—"),
             tile("Energy efficiency", f"{100 * eff:.1f}%" if np.isfinite(eff) else "—"),
             tile("Temperature range", f"{tmin:.0f}–{peak:.0f} °C" if np.isfinite(tmin) and np.isfinite(peak) else "—"),
             tile("Knee point", f"n = {k['knee_n']}" if k.get("found") else "none"),
             tile("Regeneration events", f"{n_regen}"), tile("Logged events", f"{n_ev}"),
             tile("Excluded cycles", f"{n_out}")]
    alerts = str(row["Alerts"]) if row is not None else "—"
    st.markdown(
        f'<div class="bt-resume"><div class="bt-res-head"><div class="bt-grade" style="border-color:{gcol};color:{gcol}">'
        f'{grade}</div><div><div class="bt-res-top"><span class="bt-res-badge" style="background:{col}">'
        f'{html.escape(inf["risk"].upper())}</span><span class="bt-res-name">{html.escape(c)}</span></div>'
        f'<div class="bt-res-sub">{html.escape(" · ".join(inf["conditions"]))}</div></div></div>'
        f'<div class="bt-res-sum">{html.escape(summary)}</div>'
        + "".join(rows) + f'<div class="bt-tiles">{"".join(tiles)}</div>'
        f'<div class="bt-res-ev"><b>Event log:</b> {n_ev} events {sev_chips}</div>'
        f'<div class="bt-res-alerts"><b>Alerts:</b> {html.escape(alerts)}</div></div>',
        unsafe_allow_html=True)
    note("General performance overview: grade A (healthy, large margin) to E (critical), health and remaining "
         "life, how fast it ages, the stress it has seen, how many events were logged and how many cycles were "
         "excluded as invalid. Details follow in the sections below.")


def battery_hero(scope: str, sel: Sequence[str], fs: pd.DataFrame, ev: Optional[pd.DataFrame] = None) -> None:
    groups = te.condition_groups(ct)
    if scope == "single":
        inf = battery_info(sel[0], fs, groups)
        m = meta.loc[sel[0]]
        a, b = st.columns([2, 3], vertical_alignment="center")
        with a:
            render_battery_anim([{"id": sel[0], "soh": inf["soh"], "risk": inf["risk"],
                                  "sub": f"{m['Ambient_C']:.0f} °C · {m['I_dis_A']:.1f} A · {inf['group']}"}], big=True)
        with b:
            resume_card(sel[0], inf, ev)
    else:
        cells = list(sel)[:8] if scope == "selected" else list(fs.index[:8])
        items = []
        for c in cells:
            inf = battery_info(c, fs, groups)
            n_ev = int((ev["Cell_ID"] == c).sum()) if ev is not None and len(ev) else 0
            items.append({"id": c, "soh": inf["soh"], "risk": inf["risk"],
                          "sub": f"{meta.loc[c, 'Ambient_C']:.0f} °C · {meta.loc[c, 'I_dis_A']:.1f} A · {n_ev} events"})
        st.markdown("**Battery rack: " + ("selected batteries" if scope == "selected" else "the 8 highest-risk batteries")
                    + "**")
        render_battery_anim(items, big=False)


# =============================================================================
# Cached data access & computation
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_store(path: str, mtime: float) -> te.ParquetStore:
    return te.ParquetStore(path)


@st.cache_resource(show_spinner=False)
def demo_data() -> Tuple[te.ParquetStore, pd.DataFrame]:
    """Synthetic cohort with known ground truth (offline demo / reviewer mode)."""
    master, imp, _ = te.make_synthetic_master(
        n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24),
        currents=(2, 2, 2, 2, 1, 4, 4, 2), k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    return te.ParquetStore.from_dataframe(master), imp


@st.cache_resource(show_spinner=False)
def persist_upload(name: str, size: int, _raw: bytes) -> str:
    return str(te.persist_upload(_raw, name))


@st.cache_data(show_spinner=False)
def load_impedance_path(path: str, mtime: float) -> pd.DataFrame:
    return te.load_impedance(path)


@st.cache_data(show_spinner=False, max_entries=16)
def cycle_table(_store: te.ParquetStore, store_key: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    return te.build_cycle_table_with_report(_store)


@st.cache_resource(show_spinner=False, max_entries=3)
def cell_frame(_store: te.ParquetStore, store_key: str, cell: str) -> pd.DataFrame:
    return _store.cell_frame(cell)


@st.cache_resource(show_spinner=False, max_entries=3)
def prepared_cell(_store: te.ParquetStore, store_key: str, cell: str) -> pd.DataFrame:
    return te.prepare_cell(cell_frame(_store, store_key, cell))


@st.cache_data(show_spinner=False, max_entries=8)
def validation_cached(_store: te.ParquetStore, store_key: str, cell: str) -> List[str]:
    return te.validate_master(cell_frame(_store, store_key, cell))


@st.cache_data(show_spinner=False, max_entries=32)
def ica_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, key: str, cell: str,
               n_curves: int, ir: bool, dv: float, window: int) -> List[te.ICACurve]:
    return te.ica_evolution(_prep, _ct_cell, n_curves, ir_compensate=ir, dv=dv, smooth_window=window)


@st.cache_data(show_spinner=False, max_entries=64)
def ml_cached(_ct: pd.DataFrame, key: str, cell: str, n0: int, model: str, use_pop: bool, eol_ah: float,
              strategy: str, conformal_cells: int, level: float, params_json: str = "{}",
              train_cells: Optional[Tuple[str, ...]] = None) -> te.MLForecast:
    return te.train_ml_forecast(_ct, cell, n0, model, use_population=use_pop, eol_ah=eol_ah, strategy=strategy,
                                conformal_cells=conformal_cells, band_level=level,
                                model_params=json.loads(params_json),
                                train_cells=list(train_cells) if train_cells is not None else None)


@st.cache_data(show_spinner=False, max_entries=64)
def est_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, model: str, params_json: str,
               features: Tuple[str, ...], split: str, test_frac: float, train_cells: Tuple[str, ...],
               test_cells: Tuple[str, ...], normalise: bool) -> te.EstimationResult:
    return te.train_soh_estimator(_ct, _imp, model, features, json.loads(params_json), split, test_frac,
                                  list(train_cells) or None, list(test_cells) or None, normalise)



# ---- Result backup on disk (v5.27): heavy results survive new sessions; export as a zip, commit to results/cache/ ----
CACHE_DIR = RESULTS_DIR / "cache"


def _fingerprint(h: Any, obj: Any) -> None:
    import hashlib  # noqa: F401
    if isinstance(obj, pd.DataFrame):
        h.update(("DF" + "|".join(map(str, obj.columns))).encode())
        h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
    elif isinstance(obj, pd.Series):
        h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(obj.tobytes())
    elif isinstance(obj, dict):
        for k_ in sorted(obj, key=str):
            h.update(str(k_).encode()); _fingerprint(h, obj[k_])
    elif isinstance(obj, (list, tuple)):
        h.update(b"(")
        for v_ in obj:
            _fingerprint(h, v_)
        h.update(b")")
    elif obj is None or isinstance(obj, (str, int, float, bool, np.integer, np.floating)):
        h.update(repr(obj).encode())
    else:
        import pickle
        try:
            h.update(pickle.dumps(obj, protocol=4))
        except Exception:
            h.update(repr(obj).encode())


def backed(name: str, compute: Callable[[], Any], *parts: Any) -> Any:
    """Load the result of `compute` from the disk backup if the same engine version already computed it from the same
    inputs (content, not paths); otherwise compute it and save it."""
    import hashlib
    import pickle
    h = hashlib.sha1()
    cv = getattr(te, "CACHE_VERSION", te.ENGINE_VERSION)   # results stay valid across releases that add features
    h.update(f"{name}|{cv}".encode())
    for p_ in parts:
        _fingerprint(h, p_)
    path = CACHE_DIR / f"{name}__{cv}__{h.hexdigest()[:20]}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f_:
                return pickle.load(f_)
        except Exception:
            pass
    val = compute()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f_:
            pickle.dump(val, f_, protocol=4)
        tmp.replace(path)
    except Exception:
        pass
    return val


def cache_backup_panel() -> None:
    files = sorted(CACHE_DIR.glob(f"*__{getattr(te, 'CACHE_VERSION', te.ENGINE_VERSION)}__*.pkl")) if CACHE_DIR.exists() else []
    with st.expander(f"Result backup · {len(files)} saved result(s)", icon=":material/backup:"):
        st.caption("Heavy results (live replays, frozen models, anomaly scan, shared physics, hybrid) are saved here and "
                   "reload instantly. To keep them after the app restarts, download the zip and upload its files to "
                   "the folder results/cache/ of the GitHub repository. A new app version rebuilds them.")
        if files:
            import io
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f_ in files:
                    z.write(f_, arcname=f_.name)
            st.download_button("Download the result backup (zip)", buf.getvalue(),
                               file_name=f"result_cache__v{te.ENGINE_VERSION}.zip", mime="application/zip",
                               key="cache_zip", use_container_width=True)
            st.caption(f"{sum(f_.stat().st_size for f_ in files) / 1e6:.1f} MB")


@st.cache_data(show_spinner=False, max_entries=8)
def replay_cached(_cell_df: pd.DataFrame, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str,
                  soh_eol: float, level: float, cap_every: int = 10) -> Tuple[te.EKFResult, pd.DataFrame]:
    return backed("replay", lambda: _replay_compute(_cell_df, _ct, _imp, cell, soh_eol, level, cap_every),
                  _cell_df, _ct[_ct["Cell_ID"] == cell], _imp, cell, soh_eol, level, cap_every)


def _replay_compute(_cell_df: pd.DataFrame, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], cell: str,
                    soh_eol: float, level: float, cap_every: int) -> Tuple[te.EKFResult, pd.DataFrame]:
    r = te.run_dual_twin(_cell_df, _ct, _imp, cell, te.TwinParameters(),
                         te.twin_config_for(_ct, cell, te.DualTwinConfig(capacity_every=cap_every)))
    n_max = int(r.per_cycle["n"].max() * 1.6)
    rows = []
    for n in r.per_cycle["n"].astype(int):
        if n < 5:
            continue
        f = te.twin_forecast(r, int(n), n_max, soh_eol, level)
        rs = f.rul_samples[np.isfinite(f.rul_samples)] if f.rul_samples is not None else np.array([])
        if len(rs):
            rows.append({"n": int(n), "eol_med": n + float(np.median(rs)), "eol_lo": n + float(np.quantile(rs, 0.05)),
                         "eol_hi": n + float(np.quantile(rs, 0.95)), "frac_censored": 1 - len(rs) / len(f.rul_samples)})
    return r, pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=8)
def live_cached(_ekf: te.EKFResult, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str,
                soh_eol: float, level: float, cap_every: int, models: Tuple[str, ...],
                frozen_cfg: Optional[Tuple[Any, ...]] = None, _extra: Optional[Dict[str, Any]] = None
                ) -> Tuple[Dict[int, Any], pd.DataFrame]:
    return backed("live", lambda: te.live_multi_model(_ct, cell, _ekf, soh_eol, level, models, imp=_imp, extra=_extra),
                  _ct, _imp, cell, _ekf.per_cycle if hasattr(_ekf, "per_cycle") else _ekf, soh_eol, level, cap_every,
                  models, frozen_cfg, _extra)


@st.cache_data(show_spinner=False, max_entries=4096)
def ga_fitness_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str, train: Tuple[str, ...],
                      n0: int, eol_ah: float, genome: Tuple[Tuple[str, Any], ...]) -> Dict[str, float]:
    """One GA-PINN evaluation (nested leave-one-out fitness), saved on disk: a stopped search resumes from here."""
    return backed("gapinn", lambda: te.ga_pinn_fitness(_ct, _imp, cell, n0, list(train), eol_ah, dict(genome)),
                  _ct, _imp, cell, train, n0, eol_ah, genome)


@st.cache_data(show_spinner=False, max_entries=16)
def frozen_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str, train: Tuple[str, ...],
                  fkey: str, mkey: str, soh_eol: float, refit: int, anchor: str, eol_ah: float,
                  n_start: Optional[int] = None, n_max: Optional[int] = None) -> Tuple[Dict[int, Any], pd.DataFrame]:
    """n_start / n_max: the live forecast window (te.frozen_live_window); None = the assessment's (30 % of the
    battery's own record), used by the Live twin."""
    return backed("frozen", lambda: te.frozen_stream(_ct, _imp, cell, list(train), fkey, mkey, soh_eol, refit, anchor,
                                                     n_start=n_start, n_max=n_max, eol_ah=eol_ah),
                  _ct, _imp, cell, train, fkey, mkey, soh_eol, refit, anchor, eol_ah, n_start, n_max)


@st.cache_data(show_spinner=False, max_entries=64)
def frozen_study_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str, train: Tuple[str, ...],
                        group: str, soh_eol: float, intervals: Tuple[int, ...], anchors: Tuple[str, ...], horizon: int,
                        eol_ah: float) -> pd.DataFrame:
    """One battery's frozen-model update study, cached as soon as it finishes (a group run resumes where it stopped)."""
    return te.frozen_update_study(_ct, _imp, cell, list(train), group, soh_eol, intervals, anchors, horizon, eol_ah)


def model_label(k: Optional[str]) -> str:
    """Display name of a protocol model key; GA-PINN winners carry their genes in the key (not in the registry)."""
    if not k:
        return "—"
    if k in te.PROTOCOL_MODELS:
        return te.PROTOCOL_MODELS[k][1]
    if k == "pinn:GA-PINN":
        return "GA-PINN (each battery's GA winner)"
    if "PINN-self|" in k:
        return f"Self-tuning GA-PINN · {len(te.pinn_self_candidates(k))} candidates, chosen on its own last " \
               f"{te.SELF_TUNE_WINDOW} cycles at each retraining"
    if "PINN-ga|" in k:
        g = te.ga_pinn_genome(k)
        return (f"GA-PINN · h{g['hidden']}, λ {g['physics_weight']:g}, SEI p {g['sei_p']:g}, "
                f"acceleration {'on' if g['accel'] else 'off'}")
    return k


TWIN_CORES = ("Frozen data model (Mission 2)", "PINN twin core", "GA-PINN core")
# the self-tuning GA-PINN core (STATUS 5bz) gave results identical to GA-PINN on NASA (5cc): removed from the choices,
# its engine code (pinn:PINN-self) is kept and documented
PINN_CORES = TWIN_CORES[1:]


def twin_core() -> str:
    return st.session_state.get("twin_core", TWIN_CORES[2])      # GA-PINN: best twin core (STATUS 5cc)


def ga_loaded() -> bool:
    return any(r.get("done") for r in (st.session_state.get("ga_pinn") or {}).values())


def core_refit(core: Optional[str] = None) -> int:
    """Retrain interval of the twin's core (0 = trained once at the live start). Only the PINN core uses it: its
    battery code is learned from the battery's own cycles, so retraining lets it see a fade that speeds up (B0005,
    STATUS 5bs); the frozen data model stays as validated in Mission 2 (trained once, re-anchored)."""
    return int(st.session_state.get("pinn_refit", 20)) if (core or twin_core()) in PINN_CORES else 0


def self_tuning_key(c: str, k: int = 3) -> Optional[str]:
    """Self-tuning core of battery c (STATUS 5bz): its k best nested-GA genomes + the default settings; at every
    retraining the twin picks the one that best forecast the battery's own recent cycles (causal backtest)."""
    r = (st.session_state.get("ga_pinn") or {}).get(c) or {}
    if not r.get("done") or not r.get("fitness"):
        return None
    fit = r["fitness"]
    cands = [dict(r["genomes"][g]) for g in sorted(fit, key=fit.get)[:k]]
    start = dict(te.GA_PINN_START)
    if all(te.ga_pinn_key(x) != te.ga_pinn_key(start) for x in cands):
        cands.append(start)
    return te.pinn_self_key(cands)


def ga_winner_key(c: str) -> Optional[str]:
    """Protocol key of battery c's GA-PINN winner (GA-PINN section, Integrated twin page), if its search finished."""
    r = (st.session_state.get("ga_pinn") or {}).get(c) or {}
    return r.get("best_key") if r.get("done") else None


def frozen_context(c: str, core: Optional[str] = None) -> Tuple[Optional[str], Optional[Tuple[str, str]], List[str]]:
    """Group of a battery (recorded conditions), its frozen models, and the group's other batteries. core: the twin's
    forecaster (sidebar "Twin core model"); the PINN core replaces the group's frozen forecaster, trained the same way
    (once, at the live start) and re-anchored every cycle (STATUS 5bq)."""
    groups = te.condition_groups(CT_RAW)
    grp_c = str(groups.loc[c, "Group"]) if c in groups.index else None
    fz = te.frozen_models_for(grp_c)
    core_ = core or twin_core()
    if fz and core_ == TWIN_CORES[1]:
        fz = (te.PINN_TWIN_KEY, fz[1])
    elif fz and core_ == TWIN_CORES[2]:                    # GA-PINN winner; the default PINN core until its search ends
        fz = (ga_winner_key(c) or te.PINN_TWIN_KEY, fz[1])
    train_c = [x for x in te.protocol_cells(CT_RAW, grp_c) if x != c] if fz else []
    return grp_c, fz, train_c


def frozen_unavailable_reason(c: str, grp_c: Optional[str], fz: Optional[Tuple[str, str]], train_c: List[str]) -> str:
    """Why the frozen model cannot run for this battery, in plain words (empty string when it can)."""
    if grp_c is None:
        return f"{c} has no condition group (no valid cycle), so there is no frozen model for it."
    if fz is None:
        return (f"{c} belongs to the {grp_c} group, which was not part of the accuracy assessment, so it has no frozen "
                f"model. Frozen models exist for: {', '.join(te.FROZEN_MODELS)}.")
    if len(train_c) < 2:
        return (f"{c} belongs to the {grp_c} group, whose frozen model needs at least two other batteries to retrain "
                f"on; this dataset has {len(train_c)} other usable {grp_c} battery(ies) (25 cycles or more).")
    return ""




@st.cache_data(show_spinner=False, max_entries=16)
def half_cell_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, key: str, cell: str, n_curves: int,
                     ir: bool) -> Tuple[pd.DataFrame, List[te.HalfCellFit]]:
    return te.half_cell_trajectory(_prep, _ct_cell, n_curves, ir)


@st.cache_data(show_spinner=False, max_entries=8)
def early_life_cached(_ct: pd.DataFrame, key: str, eol_ah: float, n_a: int, n_b: int) -> Dict[str, Any]:
    return te.early_life_lifetime(_ct, eol_ah, n_a, n_b)


@st.cache_data(show_spinner=False, max_entries=4)
def update_cohort_cached(_store: Any, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, eol_ah: float,
                         cells: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for c in cells:
        g = _ct[(_ct["Cell_ID"] == c) & ~_ct["outlier"]]
        soh_eol_c = te.soh_eol_for(float(g["C_bol_Ah"].iloc[0]), eol_ah)
        n0 = int(max(5, round(0.4 * g["n"].max())))
        try:
            tab, _ = te.update_frequency_study(_store.cell_frame(c), _ct, _imp, c, te.TwinParameters(),
                                               te.twin_config_for(_ct, c, te.DualTwinConfig()), n0, soh_eol_c,
                                               intervals=(1, 5, 20))
            rows.append({"Cell_ID": c, "recommended": tab.attrs.get("recommended"),
                         **{f"RMSE every {m}": float(tab.loc[m, "Tracking RMSE"]) for m in tab.index}})
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def ml_v2_cached(_ct: pd.DataFrame, key: str, cell: str, n0: int, model: str, eol_ah: float, level: float,
                 train_cells: Optional[Tuple[str, ...]], params_json: str, baseline: str = "shape") -> Any:
    tc = None if train_cells is None else list(train_cells)
    return te.train_ml_v2(_ct, cell, n0, model, eol_ah, tc, json.loads(params_json) or None, level, baseline=baseline)


@st.cache_data(show_spinner=False, max_entries=16)
def calibrated_plant_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str
                            ) -> Tuple[te.CellPhysics, pd.DataFrame]:
    return te.calibrate_plant(_ct, _imp, cell)


@st.cache_data(show_spinner=False, max_entries=8)
def fleet_cached(_ct: pd.DataFrame, key: str, eol_ah: float, limits: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    L = te.SafetyLimits(**limits)
    return te.fleet_status(_ct, eol_ah, L), te.fleet_events(_ct, L)


@st.cache_data(show_spinner=False, max_entries=4)
def hi_rank_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str) -> pd.DataFrame:
    return te.rank_health_indicators(_ct, _imp)


@st.cache_data(show_spinner=False, max_entries=4)
def hi_pca_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str) -> Dict[str, Any]:
    return te.hi_pca(_ct, _imp)


@st.cache_data(show_spinner=False, max_entries=4)
def stress_cached(_ct: pd.DataFrame, key: str, limits: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    L = te.SafetyLimits(**limits)
    return te.stress_exposure(_ct, L), te.stress_summary(_ct, L)


@st.cache_data(show_spinner=False, max_entries=4)
def stress_factors_cached(_ct: pd.DataFrame, key: str) -> Dict[str, Any]:
    return te.stress_factor_regression(_ct)


@st.cache_data(show_spinner=False, max_entries=32)
def knee_cached(_ct_cell: pd.DataFrame, key: str, cell: str) -> Dict[str, Any]:
    g = _ct_cell[~_ct_cell["outlier"]]
    return te.detect_knee(g["n"].to_numpy(), g["SOH"].to_numpy())


@st.cache_data(show_spinner=False, max_entries=16)
def modes_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, _eis: pd.DataFrame, key: str, cell: str,
                 n_curves: int) -> Tuple[List[te.DVACurve], pd.DataFrame]:
    dva = te.dva_evolution(_prep, _ct_cell, n_curves)
    ica = te.ica_evolution(_prep, _ct_cell, n_curves)
    return dva, te.degradation_modes(ica, _ct_cell, _eis)








@st.cache_data(show_spinner=False)
def load_bench_file(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def read_uploaded_table(up: Any) -> pd.DataFrame:
    raw = up.getvalue()
    import io
    return pd.read_parquet(io.BytesIO(raw)) if up.name.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))


def resolve_sources(up_master: Any, up_imp: Any, master_url: str, imp_url: str, master_sha: str
                    ) -> Tuple[Optional[te.ParquetStore], Optional[pd.DataFrame], List[str]]:
    notes: List[str] = []
    store: Optional[te.ParquetStore] = None
    imp: Optional[pd.DataFrame] = None

    master_path: Optional[Path] = None
    if up_master is not None:
        master_path = Path(persist_upload(up_master.name, up_master.size, up_master.getvalue()))
        notes.append(f"Telemetry: uploaded {up_master.name}")
    else:
        for cand in (DATA_DIR / MASTER_NAME, APP_DIR / MASTER_NAME):
            if cand.exists():
                master_path = cand
                notes.append("Telemetry: local file")
                break
    if master_path is None and master_url:
        bar = st.sidebar.progress(0.0, text="Downloading telemetry…")
        master_path = te.download_to_cache(master_url, progress=lambda f, m: bar.progress(f, text=f"Telemetry {m}"),
                                           sha256=master_sha or None)
        bar.empty()
        notes.append("Telemetry: remote URL (cached" + (", SHA-256 verified)" if master_sha else ")"))
    if master_path is not None:
        store = get_store(str(master_path), master_path.stat().st_mtime)
    elif st.session_state.get("demo"):
        store, imp = demo_data()
        notes.append("Telemetry: synthetic demo cohort (known ground truth)")

    if up_imp is not None:
        imp = te.load_impedance(up_imp.getvalue())
        notes.append(f"EIS: uploaded {up_imp.name}")
    elif imp is None:
        imp_path = next((p for p in (DATA_DIR / IMP_NAME, APP_DIR / IMP_NAME) if p.exists()), None)
        if imp_path is None and imp_url:
            imp_path = te.download_to_cache(imp_url)
        if imp_path is not None:
            imp = load_impedance_path(str(imp_path), imp_path.stat().st_mtime)
            notes.append("EIS: loaded")
    if imp is None:
        notes.append("EIS: not available")
    return store, imp, notes


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### Data sources")
    up_master = st.file_uploader("Master telemetry (.parquet)", type=["parquet"], key="up_master")
    up_imp = st.file_uploader("EIS ground truth (.parquet)", type=["parquet"], key="up_imp")
    # files uploaded on the start screen: that widget disappears once data load, so keep the files in the session
    for _w, _keep in (("up_master_main", "_kept_master"), ("up_imp_main", "_kept_imp")):
        if st.session_state.get(_w) is not None:
            st.session_state[_keep] = st.session_state[_w]
    up_master = up_master or st.session_state.get("_kept_master")
    up_imp = up_imp or st.session_state.get("_kept_imp")
    if st.session_state.get("_kept_master") is not None and st.button("Forget the uploaded files",
                                                                        icon=":material/delete:"):
        for _k in ("_kept_master", "_kept_imp"):
            st.session_state.pop(_k, None)
        st.rerun()
    with st.expander("Remote URLs", expanded=False):
        master_url = st.text_input("Telemetry URL", value=_secret("MASTER_PARQUET_URL"))
        master_sha = st.text_input("Telemetry SHA-256 pin (optional)", value=_secret("MASTER_PARQUET_SHA256"),
                                   help="Full-file digest; the download is rejected if it differs.")
        imp_url = st.text_input("EIS URL", value=_secret("IMPEDANCE_PARQUET_URL"))
    st.markdown("### Twin core model")
    st.radio("Forecaster inside the twin", TWIN_CORES, key="twin_core", index=2,
             help="Used by the Live twin, the Integrated twin and Mission 3. The PINN core is trained once at the live "
                  "start and re-anchored every cycle, like the frozen model; compare both with the twin scorecard "
                  "(Integrated twin page). The Model updating page always uses the frozen model (Mission 2).")
    st.selectbox("PINN core: retrain every N cycles", [0, 10, 20, 30], key="pinn_refit", index=2,
                 format_func=lambda v: "never (trained once at the live start)" if v == 0 else f"every {v} cycles",
                 help="Only for the PINN cores (default every 20 cycles: STATUS 5bt, B0005 45 -> 25 cycles). Retraining lets "
                      "the battery code follow a fade that speeds up; each "
                      "retraining costs one PINN training (minutes on NASA the first time, then saved).")
    if st.session_state.get("demo"):
        if st.button("Leave synthetic demo"):
            st.session_state["demo"] = False
            st.rerun()

    st.markdown("### Appearance")
    theme_choice = st.radio("Figure palette", ["Auto", "Light", "Dark"], horizontal=True,
                            help="Auto follows the active Streamlit theme (Settings → Theme).")
    publication = st.toggle("Publication style", value=False,
                            help="White background, serif type and black axes, for export into papers.")
    debug = st.toggle("Show tracebacks on errors", value=False)
    cache_backup_panel()

P = resolve_palette(theme_choice, publication)
inject_css(P)

# =============================================================================
# Hero
# =============================================================================
HERO_CSS = """<style>
.bt-hero {display: flex; align-items: center; gap: 28px; padding: 22px 30px 20px 30px;}
.bt-hero-text {position: relative; flex: 1 1 auto; min-width: 0;}
.bt-hero .bt-title {font-size: 1.85rem;}
.bt-hero .bt-sub {max-width: 60ch; font-size: 1.02rem;}
.bt-art {position: relative; flex: 0 0 460px; width: 460px; height: 210px;}
.bt-art svg {position: absolute; inset: 0; overflow: visible;}
.bt-pkt {position: absolute; left: 0; top: 0; width: 9px; height: 9px; border-radius: 50%;
  offset-anchor: 50% 50%; offset-rotate: 0deg; animation: bt-move 2.8s linear infinite; opacity: 0;}
.bt-pkt-up {offset-path: path('M 118 72 C 190 14, 270 14, 342 72'); background: #7FE3FF;
  box-shadow: 0 0 10px 2px rgba(127,227,255,0.9);}
.bt-pkt-down {offset-path: path('M 342 150 C 270 208, 190 208, 118 150'); background: #FFD166;
  box-shadow: 0 0 10px 2px rgba(255,209,102,0.9);}
@keyframes bt-move {0% {offset-distance: 0%; opacity: 0;} 12% {opacity: 1;} 88% {opacity: 1;}
  100% {offset-distance: 100%; opacity: 0;}}
.bt-level {transform-box: fill-box; transform-origin: 50% 100%; animation: bt-charge 7s ease-in-out infinite;}
@keyframes bt-charge {0%, 100% {transform: scaleY(1);} 50% {transform: scaleY(0.3);}}
.bt-draw {stroke-dasharray: 240; stroke-dashoffset: 240; animation: bt-draw 7s ease-in-out infinite;}
@keyframes bt-draw {0% {stroke-dashoffset: 240;} 55%, 90% {stroke-dashoffset: 0;} 100% {stroke-dashoffset: 0; opacity: 0;}}
.bt-ring {transform-box: fill-box; transform-origin: center; animation: bt-ring 2.2s ease-out infinite;}
@keyframes bt-ring {0% {transform: scale(1); opacity: 0.9;} 100% {transform: scale(2.0); opacity: 0;}}
.bt-scan {animation: bt-scan 3.5s ease-in-out infinite;}
@keyframes bt-scan {0%, 100% {transform: translateY(0); opacity: 0.0;} 10% {opacity: 0.8;} 50% {transform: translateY(118px); opacity: 0.8;} 60% {opacity: 0;}}
.bt-glow {animation: bt-glow 3s ease-in-out infinite;}
@keyframes bt-glow {0%, 100% {opacity: 0.35;} 50% {opacity: 0.9;}}
@media (max-width: 1150px) {.bt-art {display: none;}}
@media (prefers-reduced-motion: reduce) {.bt-pkt, .bt-level, .bt-draw, .bt-ring, .bt-scan, .bt-glow {animation: none;}
  .bt-pkt {opacity: 0;} .bt-draw {stroke-dashoffset: 0;}}
</style>"""

HERO_ART = """<div class="bt-art" aria-hidden="true">
<svg width="460" height="210" viewBox="0 0 460 210" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <linearGradient id="btMetal" x1="0" x2="1" y1="0" y2="0">
   <stop offset="0" stop-color="#8FA6BE"/><stop offset="0.45" stop-color="#EEF5FB"/><stop offset="1" stop-color="#6F869F"/>
  </linearGradient>
  <linearGradient id="btCharge" x1="0" x2="0" y1="0" y2="1">
   <stop offset="0" stop-color="#9BF6C9"/><stop offset="1" stop-color="#1FB57A"/>
  </linearGradient>
 </defs>
 <!-- data links -->
 <path d="M 118 72 C 190 14, 270 14, 342 72" fill="none" stroke="rgba(127,227,255,0.55)" stroke-width="1.6" stroke-dasharray="4 5"/>
 <path d="M 342 150 C 270 208, 190 208, 118 150" fill="none" stroke="rgba(255,209,102,0.55)" stroke-width="1.6" stroke-dasharray="4 5"/>
 <text x="230" y="22" text-anchor="middle" font-size="10.5" letter-spacing="0.2" fill="rgba(255,255,255,0.92)" font-weight="700">Voltage, current, temperature</text>
 <text x="230" y="206" text-anchor="middle" font-size="10.5" letter-spacing="0.2" fill="rgba(255,255,255,0.92)" font-weight="700">Health, remaining life, control</text>
 <!-- update engine -->
 <circle class="bt-ring" cx="230" cy="111" r="24" fill="none" stroke="#7FE3FF" stroke-width="2"/>
 <circle cx="230" cy="111" r="24" fill="rgba(14,23,38,0.55)" stroke="rgba(255,255,255,0.75)" stroke-width="1.4"/>
 <text x="230" y="108" text-anchor="middle" font-size="11" font-weight="800" fill="#FFFFFF">EKF</text>
 <text x="230" y="121" text-anchor="middle" font-size="8" fill="rgba(255,255,255,0.8)">update</text>
 <line x1="206" y1="111" x2="150" y2="111" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
 <line x1="254" y1="111" x2="316" y2="111" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
 <!-- physical 18650 cell -->
 <ellipse class="bt-glow" cx="70" cy="112" rx="58" ry="82" fill="rgba(255,209,102,0.10)"/>
 <rect x="58" y="30" width="24" height="12" rx="3" fill="#DDE7F1" stroke="rgba(255,255,255,0.8)"/>
 <rect x="30" y="40" width="80" height="145" rx="14" fill="url(#btMetal)" stroke="rgba(255,255,255,0.85)" stroke-width="1.5"/>
 <rect x="42" y="56" width="56" height="113" rx="7" fill="rgba(11,61,145,0.45)"/>
 <rect class="bt-level" x="42" y="56" width="56" height="113" rx="7" fill="url(#btCharge)"/>
 <text x="70" y="118" text-anchor="middle" font-size="15" font-weight="900" fill="rgba(11,40,80,0.75)">+</text>
 <text x="70" y="200" text-anchor="middle" font-size="9.5" letter-spacing="0.2" font-weight="800" fill="#FFFFFF">Physical cell</text>
 <!-- digital twin -->
 <rect x="362" y="30" width="24" height="12" rx="3" fill="none" stroke="#7FE3FF" stroke-width="1.4" stroke-dasharray="3 3"/>
 <rect x="350" y="40" width="80" height="145" rx="14" fill="rgba(127,227,255,0.10)" stroke="#7FE3FF" stroke-width="1.6" stroke-dasharray="6 4"/>
 <g stroke="rgba(127,227,255,0.22)" stroke-width="1">
  <line x1="360" y1="70" x2="420" y2="70"/><line x1="360" y1="100" x2="420" y2="100"/>
  <line x1="360" y1="130" x2="420" y2="130"/><line x1="360" y1="160" x2="420" y2="160"/>
  <line x1="375" y1="55" x2="375" y2="172"/><line x1="395" y1="55" x2="395" y2="172"/><line x1="415" y1="55" x2="415" y2="172"/>
 </g>
 <polyline class="bt-draw" points="360,64 372,67 384,72 396,80 405,90 412,104 417,122 421,146" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/>
 <line x1="360" y1="140" x2="422" y2="140" stroke="#FF8A4C" stroke-width="1.4" stroke-dasharray="3 3"/>
 <rect class="bt-scan" x="352" y="50" width="76" height="3" rx="1.5" fill="rgba(127,227,255,0.9)"/>
 <text x="390" y="200" text-anchor="middle" font-size="9.5" letter-spacing="0.2" font-weight="800" fill="#FFFFFF">Digital twin</text>
</svg>
<span class="bt-pkt bt-pkt-up" style="animation-delay:0s"></span>
<span class="bt-pkt bt-pkt-up" style="animation-delay:0.7s"></span>
<span class="bt-pkt bt-pkt-up" style="animation-delay:1.4s"></span>
<span class="bt-pkt bt-pkt-up" style="animation-delay:2.1s"></span>
<span class="bt-pkt bt-pkt-down" style="animation-delay:0.35s"></span>
<span class="bt-pkt bt-pkt-down" style="animation-delay:1.75s"></span>
</div>"""


HERO_ART = re.sub(r"<!--.*?-->", "", HERO_ART)
HERO_ART = " ".join(line.strip() for line in HERO_ART.splitlines())      # one line: no markdown code blocks
HERO_CSS = " ".join(line.strip() for line in HERO_CSS.splitlines())
st.markdown(
    HERO_CSS +
    '<div class="bt-hero"><div class="bt-hero-text">'
    '<div class="bt-kicker">NASA Ames lithium-ion ageing data</div>'
    '<div class="bt-title">Battery Digital Twin</div>'
    '<div class="bt-sub">A physical cell and its virtual copy, synchronised cycle by cycle: diagnose ageing, '
    'forecast remaining life and decide when to act.</div>'
    '</div>' + HERO_ART + '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Data ingestion
# =============================================================================
store: Optional[te.ParquetStore] = None
imp: Optional[pd.DataFrame] = None
ct: Optional[pd.DataFrame] = None
cell_errors: Dict[str, str] = {}
source_notes: List[str] = []
try:
    store, imp, source_notes = resolve_sources(up_master, up_imp, master_url.strip(), imp_url.strip(),
                                               master_sha.strip())
    if store is not None:
        with st.spinner("Extracting per-cycle features…"):
            ct, cell_errors = cycle_table(store, store.key)
except Exception as exc:
    report_error("Data ingestion failed", exc, verbose=True)

if store is None or ct is None:
    st.markdown("#### Load your data")
    st.caption("Upload the NASA telemetry file (required) and the EIS file (optional). You can also do this later from "
               "the sidebar (open it with the » arrow at the top left), place the files in `./data/`, or set remote "
               "URLs in the sidebar.")
    u1, u2 = st.columns(2)
    u1.file_uploader("Master telemetry · battery_master_data.parquet", type=["parquet"], key="up_master_main")
    u2.file_uploader("EIS ground truth · impedance_ground_truth.parquet (optional)", type=["parquet"], key="up_imp_main")
    st.markdown("#### Or explore without data")
    if st.button("Load synthetic demo cohort", type="primary", icon=":material/science:"):
        st.session_state["demo"] = True
        st.rerun()
    st.caption("A synthetic cohort whose true ageing parameters are known: useful to try every view.")
    st.stop()

meta = te.cell_meta(ct)
cells = list(meta.index)
DATA_KEY = f"{store.key}|engine-{getattr(te, 'CACHE_VERSION', te.ENGINE_VERSION)}"   # changes only when an existing computation changes


def _cell_label(cid: str) -> str:
    row = meta.loc[cid]
    return (f"{cid}  ({fmt(row['Ambient_C'], '.0f', '°C')}, {fmt(row['I_dis_A'], '.1f', 'A')}, "
            f"{int(row['cycles'])} cycles)")


# =============================================================================
# Global control bar
# =============================================================================
@st.cache_data(show_spinner=False)
def ref_ct_cached(_ct: pd.DataFrame, data_key: str) -> pd.DataFrame:
    return te.reference_condition_ct(_ct)


CT_RAW, DATA_KEY_RAW = ct, DATA_KEY      # as recorded: the accuracy assessment's groups never depend on the toggle below


with st.container(border=True):
    g1, g2, g3 = columns([1.5, 1.0, 2.3], gap="large", vertical_alignment="center")
    with g1:
        cell = st.selectbox("Target cell", cells, key="cell", format_func=_cell_label,
                            index=cells.index("B0005") if "B0005" in cells else 0)
    with g2:
        eol_kind = st.radio("End of life", ["Fixed capacity (Ah)", "% of initial capacity"], horizontal=True,
                            key="eol_kind", help="Fixed capacity: the NASA convention (1.4 Ah = 30 % fade of 2 Ah), each "
                                                 "battery's SOH threshold is this capacity divided by its own initial "
                                                 "capacity. % of initial capacity: the same relative threshold for every "
                                                 "battery, which suits groups whose cells start near the fixed value "
                                                 "(cold cells hold much less capacity). The replacement SOH in "
                                                 "Operations is a separate, optimised decision.")
        if eol_kind.startswith("Fixed"):
            eol_ah = st.number_input("End-of-life capacity (Ah)", 1.05, 2.0, float(te.DEFAULT_EOL_AH), 0.05, key="eol_abs")
        else:
            eol_ah = st.number_input("End of life (% of initial capacity)", 50, 95, 80, 1, key="eol_rel") / 100.0
        if st.toggle("Reference-condition SOH", key="ref_soh",
                     help="Capacity depends on the load and temperature at the moment of measurement. For campaigns "
                          "whose load or ambient changes during the test (the mixed-conditions group), keep only each "
                          "battery's cycles at its initial current and ambient, so SOH measures ageing only. Applies to "
                          "every view; the other cycles are treated as invalid."):
            ct = ref_ct_cached(ct, DATA_KEY)
            DATA_KEY = DATA_KEY + "|reference-condition"

    c_bol = float(meta.loc[cell, "C_bol_Ah"])
    soh_eol = te.soh_eol_for(c_bol, eol_ah)
    ct_cell = ct[ct["Cell_ID"] == cell].sort_values("n")
    eis_cell = te.valid_eis(imp, cell)

    with g3:
        chips = "".join(f'<span class="bt-chip">{html.escape(n)}</span>' for n in source_notes)
        if cell_errors:
            chips += f'<span class="bt-chip">⚠ {len(cell_errors)} cell(s) skipped</span>'
        st.markdown(
            f'<div class="bt-status">Analysing <b>{html.escape(cell)}</b>: '
            f'SOH<sub>EOL</sub> = {soh_eol:.3f} ('
            + (f'{eol_ah:.2f} Ah of {c_bol:.3f} Ah initial' if eol_ah > 1 else f'{100 * eol_ah:.0f} % of its {c_bol:.3f} Ah initial')
            + ')</div>'
            f'<div>{chips}</div>',
            unsafe_allow_html=True,
        )

_fs_bar, _ = fleet_cached(ct, DATA_KEY, float(eol_ah), asdict(te.SafetyLimits()))
_n_crit = int((_fs_bar["Risk"] == "Critical").sum()) if len(_fs_bar) else 0
_n_warn = int((_fs_bar["Risk"] == "Warning").sum()) if len(_fs_bar) else 0
st.markdown(
    '<div class="bt-statusbar">'
    '<span class="bt-live"><span class="bt-dot"></span>Replaying recorded data</span>'
    f'<span>Engine <b>v{te.ENGINE_VERSION}</b></span>'
    f'<span>Source <b>{"synthetic demo" if st.session_state.get("demo") else "telemetry"}</b></span>'
    f'<span><b>{len(meta)}</b> cells, <b>{int(ct["n"].count()):,}</b> cycles</span>'
    f'<span class="bt-sev bt-sev-crit">{_n_crit} critical</span>'
    f'<span class="bt-sev bt-sev-warn">{_n_warn} warning</span>'
    f'<span class="bt-clock">{time.strftime("%Y-%m-%d %H:%M")} UTC</span>'
    '</div>', unsafe_allow_html=True)
view = nav(VIEWS, key="view")
st.session_state["_sec_n"] = 0
_scope_lbl = str(st.session_state.get("diag_scope", "Single battery")).split(": ")[-1].split(" ", 1)[-1]
_sch = {"within": "within a battery", "same": "same operating conditions",
        "across": "across batteries"}.get(st.session_state.get("sch_mode", "within"), "")
st.markdown(
    '<div class="bt-context">'
    f'<span><i>Battery</i> <b>{html.escape(cell_label(cell, meta))}</b></span>'
    f'<span><i>Scope</i> <b>{html.escape(_scope_lbl)}</b></span>'
    f'<span><i>End of life</i> <b>{float(eol_ah):.2f} Ah</b></span>'
    f'<span><i>Training</i> <b>{html.escape(_sch)}</b></span>'
    '<span class="bt-context-hint">change the battery and EOL in the control bar above; other pickers are local to their section</span>'
    '</div>', unsafe_allow_html=True)


# =============================================================================
# View 1: data & diagnostics
# =============================================================================
@fragment
def fade_explorer() -> None:
    """Multi-battery fade chart linked to multi-cycle raw telemetry."""
    all_cells = list(meta.index)
    c1, c2, c3, c4 = st.columns([3, 1.3, 1.6, 1.1])
    cells = list(st.session_state.get("_diag_sel") or [cell])
    c1.markdown(f"**{len(cells)} batter{'y' if len(cells) == 1 else 'ies'}** from the selection above"
                + ("" if len(cells) <= 8 else " (whole fleet: click legend entries to hide cells)"))
    yvar = c2.radio("Metric", ["SOH", "Capacity_Ah"], key="fade_metric",
                    format_func=lambda v: "State of health" if v == "SOH" else "Capacity (Ah)")
    x_mode = c3.radio("x-axis", ["n", "Ah", "frac"], key="fade_x",
                      format_func={"n": "Cycle number", "Ah": "Ah throughput", "frac": "Fraction of life"}.get)
    show_cohort = c4.toggle("Show cohort", value=len(cells) == 1, key="fade_cohort")
    knees = {c: knee_cached(ct[ct["Cell_ID"] == c], DATA_KEY, c) for c in cells}
    eol_line = (soh_eol if yvar == "SOH" else eol_ah) if len(cells) == 1 else (None if yvar == "SOH" else eol_ah)
    fig = fig_fade_compare(ct, meta, cells, yvar, eol_line, P, knees, show_cohort, x_mode)
    event = show_selectable(fig, key="fade_multi")
    export_row(fig, "fade", ct[ct["Cell_ID"].isin(cells)][["Cell_ID", "n", "Cycle_Index", "cum_Ah", "SOH",
                                                           "Capacity_Ah", "outlier", "regen"]])
    if len(cells) > 1:
        rows = []
        for c in cells:
            g = ct[(ct["Cell_ID"] == c) & ~ct["outlier"]]
            k = knees[c]
            rows.append({"Battery": cell_label(c, meta), "Cycles": int(g["n"].max()),
                         "Final SOH": float(g["SOH"].tail(3).median()),
                         "Fade per 100 cycles (%)": 100 * (1 - float(g["SOH"].tail(3).median()))
                         / max(float(g["n"].max()), 1) * 100,
                         "Knee at n": k["knee_n"] if k.get("found") else None,
                         "Regeneration events": int(g["regen"].sum())})
        show_table(pd.DataFrame(rows).set_index("Battery").style.format(
            {"Final SOH": "{:.3f}", "Fade per 100 cycles (%)": "{:.2f}", "Knee at n": "{:.0f}"}, na_rep="—"))

    # ---- raw telemetry of one or many discharges ----
    st.markdown("##### Raw telemetry")
    picked_cell, picked_n = cells[0], None
    try:
        pts = event.selection.points if event is not None else []
        if pts and pts[0].get("customdata") is not None:
            picked_n, picked_cell = int(float(pts[0]["customdata"][0])), str(pts[0]["customdata"][1])
    except Exception:
        picked_n = None
    t1, t2, t3 = st.columns([2, 2, 1.4])
    tcell = t1.selectbox("Battery", cells, index=cells.index(picked_cell) if picked_cell in cells else 0,
                         key="trace_cell", format_func=lambda c: cell_label(c, meta))
    mode = t2.selectbox("Cycles to plot", TRACE_MODES, index=0, key="trace_mode")
    x_axis = t3.radio("x-axis", ["time", "capacity"], key="trace_x",
                      format_func={"time": "Time", "capacity": "Discharged Ah"}.get)
    tc = ct[(ct["Cell_ID"] == tcell) & ~ct["outlier"]].sort_values("n")
    ns = tc["n"].astype(int).tolist()
    if not ns:
        st.info("No valid discharges for this battery.")
        return
    if mode == "Single cycle":
        default = picked_n if (picked_n in ns and picked_cell == tcell) else ns[0]
        chosen = [int(st.select_slider("Discharge cycle", options=ns, value=default, key=f"trace_one_{tcell}"))]
        if picked_n is None:
            st.caption("Tip: click any point on the fade chart to open that discharge.")
    elif mode == "Choose cycles":
        step = max(1, len(ns) // 4)
        chosen = st.multiselect("Discharge cycles", ns, default=ns[::step][:5], key=f"trace_many_{tcell}")
    elif mode == "Every k-th cycle":
        k = st.number_input("k", 1, max(1, len(ns)), max(1, len(ns) // 10), key=f"trace_k_{tcell}")
        chosen = ns[:: int(k)]
    elif mode == "Range of cycles":
        a, b = st.select_slider("Range", options=ns, value=(ns[0], ns[min(len(ns) - 1, 20)]), key=f"trace_rng_{tcell}")
        chosen = [n for n in ns if a <= n <= b]
    else:
        chosen = ns
    if not chosen:
        st.info("Select at least one discharge.")
        return
    if len(chosen) > 80:
        st.caption(f"{len(chosen)} discharges: each is downsampled for speed; the colour bar maps colour to cycle.")
    picks = [(int(tc.loc[tc["n"] == n, "Cycle_Index"].iloc[0]), int(n)) for n in chosen]
    prep = prepared_cell(store, DATA_KEY, tcell)
    cis = [c for c, _ in picks]
    show(fig_cycle_traces(prep, picks, P, x_axis, max_pts=200 if len(picks) > 40 else 500), key="trace_multi",
         data=prep[prep["Cycle_Index"].isin(cis)][["Cycle_Index", "Time_s", "Voltage_V", "Current_A", "Temp_C"]])


@fragment
def ica_section() -> None:
    c1, c2, c3, c4 = st.columns(4)
    n_curves = c1.slider("Curves across life", 2, 12, 6)
    ir = c2.toggle("IR-compensate voltage", value=True,
                   help="Adds |I|·R_dc to the terminal voltage to remove the ohmic shift.")
    dv = c3.select_slider("Voltage bin (mV)", [5, 10, 15, 20], value=10) / 1000
    win = c4.select_slider("Savitzky–Golay window", [5, 7, 9, 11, 15, 21], value=9)
    with st.spinner("Computing dQ/dV curves…"):
        prep = prepared_cell(store, DATA_KEY, cell)
        curves = ica_cached(prep, ct_cell, DATA_KEY, cell, n_curves, ir, dv, win)
    if not curves:
        st.warning("Not enough voltage resolution in this cell's discharges to compute dQ/dV.")
        return
    a = b = st.container()  # stacked full width
    with a:
        data = pd.concat([pd.DataFrame({"n": c.n, "V": c.voltage, "dQdV": c.dqdv}) for c in curves])
        show(fig_ica(curves, P), key="ica", data=data)
    with b:
        show(fig_peaks(curves, P), key="ica_peaks", export=False)
        diag = te.diagnose_degradation(curves)
        if diag["available"]:
            card(f"Degradation reading, n = {diag['n_ref']} to n = {diag['n_cur']}",
                 [f"Capacity ratio {diag['cap_ratio']:.2f}, peak height ratio "
                  f"{diag['height_ratio']:.2f}, peak shift {diag['shift_mV']:.0f} mV"]
                 + list(diag["interpretation"]))
    st.caption("NASA discharges run near 1C, so curves are not near-equilibrium: peak broadening partly "
               "reflects kinetic overpotential and the mode reading is indicative only.")


@fragment
def health_indicator_section() -> None:
    st.markdown("Every candidate health indicator is normalised to its beginning-of-life value and scored with the "
                "prognostic-parameter criteria of Coble & Hines: association with SOH, monotonicity, trendability "
                "(same direction in every cell) and prognosability (cells end at similar values). *LOCO RMSE* asks "
                "whether the indicator alone can stand in for SOH on a cell it has never seen.")
    with st.spinner("Scoring health indicators across the cohort…"):
        tab = hi_rank_cached(ct, imp, DATA_KEY)
        pca = hi_pca_cached(ct, imp, DATA_KEY)
    if tab.empty:
        st.info("Not enough cells with 20+ valid cycles to rank health indicators.")
        return
    show(fig_hi_rank(tab, P), key="hi_rank", data=tab.reset_index())
    show_table(tab.drop(columns=["Unit"]).style.format(
        {"|ρ| with SOH": "{:.2f}", "Monotonicity": "{:.2f}", "Trendability": "{:.2f}", "Prognosability": "{:.2f}",
         "LOCO RMSE (SOH)": "{:.4f}", "Fitness": "{:.2f}"}, na_rep="—")
        .highlight_max(subset=["Fitness"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    non_cap = tab.drop(index=[k for k in ("Capacity_Ah", "t_dis_s", "Q_ch_Ah", "t_cc_s", "E_dis_Wh")
                              if k in tab.index], errors="ignore")
    best_power = non_cap.index[0] if len(non_cap) else None
    a = b = st.container()  # stacked full width
    with a:
        top = [k for k in tab.index[:3]] + ([best_power] if best_power and best_power not in tab.index[:3] else [])
        traj = te.hi_trajectories(ct, imp, cell, top)
        if len(traj):
            show(fig_hi_traj(traj, P, cell), key="hi_traj", data=traj)
    with b:
        if pca.get("available"):
            show(fig_pca(pca, P), key="hi_pca", export=False)
    if pca.get("available"):
        ev = pca["explained"]
        verdict = ("a single health parameter captures the ageing of this cohort" if pca["single_parameter"] else
                   "ageing is multi-dimensional: capacity fade and power fade (resistance) evolve partly "
                   "independently, which supports a multi-parameter state θ = [SOH, R_int, R_ct]")
        card("Mission 1 answer", [
            f"Best-ranked indicator: {tab.iloc[0]['Indicator']} (fitness {tab.iloc[0]['Fitness']:.2f}); "
            f"best power-fade indicator: {te.HI_CATALOG[best_power].label if best_power else '—'}",
            f"PC1 explains {100 * ev[0]:.0f}% of indicator variance (|corr| with SOH {pca['pc_soh_corr'][0]:.2f}); "
            f"PC2 explains {100 * ev[1]:.0f}% → {verdict}."])


@fragment
def modes_section() -> None:
    st.markdown("The white-box view of ageing (Birkl et al. 2017; Menye et al. 2025) splits capacity and power fade "
                "into **loss of lithium inventory** (SEI/CEI growth, plating), **loss of active material** (particle "
                "cracking, transition-metal dissolution, delamination) and **conductivity loss** (interphase and "
                "contact resistance). DVA complements ICA: features that move together indicate LLI, while shrinking "
                "distances between DVA peaks indicate LAM on the electrode that owns them.")
    n_curves = st.slider("Curves across life", 3, 10, 6, key="modes_n")
    with st.spinner("Computing DVA and mode trajectories…"):
        prep = prepared_cell(store, DATA_KEY, cell)
        dva, modes = modes_cached(prep, ct_cell, eis_cell, DATA_KEY, cell, n_curves)
    a = b = st.container()  # stacked full width
    with a:
        if dva:
            data = pd.concat([pd.DataFrame({"n": c.n, "Q_Ah": c.q, "dVdQ": c.dvdq}) for c in dva])
            show(fig_dva(dva, P), key="dva", data=data)
        else:
            st.info("Not enough voltage resolution for DVA on this cell.")
    with b:
        if len(modes):
            show(fig_modes(modes, P), key="modes", data=modes)
    if len(modes):
        last = modes.iloc[-1]
        dom = "LLI" if last["LLI (proxy)"] >= last["LAM (proxy)"] else "LAM"
        card(f"Mode reading at n = {int(last['n'])}", [
            f"Capacity loss {last['Capacity loss']:.1f}% = LLI ≈ {last['LLI (proxy)']:.1f}% + LAM ≈ "
            f"{last['LAM (proxy)']:.1f}% → dominant capacity-loss mode: {dom}",
            f"Conductivity loss: load-step resistance {last['CL: R_dc growth']:+.0f}%"
            + (f", EIS Rₑ+R_ct {last['CL: EIS Rₑ+R_ct growth']:+.0f}%" if "CL: EIS Rₑ+R_ct growth" in modes else "")
            + f"; ICA peak shift {last['Peak shift (mV)']:.0f} mV"])
    st.caption("Proxies from ~1C full-cell data without half-cell references: use them for trends and for comparing "
               "cells, not as absolute mode quantities.")


@fragment
def stress_section() -> None:
    with st.expander("Screening thresholds", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        lim = dict(T_warn_C=float(c1.number_input("Accelerated-ageing T (°C)", 30.0, 60.0, 45.0, 1.0)),
                   T_crit_C=float(c2.number_input("SEI-decomposition onset (°C)", 45.0, 90.0, 60.0, 1.0)),
                   T_plating_C=float(c3.number_input("Plating-risk charge T (°C)", 0.0, 25.0, 10.0, 1.0)),
                   V_deep_V=float(c4.number_input("Deep-discharge limit (V)", 2.0, 3.0, 2.5, 0.05)))
    sx, summ = stress_cached(ct, DATA_KEY, lim)
    L = te.SafetyLimits(**lim)
    show(fig_stress(sx, L, P, cell), key="stress", data=sx[sx["Cell_ID"] == cell][
        ["n", "T_max_C", "V_min_V", "PRI", "hot", "critical_T", "plating", "deep", "high_rate"]])
    mech = pd.DataFrame([{"Stressor": v[0], "Mechanism (literature)": v[1]} for v in te.STRESS_MECHANISMS.values()])
    a = b = st.container()  # stacked full width
    with a:
        st.markdown("**Share of cycles exposed (%) per cell**")
        sel_ = st.session_state.get("_diag_sel")
        if sel_:
            summ = summ[summ.index.isin(sel_)]
        show_table(summ.style.format("{:.0f}", subset=[c for c in summ.columns if c not in ("Max PRI",)])
                   .format("{:.2f}", subset=["Max PRI"]))
    with b:
        st.markdown("**Stressor → degradation mechanism**")
        show_table(mech.set_index("Stressor"))



def risk_banner(row: pd.Series, cell_id: str) -> None:
    col = RISK_COLORS[row["Risk"]]
    st.markdown(f'<div class="bt-alert"><span class="bt-badge" style="background:{col}">{html.escape(row["Risk"].upper())}'
                f'</span><span class="bt-msg"><b>{html.escape(cell_id)}</b>: {html.escape(row["Alerts"])}</span></div>',
                unsafe_allow_html=True)


DIAG_TABS = [":material/space_dashboard: Overview", ":material/trending_down: Ageing",
             ":material/science: Electrochemistry", ":material/warning: Stress & safety"]


def sub_nav(key: str, options: Sequence[str]) -> str:
    """Secondary navigation inside a view; only the chosen tab is computed (fast pages)."""
    st.session_state.setdefault(key, options[0])
    try:
        v = st.segmented_control("Section", list(options), key=key, label_visibility="collapsed")
    except Exception:
        v = st.radio("Section", list(options), key=key, horizontal=True, label_visibility="collapsed")
    return v or options[0]


DIAG_SCOPES = ("single", "selected", "fleet")
DIAG_SCOPE_LABEL = {"single": ":material/battery_full: Single battery",
                    "selected": ":material/stacks: Selected batteries",
                    "fleet": ":material/grid_view: Whole fleet"}


def _pick_critical(fs: pd.DataFrame, k: int = 6) -> None:
    st.session_state["diag_cells"] = list(fs.index[:k])
    st.session_state["diag_scope"] = DIAG_SCOPE_LABEL["selected"]


def battery_status_card(cid: str, fs: pd.DataFrame, lim: Dict[str, float], key: str) -> None:
    """Alert banner, gauge cluster and key facts for one battery."""
    if cid not in fs.index:
        st.info(f"{cid}: not enough valid cycles for a status card.")
        return
    row = fs.loc[cid]
    risk_banner(row, cell_label(cid, meta))
    show(fig_gauges(row, P, te.SafetyLimits(**lim)), key=f"gauges_{key}", export=False)
    k = st.columns(5)
    k[0].metric("Valid cycles", int(meta.loc[cid, "cycles"]))
    k[1].metric("Initial capacity", fmt(meta.loc[cid, "C_bol_Ah"], ".3f", "Ah"))
    k[2].metric("Capacity fade", fmt(meta.loc[cid, "fade_pct"], ".1f", "%"))
    k[3].metric("Fade per 100 cycles", fmt(row["Fade per 100 cycles (%)"], ".2f", "%"))
    k[4].metric("Regeneration events", int(meta.loc[cid].get("regen_events", 0) or 0))


def triage_table(fs: pd.DataFrame, key: str) -> None:
    tbl = fs.reset_index()[["Cell_ID", "Risk", "SOH", "Health margin", "Quick RUL", "Fade per 100 cycles (%)",
                            "R growth (%)", "Peak T (°C)", "Cycles", "Ambient_C", "I_dis_A", "Alerts"]].copy()
    tbl["Risk"] = tbl["Risk"].map(lambda r: f"{RISK_ICON[r]} {r}")
    tbl["Quick RUL"] = tbl["Quick RUL"].replace(np.inf, np.nan)
    try:
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=min(640, 38 + 35 * len(tbl)), key=key,
                     column_config={
                         "Cell_ID": st.column_config.TextColumn("Battery", width="small"),
                         "SOH": st.column_config.ProgressColumn("SOH", min_value=0.0, max_value=1.05, format="%.3f"),
                         "Health margin": st.column_config.ProgressColumn("Health margin", min_value=0.0, max_value=1.0,
                                                                          format="%.2f"),
                         "Quick RUL": st.column_config.NumberColumn("Quick RUL", format="%.0f cyc"),
                         "Fade per 100 cycles (%)": st.column_config.NumberColumn("Fade /100 cyc", format="%.2f %%"),
                         "R growth (%)": st.column_config.NumberColumn("R growth", format="%+.0f %%"),
                         "Peak T (°C)": st.column_config.NumberColumn("Peak T", format="%.1f °C"),
                         "Ambient_C": st.column_config.NumberColumn("Ambient", format="%.0f °C"),
                         "I_dis_A": st.column_config.NumberColumn("Load", format="%.1f A"),
                         "Alerts": st.column_config.TextColumn("Alerts", width="large")})
    except Exception:
        show_table(tbl)


def event_log(ev: pd.DataFrame, cells: Sequence[str], key: str) -> None:
    c1, c2 = st.columns([3, 1])
    sev = c1.multiselect("Severity", ["Critical", "Warning", "Watch", "Info"], default=["Critical", "Warning", "Watch"],
                         key=f"{key}_sev")
    evf = ev[ev["Severity"].isin(sev) & ev["Cell_ID"].isin(cells)]
    c2.metric("Events shown", len(evf))
    evf = evf.assign(Severity=evf["Severity"].map(lambda s_: {"Critical": "🔴", "Warning": "🟠", "Watch": "🟡",
                                                              "Info": "🔵"}[s_] + " " + s_))
    if evf.empty:
        st.success("No events of the chosen severity for this selection.")
        return
    try:
        st.dataframe(evf, hide_index=True, use_container_width=True, height=min(420, 38 + 35 * len(evf)), key=key,
                     column_config={"Cell_ID": st.column_config.TextColumn("Battery"),
                                    "n": st.column_config.NumberColumn("Cycle", format="%d")})
    except Exception:
        show_table(evf)


def selection_report(sel: Sequence[str], fs: pd.DataFrame, ev: pd.DataFrame, lim: Dict[str, float]) -> None:
    label = sel[0] if len(sel) == 1 else f"{len(sel)} batteries"
    if st.button(f"Build report · {label}", key="rep_go", icon=":material/description:", type="primary"):
        with st.spinner("Assembling report…"):
            knees = {c: knee_cached(ct[ct["Cell_ID"] == c], DATA_KEY, c) for c in sel[:12]}
            fs_sel = fs[fs.index.isin(sel)]
            secs: List[Tuple[str, Any]] = [
                ("Selection", ", ".join(cell_label(c, meta) for c in sel)),
                ("Status", fs_sel.drop(columns=["risk_level"])),
                ("Capacity fade", fig_fade_compare(ct, meta, list(sel), "SOH", soh_eol if len(sel) == 1 else None, P,
                                                   knees, len(sel) == 1, "n")),
                ("Risk matrix (selection against the fleet)", fig_risk_matrix(fs, P, sel[0],
                                                                              list(sel) if len(sel) < len(fs) else None)),
            ]
            for c in sel[:8]:
                if c in fs.index:
                    secs.append((f"Instrument cluster · {c}", fig_gauges(fs.loc[c], P, te.SafetyLimits(**lim))))
                    secs.append((f"Events · {c}", ev[ev["Cell_ID"] == c]))
            saved = st.session_state.get("cmp")
            if saved and saved.get("res") is not None and saved["res"].cell_id in sel:
                try:
                    secs.append((f"Forecasts · {saved['res'].cell_id}", fig_compare(saved["res"], P)))
                    secs.append(("Forecast metrics", te.metrics_table(saved["res"].metrics)))
                except Exception:
                    pass
            st.session_state["report"] = (tuple(sel), build_report_html(f"Battery health report · {label}", secs))
    rep = st.session_state.get("report")
    if rep and tuple(rep[0]) == tuple(sel):
        download("Download report (HTML)", rep[1], f"battery_report_{label.replace(' ', '_')}.html", "text/html",
                 key="rep_dl")


def status_section(scope: str, sel: List[str], fs: pd.DataFrame, ev: pd.DataFrame, lim: Dict[str, float]) -> None:
    """Health status for the current scope: instrument cards (single / selected) or triage (fleet)."""
    if fs.empty:
        st.info("Not enough valid cycles to assess health.")
        return
    if scope == "fleet":
        k = st.columns(6)
        k[0].metric("Batteries", len(fs))
        k[1].metric("Mean SOH", fmt(100 * fs["SOH"].mean(), ".1f", "%"))
        k[2].metric("Past end of life", int((fs["SOH"] <= fs["SOH_EOL"]).sum()))
        k[3].metric("Critical / warning", f"{int((fs['Risk'] == 'Critical').sum())} / "
                                          f"{int((fs['Risk'] == 'Warning').sum())}")
        k[4].metric("Knees detected", int(fs["Knee"].sum()))
        k[5].metric("Fleet cycles logged", f"{int(fs['Cycles'].sum()):,}")
        show(fig_fleet_map(fs, P), key="fleet_map", data=fs.reset_index())
        show(fig_risk_matrix(fs, P, cell), key="risk_matrix", data=fs.reset_index())
        triage_table(fs, "triage_all")
        note("Every battery ranked worst-first: health margin bars, quick remaining life, fade speed and the alerts behind the risk level.")
    else:
        if len(sel) > 1:                    # single scope: the résumé card above already shows these numbers
            for c in sel:
                battery_status_card(c, fs, lim, c)
            triage_table(fs[fs.index.isin(sel)], "triage_sel")
            note("Side-by-side status of the selected batteries: which one is closest to end of life and why.")
        show(fig_risk_matrix(fs, P, sel[0], sel), key="risk_matrix", data=fs.reset_index())
        st.caption("Selected batteries in colour, the rest of the fleet in grey. Quick RUL is a robust trend of the "
                   "last 20 cycles for triage; the Models view gives the full probabilistic RUL.")
    with st.expander("Event log", expanded=scope != "fleet", icon=":material/list_alt:"):
        event_log(ev, sel, "ev")
        note("Chronological log of notable events (end of life, knees, over-temperature, cold charging, regeneration): when each battery's trouble started.")
    with st.expander("Report", icon=":material/description:"):
        st.caption("Self-contained HTML with interactive charts for the current selection (print to PDF from the "
                   "browser).")
        selection_report(sel if scope != "fleet" else list(fs.index[:8]), fs, ev, lim)


def _best_key(tab: pd.DataFrame, col: str, lowest: bool = False) -> Optional[str]:
    """Index of the largest (or lowest) finite value in ``col``; None when empty or all NaN."""
    if tab is None or not len(tab) or col not in tab:
        return None
    v = pd.to_numeric(tab[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        return None
    return v.idxmin() if lowest else v.idxmax()


def live_accuracy_cards(sk: pd.DataFrame, fr: Any, show_models: Sequence[str], reveal: bool,
                        eol_true: Optional[int]) -> None:
    models = [m for m in show_models if fr is not None and m in fr.forecasts]
    if not models:
        return
    st.markdown("**Live accuracy** · 5-cycle-ahead predictions scored so far (causal)")
    by = sk.set_index("key") if len(sk) else pd.DataFrame()
    best = _best_key(by, "Accuracy (%)")
    cols = st.columns(len(models))
    for c, m in zip(cols, models):
        name = te.LIVE_MODELS[m].split(" · ")[0]
        if m in by.index and np.isfinite(by.loc[m, "Accuracy (%)"]):
            c.metric(("★ " if m == best else "") + name, f"{by.loc[m, 'Accuracy (%)']:.2f}%",
                     delta=f"RMSE {by.loc[m, 'RMSE (5-step ahead)']:.4f}", delta_color="off",
                     help=f"{int(by.loc[m, 'Predictions scored'])} predictions scored; bias "
                          f"{by.loc[m, 'Bias']:+.4f} (positive = optimistic).")
        else:
            c.metric(name, "—", help="Scored once the first 5-cycle-ahead prediction can be checked.")
    if reveal:
        good = ct_cell[~ct_cell["outlier"]]
        hind = te.live_forecast_accuracy(fr, good, soh_eol, eol_true)
        if len(hind):
            hb = hind.set_index("key")
            best_h = _best_key(hb, "Accuracy (%)")
            st.markdown(f"**Forecast accuracy against the actual future** · forecasts made at n = {fr.n}, scored on "
                        f"the {int(hb['Cycles scored'].max())} cycles that followed (hindsight)")
            cols = st.columns(len(models))
            for c, m in zip(cols, models):
                if m in hb.index and np.isfinite(hb.loc[m, "Accuracy (%)"]):
                    r = hb.loc[m]
                    eol_txt = (f"EOL error {r['EOL error (cycles)']:+.0f} cyc" if np.isfinite(r["EOL error (cycles)"])
                               else f"coverage {100 * r['Band coverage']:.0f}%")
                    c.metric(("★ " if m == best_h else "") + te.LIVE_MODELS[m].split(" · ")[0],
                             f"{r['Accuracy (%)']:.2f}%", delta=eol_txt, delta_color="off",
                             help=f"RMSE {r['RMSE']:.4f}; band coverage {100 * r['Band coverage']:.0f}%.")


def _study_answers(res: Dict[str, Any]) -> Dict[str, List[str]]:
    """Plain-language answers to the Mission questions, generated from the evidence."""
    A: Dict[str, List[str]] = {"M1": [], "M2": [], "M3": []}
    hi = res.get("hi")
    if hi is not None and len(hi):
        top = hi.iloc[0]
        A["M1"].append(f"Best health indicator: {top['Indicator']} (fitness {top['Fitness']:.2f}); best operando "
                       f"power-fade indicator: {hi[~hi.index.isin(['Capacity_Ah', 'E_dis_Wh', 't_dis_s', 'Q_ch_Ah', 't_cc_s'])].iloc[0]['Indicator'] if len(hi) > 1 else '—'}.")
    pca = res.get("pca") or {}
    if pca.get("available"):
        ev = pca["explained"]
        A["M1"].append(f"One parameter or several? PC1 explains {100 * ev[0]:.0f}% of indicator variance and PC2 "
                       f"{100 * ev[1]:.0f}%: " + ("a single health parameter suffices." if pca["single_parameter"] else
                                                  "ageing is multi-dimensional (capacity and power fade diverge), "
                                                  "so the twin tracks SOH and two resistances."))
    sf = res.get("sf") or {}
    fleet_ = st.session_state.get("itw_fleet") or {}
    if fleet_.get("main") is not None:                     # identified effects from the shared physics (not the
        idf = te.whatif_identifiability(fleet_["main"], te.condition_groups(CT_RAW))   # pooled screening regression)
        by = {s_: [c_ for c_, v_ in zip(idf["condition"], idf["status"]) if v_ == s_]
              for s_ in ("identified", "limited", "not identifiable")}
        A["M1"].append("Operating conditions (shared physics): identified: " + (", ".join(by["identified"]) or "none")
                       + "; at the edge of its range: " + (", ".join(by["limited"]) or "none")
                       + "; not identifiable from NASA: " + (", ".join(by["not identifiable"]) or "none") + ".")
    mc = res.get("mech_checks")
    if mc is not None and len(mc):
        A["M1"].append("Mechanisms: " + "; ".join(f"{k.lower()} → {r['Verdict']}" for k, r in mc.iterrows()) + ".")
    summ = res.get("summary")
    if summ is not None and len(summ):
        allb = summ.loc["All batteries"] if "All batteries" in summ.index.get_level_values(0) else None
        if allb is not None:
            best = allb["median RMSE"].idxmin()
            A["M2"].append(f"Across all batteries the most accurate 20-cycle forecaster is {te.LIVE_MODELS[best]} "
                           f"(median RMSE {allb.loc[best, 'median RMSE']:.4f}, accuracy "
                           f"{allb.loc[best, 'median accuracy (%)']:.2f}%, band coverage {allb.loc[best, 'coverage']:.0%}).")
        for grp in [g for g in te.GROUP_ORDER if g in summ.index.get_level_values(0)]:
            gs = summ.loc[grp]
            b = gs["median RMSE"].idxmin()
            A["M2"].append(f"{grp}: best {te.LIVE_MODELS[b]} ({gs.loc[b, 'median RMSE']:.4f}; twin "
                           f"{gs['median RMSE'].get('twin', float('nan')):.4f}).")
    pt = res.get("paired")
    if pt is not None and len(pt) and "All batteries" in pt.index:
        r = pt.loc["All batteries"]
        A["M2"].append(f"Live ensemble vs twin (paired): better in {r['ens better in']} forecasts, median difference "
                       f"{r['median difference']:+.4f}, Wilcoxon p = {r['Wilcoxon p']:.3g}.")
    up = res.get("update")
    if up is not None and len(up):
        rec = up["recommended"].dropna()
        if len(rec):
            A["M2"].append(f"Update interval: the sparsest schedule within tolerance is every {int(rec.median())} "
                           f"cycle(s) (median over {len(rec)} batteries; range {int(rec.min())}–{int(rec.max())}).")
    mt = st.session_state.get("m3_maint")
    if mt:
        sm = mt["summary"]
        if "twin-informed" in sm.index:
            A["M3"].append(f"Replacement on the real lives: the twin captures "
                           f"{100 * sm.loc['twin-informed', 'share of perfect-knowledge gain']:.0f} % of the possible "
                           f"gain over run-to-end-of-life ({int(sm.loc['twin-informed', 'unplanned'])} unplanned of "
                           f"{int(sm.loc['twin-informed', 'batteries'])}); best fixed age "
                           + (f"{100 * sm.loc['best fixed age', 'share of perfect-knowledge gain']:.0f} %."
                              if "best fixed age" in sm.index else "n/a."))
            if "twin + alarm + age limit" in sm.index:
                A["M3"].append(f"With the anomaly alarms in the decision: twin + alarm + age limit "
                               f"{100 * sm.loc['twin + alarm + age limit', 'share of perfect-knowledge gain']:.0f} % "
                               f"({int(sm.loc['twin + alarm + age limit', 'unplanned'])} unplanned).")
    ops = st.session_state.get("m3_ops")
    if ops is not None and len(ops):
        best = (ops.sort_values("profit rate (per hour)", ascending=False)
                .groupby(["battery", "battery_cost_cycles", "buy_price_ratio", "soh_eol"]).head(1))
        A["M3"].append(f"Operation: the battery's own current and cut-off is best in {100 * best['own'].mean():.0f} % "
                       f"of battery × price settings ({best['battery'].nunique()} batteries).")
    lf = st.session_state.get("m3_life")
    if lf:
        sm = lf["summary"]
        for gname in [x for x in sm.index.get_level_values(0).unique() if x != "all"]:
            g = sm.loc[gname, "vs usual practice (% of operating profit per hour)"]
            A["M3"].append(f"Expected benefit, {gname} ({int(sm.loc[gname, 'batteries'].iloc[0])} batteries): twin "
                           f"strategy {g['twin strategy']:+.0f} % lifecycle profit per hour vs usual practice "
                           f"({g['twin replacement']:+.0f} % replacement, "
                           f"{g['twin strategy'] - g['twin replacement']:+.0f} % operation); upper bound "
                           f"{g['upper bound']:+.0f} %.")
    if not A["M3"]:
        A["M3"].append("Run the operation and replacement studies in Operations & control to fill in this answer.")
    return A


def view_study() -> None:
    recommendations("study")
    section("Study results: answers to the Mission questions")
    st.markdown("Runs the live multi-model twin on **every usable battery** and scores each model's forecasts "
                "(made at 30% and 50% of recorded life, over the next 20 cycles) per operating-condition group, "
                "with paired statistical tests, mechanism-physics checks and a cohort update-interval study. "
                "Results use the NASA data loaded in this session; nothing here is synthetic unless the demo "
                "cohort is loaded.")
    groups = te.condition_groups(ct)
    with st.expander(f"Evaluation groups · {groups['Group'].nunique()} groups, {len(groups)} batteries",
                     icon=":material/category:"):
        show_table(groups[["Group", "Ambient_C", "I_dis_A", "rest_frac", "cycles"]].sort_values("Group")
                   .style.format({"Ambient_C": "{:.0f}", "I_dis_A": "{:.1f}", "rest_frac": "{:.2f}"}))
        st.caption("Pulsed-load cells (square-wave discharge) use a pulse-aware twin (no partial-window voltage "
                   "model). Corrupted-logging cells are reported separately so they do not distort the others.")
    c1, c2 = st.columns(2)
    n_est = int((groups["cycles"] >= 25).sum())
    run = c1.button(f"Run cohort study (≈ {3 * n_est + 20} s)", key="study_go", type="primary",
                    icon=":material/science:")
    with_update = c2.toggle("Include cohort update-interval study", value=True, key="study_upd")
    if run:
        prog = st.progress(0.0, text="Starting…")
        cb = lambda f, m: prog.progress(min(max(float(f), 0.0), 1.0), text=m)
        try:
            val, mech = te.cohort_validation(store, ct, imp, float(eol_ah), progress=cb)
            res = {"val": val, "mech": mech, "summary": te.cohort_summary(val),
                   "paired": te.paired_model_test(val, "ens", "twin") if len(val) else pd.DataFrame(),
                   "mech_checks": te.mechanism_checks(mech), "hi": hi_rank_cached(ct, imp, DATA_KEY),
                   "pca": hi_pca_cached(ct, imp, DATA_KEY), "sf": stress_factors_cached(ct, DATA_KEY)}
            if with_update:
                prog.progress(0.95, text="Update-interval study on representative batteries…")
                reps = tuple(groups.reset_index().sort_values("cycles", ascending=False).groupby("Group").head(2)["Cell_ID"])
                res["update"] = update_cohort_cached(store, ct, imp, DATA_KEY, float(eol_ah), reps)
            st.session_state["study"] = {"key": DATA_KEY, "res": res}
        except Exception as exc:
            report_error("Cohort study failed", exc, debug)
        prog.empty()
    saved = st.session_state.get("study")
    if not saved or saved["key"] != DATA_KEY:
        st.info("Press **Run cohort study** to generate the evidence. It runs once per dataset and is kept for the "
                "session.")
        return
    res = saved["res"]
    A = _study_answers(res)
    status = {"M1": "answered" if A["M1"] else "open",
              "M2": "answered" if (res.get("update") is not None and len(res["update"])) else "partly answered",
              "M3": "answered" if (st.session_state.get("m3_maint") and st.session_state.get("m3_ops") is not None)
              else "partly answered"}
    chips = "".join(f'<div class="bt-mchip bt-m-{v.split()[0]}"><b>{m}</b> {html.escape(v)}</div>' for m, v in status.items())
    st.markdown(f'<div class="bt-mstrip">{chips}</div>', unsafe_allow_html=True)
    section("Mission 1 · Health, operating conditions and mechanisms")
    card("Answers", A["M1"] or ["—"])
    if res.get("hi") is not None and len(res["hi"]):
        show(fig_hi_rank(res["hi"], P), key="st_hi", data=res["hi"].reset_index())
    if len(res["mech"]):
        show(fig_mech_groups(res["mech"], P), key="st_mech", data=res["mech"])
    if res.get("mech_checks") is not None and len(res["mech_checks"]):
        show_table(res["mech_checks"].style.format({"median share (group)": "{:.0%}", "median share (reference)": "{:.0%}",
                                                    "Mann-Whitney p": "{:.3g}"}, na_rep="—"))
        st.caption("The attribution is model-based. Where the half-cell fit is available, compare plating + SEI with "
                   "the fitted LLI and the LAM share with the fitted LAM in Diagnostics.")
    section("Mission 2 · Prediction accuracy, informative variables and update frequency")
    card("Answers", A["M2"] or ["—"])
    if len(res["summary"]):
        show(fig_cohort_models(res["summary"], P), key="st_models", data=res["summary"].reset_index())
        show_table(res["summary"].style.format({"median RMSE": "{:.4f}", "IQR low": "{:.4f}", "IQR high": "{:.4f}",
                                                "median accuracy (%)": "{:.2f}", "coverage": "{:.0%}",
                                                "median |RUL error|": "{:.0f}"}, na_rep="—"))
    if res.get("paired") is not None and len(res["paired"]):
        st.markdown("**Is the live ensemble reliably better than the twin?** (paired Wilcoxon test on forecast RMSE)")
        show_table(res["paired"].style.format({c: "{:.4f}" for c in res["paired"].columns if "RMSE" in c or "difference" in c}
                                              | {"Wilcoxon p": "{:.3g}"}, na_rep="—"))
    if res.get("update") is not None and len(res["update"]):
        st.markdown("**How often should the twin update?** (tracking RMSE when updating every 1, 5 or 20 cycles)")
        show_table(res["update"].set_index("Cell_ID").style.format("{:.4f}", subset=[c for c in res["update"].columns
                                                                                   if c.startswith("RMSE")]))
    section("Mission 3 · Integrated operation and maintenance")
    card("Answers", A["M3"])
    try:
        _, tab = calibrated_plant_cached(ct, imp, DATA_KEY, cell)
        st.markdown(f"**Plant calibrated on {cell}** (used by Operations & control when calibration is on)")
        show_table(tab.style.format({"Value": "{:.4g}"}))
    except Exception:
        pass
    section("Export")
    val_csv = res["val"].to_csv(index=False).encode()
    download("Download all forecast scores (CSV)", val_csv, "cohort_validation.csv", "text/csv", key="st_csv")
    if st.button("Build study report (HTML)", key="st_rep", icon=":material/description:"):
        secs: List[Tuple[str, Any]] = [("Mission 1 answers", " | ".join(A["M1"]))]
        if res.get("hi") is not None and len(res["hi"]):
            secs.append(("Health-indicator ranking", fig_hi_rank(res["hi"], P)))
        if len(res["mech"]):
            secs += [("Mechanism shares by condition", fig_mech_groups(res["mech"], P)),
                     ("Mechanism physics checks", res["mech_checks"])]
        secs += [("Mission 2 answers", " | ".join(A["M2"]))]
        if len(res["summary"]):
            secs += [("Forecast error by condition and model", fig_cohort_models(res["summary"], P)),
                     ("Cohort summary", res["summary"].reset_index())]
        if res.get("paired") is not None and len(res["paired"]):
            secs.append(("Live ensemble vs twin (paired test)", res["paired"].reset_index()))
        if res.get("update") is not None and len(res["update"]):
            secs.append(("Update-interval study", res["update"]))
        secs += [("Mission 3 answers", " | ".join(A["M3"])), ("Evaluation groups", groups.reset_index())]
        st.session_state["study_report"] = build_report_html("Self-updating digital twin · study results", secs)
    if st.session_state.get("study_report"):
        download("Download study report (HTML)", st.session_state["study_report"], "study_results.html", "text/html",
                 key="st_rep_dl")



EVENT_STYLE = {"alarm": ("red", ":material/warning:"), "common": ("orange", ":material/groups:"),
               "eol": ("violet", ":material/battery_alert:"),
               "cleared": ("green", ":material/check_circle:"), "info": ("gray", ":material/info:")}


def live_anomaly_prepare(cell_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """Replay of this battery with both indicators, its common-cause classification and its event log (the fleet scan
    runs once and is cached). Returns (replay, events, note)."""
    cells_an = tuple(sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(CT_RAW, g)}))
    if cell_id not in cells_an:
        return None, None, ("Live anomaly detection needs a battery of the constant-condition groups ("
                            + ", ".join(te.NASA_PHYSICS_GROUPS) + ").")
    with st.spinner("Preparing live anomaly detection for the whole fleet (runs once, then cached)…"):
        scan = anomaly_scan2_cached(CT_RAW, imp, DATA_KEY_RAW, cells_an, 1.8, 3.0)
    groups = te.condition_groups(CT_RAW)
    campaign = {c: str(groups.loc[c, "Group"]) if c in groups.index else c for c in scan}
    cc = te.anomaly_common_cause(scan, campaign)
    return scan[cell_id], te.anomaly_events(scan[cell_id], cc.get(cell_id)), ""


def live_anomaly_panel(tr: pd.DataFrame, ev: pd.DataFrame, n: int, n_max: float) -> None:
    """What the twin knows about anomalies up to cycle n: indicator status, events so far, and the indicators."""
    now = tr[tr["n"] <= n]
    st.markdown("##### :material/monitor_heart: Live anomaly detection")
    c1, c2 = st.columns(2)
    k_now = now["kappa"].dropna()
    f_on = bool(now["fleet alarm"].iloc[-1]) if len(now) else False
    c1.metric("FLEET · ageing speed vs the fleet", f"{k_now.iloc[-1]:.2f}×" if len(k_now) else "warming up",
              delta="ALARM" if f_on else ("normal" if len(k_now) else None),
              delta_color="inverse" if f_on else "off")
    z_now = now["slope z"].dropna()
    o_on = bool(now["own alarm"].iloc[-1]) if len(now) else False
    need = int(tr.loc[tr["slope z"].notna(), "n"].min()) if tr["slope z"].notna().any() else None
    c2.metric("OWN · fade slope vs its own past (z)", f"{z_now.iloc[-1]:.1f}" if len(z_now) else
              ("not enough history yet" if need is None or n < need else "—"),
              delta="ALARM" if o_on else (f"active from cycle {need}" if need and n < need else
                                          "normal" if len(z_now) else "needs ~70 cycles"),
              delta_color="inverse" if o_on else "off")
    seen = ev[ev["cycle"] <= n].sort_values("cycle", ascending=False)
    if seen.empty:
        st.caption("No event yet: the battery behaves as expected so far.")
    else:
        lines = []
        for r in seen.itertuples(index=False):
            col, ic = EVENT_STYLE.get(r.kind, ("gray", ":material/info:"))
            lab = {"alarm": "alarm", "common": "common event", "cleared": "back to normal", "info": "note", "eol": "end of life"}[r.kind]
            lines.append(f"{ic} :{col}[**cycle {r.cycle:.0f} · {r.indicator} {lab}**] — {r[3]}")
        with st.container(height=min(260, 48 + 46 * len(lines)), border=True):
            st.markdown("  \n".join(lines))
    show(fig_anomaly2(now, P, f"{cell} (up to cycle {n})", 1.8, 3.0, height=430, n_max=n_max), key="rp_anom",
         export=False)


def view_replay() -> None:
    recommendations("live")
    section(f"Live twin replay · {cell}")
    st.markdown("The recorded life of the battery is streamed one discharge at a time. At every step the dual "
                "time-scale EKF assimilates only the partial-window voltage and load-step resistance of the new "
                "cycle, updates SOH, resistances and the personal degradation rate *k*, and re-forecasts the "
                "remaining life. Nothing after the cursor is visible to the twin.")
    level = 0.9
    cap_every = st.select_slider("Reference capacity check every N cycles (0 = operando only)", [0, 5, 10, 20, 50],
                                 value=10, key="rp_cap",
                                 help="Between checks the twin sees only partial-window voltage and load-step "
                                      "resistance. Set 0 to watch pure operando tracking, including its drift.")
    grp_c, fz, train_c = frozen_context(cell)
    use_fz, refit, anchor = False, 10, "capacity"
    why_not = frozen_unavailable_reason(cell, grp_c, fz, train_c)
    with st.container(border=True):
        st.markdown(f"**Frozen assessment model** · battery **{html.escape(cell)}** · group "
                    f"**{html.escape(grp_c or 'none')}**")
        if why_not:
            st.info(why_not, icon=":material/info:")
        else:
            f1, f2, f3 = st.columns([1.3, 1.2, 2.0], vertical_alignment="center")
            use_fz = f1.toggle(f"Frozen {grp_c} model", value=True, key="rp_fz",
                               help=f"The assessment winner of the {grp_c} group: forecast "
                                    f"{model_label(fz[0])}, SOH-now {model_label(fz[1])} "
                                    f"({te.FROZEN_SOURCE}). It is retrained on the group's other batteries plus this "
                                    "battery's history up to now, and its forecast is re-anchored every cycle to the "
                                    "current SOH.")
            refit = f2.select_slider("Retrain every (cycles)", [5, 10, 20, 50, 0], value=0, key="rp_fz_refit",
                                     format_func=lambda v: "never" if v == 0 else str(v), disabled=not use_fz)
            anchor = f3.radio("Re-anchor every cycle to", list(te.FROZEN_ANCHORS), key="rp_fz_anchor", horizontal=True,
                              format_func=lambda k_: te.FROZEN_ANCHORS[k_].split(" (")[0], disabled=not use_fz,
                              help="Measured capacity: lab, capacity measured on every discharge. SOH-now model: "
                                   "field mode, no capacity measured after the start; the frozen SOH-now model "
                                   "estimates it from the operando signals and retraining sees only those estimates. "
                                   "None: the model's own forecast.")
            st.caption(f"Forecast: {model_label(fz[0])} · SOH-now: {model_label(fz[1])} · "
                       f"retrained on {len(train_c)} other {grp_c} batteries plus this one's history"
                       + (" · switch on Reference-condition SOH in the control bar to replay this battery as it was "
                          "assessed" if grp_c == "Mixed conditions" and not st.session_state.get("ref_soh") else ""))
    show_models = st.multiselect(
        "Live models", list(te.LIVE_MODELS),
        default=[m for m in ("twin", "mech", "pf", "trend", "frozen", "ens") if m in te.LIVE_MODELS], key="rp_models",
        format_func=lambda m: (te.LIVE_MODELS[m] if m != "frozen" else
                               (f"Frozen {grp_c} model · {model_label(fz[0])}" if use_fz else
                                "Frozen model · not running for this battery")),
        help="ECM twin: physics + operando voltage. Particle filter: power law with the fleet prior, robust to knees. "
             "Adaptive trend KF: level-slope-curvature filter that bends quickly when fade accelerates. Hierarchical "
             "Bayes: fleet-informed power law. Mechanistic PF: SEI, lithium-plating and LAM kinetics driven by each "
             "cycle's measured temperature and current, so the operating conditions decide which mechanism grows. "
             "Live ensemble: weights every model by its recent 5-cycle-ahead error.")
    fz_frames = None
    if use_fz:
        with st.spinner(f"Streaming the frozen {grp_c} model (" + ("trained once" if int(refit) == 0 else
                                                                    f"retrained every {int(refit)} cycles") + "; saved, so it runs only once)…"):
            try:
                ns_, nm_ = te.frozen_live_window(ct, cell, train_c)   # live start: 30 % of the training lives
                fz_frames, _fz_track = frozen_cached(ct, imp, DATA_KEY, cell, tuple(train_c), fz[0], fz[1],
                                                     float(soh_eol), int(refit), anchor, float(eol_ah), ns_, nm_)
            except Exception as exc:
                report_error("Frozen model stream failed", exc, debug)
                use_fz = False
    with st.spinner("Streaming the battery through all models (runs once, then animates)…"):
        res, _track_old = replay_cached(cell_frame(store, DATA_KEY, cell), ct, imp, DATA_KEY, cell, float(soh_eol),
                                        level, int(cap_every))
        st.session_state["ekf"] = {"key": (DATA_KEY, cell, "live"), "res": res}     # feeds Operations initialisation
        run_models = ("twin", "mech", "pf", "trend", "hb")
        frames, track = live_cached(res, ct, imp, DATA_KEY, cell, float(soh_eol), level, int(cap_every), run_models,
                                    frozen_cfg=(fz[0], fz[1], int(refit), anchor) if use_fz else None,
                                    _extra={"frozen": fz_frames} if use_fz and fz_frames else None)
    pc = res.per_cycle
    ns = pc["n"].astype(int).tolist()
    if len(ns) < 6:
        st.info("This battery has too few cycles for a replay.")
        return
    key_n, key_p = f"rp_n_{cell}", "rp_play"
    st.session_state.setdefault(key_n, ns[min(5, len(ns) - 1)])
    st.session_state.setdefault(key_p, False)
    c = st.columns([1, 1, 1, 1, 1.4, 1.4, 1.4])
    playing = bool(st.session_state[key_p])
    if c[0].button("Pause" if playing else "Play", key="rp_toggle", type="primary",
                   icon=":material/pause:" if playing else ":material/play_arrow:"):
        st.session_state[key_p] = not playing
        if not playing and st.session_state[key_n] >= ns[-1]:
            st.session_state[key_n] = ns[5]
        st.rerun()
    if c[1].button("Step", key="rp_step", icon=":material/skip_next:"):
        st.session_state[key_n] = min(ns[-1], st.session_state[key_n] + 1)
    if c[2].button("Back", key="rp_back", icon=":material/skip_previous:"):
        st.session_state[key_n] = max(ns[0], st.session_state[key_n] - 1)
    if c[3].button("Reset", key="rp_reset", icon=":material/replay:"):
        st.session_state[key_n] = ns[min(5, len(ns) - 1)]
        st.session_state[key_p] = False
    speed = c[4].select_slider("Cycles per frame", [1, 2, 3, 5, 10], value=2, key="rp_speed")
    interval = c[5].select_slider("Frame interval (s)", [0.4, 0.7, 1.0, 1.5], value=0.7, key="rp_int")
    st.toggle("Show the animated battery draining with the replay", value=True, key="rp_3d")
    reveal = c[6].toggle("Reveal future data", value=False, key="rp_reveal",
                         help="Show the cycles the twin has not seen yet, to judge the forecast.")
    good = ct_cell[~ct_cell["outlier"]]
    eol_true = te.first_crossing(good["n"].to_numpy(), good["SOH"].to_numpy(), soh_eol, smooth=5)
    n_max = int(max(pc["n"].max() * 1.35, (eol_true or 0) * 1.1))
    frag = getattr(st, "fragment", None)
    live_an = st.toggle("Live anomaly detection", value=True, key="rp_an",
                        help="Two indicators run as the battery is replayed: FLEET (ages faster than the fleet's physics) "
                             "and OWN (fade steeper than in its own past); every change is logged in plain words. The "
                             "fleet scan runs once (about a minute), then it is instant.")
    an_tr = an_ev = None
    if live_an:
        try:
            an_tr, an_ev, an_note = live_anomaly_prepare(cell)
            if an_note:
                st.info(an_note)
        except Exception as exc:
            report_error("Live anomaly detection unavailable", exc, debug)

    def frame() -> None:
        if st.session_state.get(key_p):
            nxt = st.session_state[key_n] + int(speed)
            if nxt >= ns[-1]:
                nxt, st.session_state[key_p] = ns[-1], False
            st.session_state[key_n] = nxt
        n = int(st.session_state[key_n])
        n = min(ns, key=lambda v: abs(v - n))
        row = pc[pc["n"] == n].iloc[0]
        fr = frames.get(n) or (frames[max(k_ for k_ in frames if k_ <= n)] if any(k_ <= n for k_ in frames) else None)
        meas = good[good["n"] <= n]["SOH"].tail(1)
        ens_rul = fr.rul.get("ens") if fr is not None else None
        lead = max(fr.weights, key=fr.weights.get) if fr is not None and fr.weights else None
        k = st.columns(6)
        k[0].metric("Cycle", f"{n} / {ns[-1]}")
        k[1].metric("Twin SOH", f"{row['SOH']:.3f}", delta=f"±{2 * row['SOH_std']:.3f} (2σ)", delta_color="off")
        k[2].metric("Measured SOH", f"{float(meas.iloc[0]):.3f}" if len(meas) else "—")
        k[3].metric("Ensemble RUL", f"{ens_rul[0]:.0f} cyc" if ens_rul and np.isfinite(ens_rul[0]) else "beyond horizon",
                    delta=(f"90%: {ens_rul[1]:.0f}–{ens_rul[2]:.0f}" if ens_rul and np.isfinite(ens_rul[1])
                           and np.isfinite(ens_rul[2]) else None), delta_color="off")
        k[4].metric("Most trusted now", te.LIVE_MODELS[lead].split(" · ")[0] if lead else "—",
                    delta=f"weight {fr.weights[lead]:.2f}" if lead else None, delta_color="off")
        sh = fr.mech_shares if fr is not None else None
        if sh:
            dom = max(sh, key=sh.get)
            k[5].metric("Dominant mechanism", dom, delta=f"{100 * sh[dom]:.0f}% of the loss so far", delta_color="off",
                        help="Mechanistic PF: share of the capacity lost so far owed to SEI growth, lithium plating "
                             "and loss of active material, driven by the measured temperature and current.")
        else:
            k[5].metric("Personal k / prior", f"{row['k_ah'] / res.params.get('k_prior', row['k_ah']):.2f}×",
                        help="The twin's own degradation rate relative to the fleet prior (Sage–Husa adaptive).")
        st.progress(min(1.0, (n - ns[0]) / max(ns[-1] - ns[0], 1)),
                    text=f"{'▶ streaming' if st.session_state.get(key_p) else '⏸ paused'} · "
                         f"{len(good[good['n'] <= n])} discharges assimilated")
        sk = te.live_skill_table(track[track["n"] <= n])
        acc_map = dict(zip(sk["key"], sk["Accuracy (%)"])) if len(sk) else {}
        if st.session_state.get("rp_3d", True):
            rul_now = (fr.rul.get("ens") or (np.nan,))[0] if fr is not None else np.nan
            risk_now = ("Critical" if row["SOH"] <= soh_eol or (np.isfinite(rul_now) and rul_now < 15) else
                        "Warning" if np.isfinite(rul_now) and rul_now < 50 else "Healthy")
            render_battery_anim([{"id": cell, "soh": float(row["SOH"]), "risk": risk_now,
                                  "sub": f"cycle {n} · twin estimate"}], big=False)
        show(fig_live_main(ct_cell, pc, fr, n, soh_eol, P, reveal, n_max, show_models, acc_map), key="rp_main",
             export=False)
        if an_tr is not None and an_ev is not None:
            live_anomaly_panel(an_tr, an_ev, n, n_max)
        live_accuracy_cards(sk, fr, show_models, reveal, eol_true)
        show(fig_live_track(track, n, eol_true, P, n_max, show_models), key="rp_track", export=False)
        if len(sk):
            with st.expander("Live model scoreboard", icon=":material/leaderboard:"):
                show_table(sk.drop(columns=["key"]).set_index("Model").style.format(
                    {"Accuracy (%)": "{:.2f}", "RMSE (5-step ahead)": "{:.4f}", "MAE": "{:.4f}", "Bias": "{:+.4f}",
                     "Final weight": "{:.2f}"}, na_rep="—")
                    .highlight_max(subset=["Accuracy (%)"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
                st.caption("Scored causally: every prediction was made 5 cycles before the measurement it is "
                           "compared with, using only data available at that time. Accuracy = 100 × (1 − mean "
                           "absolute percentage error).")

    if frag is not None:
        try:
            frag(run_every=f"{interval}s" if st.session_state.get(key_p) else None)(frame)()
        except TypeError:
            frame()
    else:
        frame()
    with st.expander("Mechanistic model: governing equations", icon=":material/functions:"):
        st.markdown("The mechanistic particle filter integrates the degradation kinetics cycle by cycle with the "
                    "**measured** cell temperature T and current I, so the operating conditions decide which mechanism "
                    "grows: Arrhenius acceleration of SEI growth when hot, the cold gate of lithium plating below "
                    "~10 °C, the C-rate power of LAM under high current. Rate constants and latent losses are "
                    "estimated jointly by a particle filter from the measured capacity.")
        st.latex(r"\mathrm{SOH} = s_0 - Q_\mathrm{SEI} - Q_\mathrm{pl} - Q_\mathrm{LAM}")
        st.latex(r"\Delta Q_\mathrm{SEI} = k_\mathrm{SEI}\,e^{\frac{E_\mathrm{SEI}}{R}\left(\frac{1}{T_\mathrm{ref}}"
                 r"-\frac{1}{T}\right)}\,\frac{2\,\mathrm{SOH}}{1 + Q_\mathrm{SEI}/\delta}")
        st.latex(r"\Delta Q_\mathrm{pl} = k_\mathrm{pl}\,e^{\frac{E_\mathrm{pl}}{R}\left(\frac{1}{T}-\frac{1}"
                 r"{T_\mathrm{ref}}\right)}\,\frac{I_\mathrm{ch}}{C_0}\left[\frac{1}{1+e^{(T-T_\mathrm{onset})/3}}"
                 r" + \kappa\,\frac{Q_\mathrm{LAM}}{0.05}\right]")
        st.latex(r"\Delta Q_\mathrm{LAM} = k_\mathrm{LAM}\,e^{\frac{E_\mathrm{LAM}}{R}\left(\frac{1}{T_\mathrm{ref}}"
                 r"-\frac{1}{T}\right)}\left(\frac{I}{C_0}\right)^{\beta} 2\,\mathrm{SOH}\left(1 + "
                 r"\frac{Q_\mathrm{LAM}}{\varepsilon}\right)")
        st.caption("Priors: E_SEI = 30, E_pl = 50, E_LAM = 20 kJ/mol; rate constants log-normal around literature-scale "
                   "values. The attributed shares are model-based: validate them against the half-cell LLI / LAM in "
                   "Diagnostics (plating and SEI are LLI, LAM is LAM).")
    pinn_equations_panel()
    methods_panel()
    ablation_section()
    st.caption("Harsh operation (cold plating, knees, high temperature) breaks any single physics law. The live "
               "ensemble moves its weight, cycle by cycle, to the models that are currently predicting well: watch "
               "the weight panel shift when the fade accelerates. "
               "The EOL band narrowing and the personal rate k settling are the self-updating behaviour: the twin "
               "starts from the population prior and converges on this battery's own ageing law as evidence "
               "accumulates. Toggle 'Reveal future data' to judge each forecast against what actually happened.")


@fragment
def half_cell_section() -> None:
    st.markdown("The ICA/DVA proxies above become quantitative here. The pseudo-OCV (discharge voltage + |I|·R_dc) is "
                "fitted with literature half-cell potentials. Four parameters describe the electrode balance: positive "
                "and negative capacities C_p and C_n, and their stoichiometries y₀ (LiᵧCoO₂) and x₀ (LiₓC₆) at the top of "
                "charge. A fifth, η, absorbs the residual polarisation. Their evolution separates loss of lithium "
                "inventory from loss of active material on each electrode.")
    with st.expander("Model equations", icon=":material/functions:"):
        st.latex(r"V(Q) = U_\mathrm{p}\!\left(y_0 + \tfrac{Q}{C_\mathrm{p}}\right) - "
                 r"U_\mathrm{n}\!\left(x_0 - \tfrac{Q}{C_\mathrm{n}}\right) - \eta")
        st.latex(r"\mathrm{LAM_{PE}} = 1 - \frac{C_\mathrm{p}}{C_\mathrm{p,0}},\quad "
                 r"\mathrm{LAM_{NE}} = 1 - \frac{C_\mathrm{n}}{C_\mathrm{n,0}},\quad "
                 r"\mathrm{LLI} = 1 - \frac{x_0 C_\mathrm{n} + y_0 C_\mathrm{p}}{(x_0 C_\mathrm{n} + y_0 C_\mathrm{p})_0}")
        st.caption("U_p: LiCoO₂ (Ramadass et al., J. Electrochem. Soc. 151, 2004). U_n: graphite (Doyle et al., "
                   "J. Electrochem. Soc. 143, 1996). Global search (differential evolution) for the first curve, then "
                   "warm-started local fits (Dubarry et al. 2012; Birkl et al. 2017). On synthetic truth the method "
                   "recovers LLI and LAM_PE within about 1 percentage point.")
    c1, c2 = st.columns(2)
    n_curves = c1.slider("Curves across life", 3, 12, 6, key="hc_n")
    ir = c2.toggle("IR-compensate", value=True, key="hc_ir")
    if st.button("Fit half-cell model", key="hc_go", icon=":material/science:", type="primary"):
        with st.spinner("Fitting electrode balance (global search on the first curve)…"):
            try:
                st.session_state["hc"] = (cell, half_cell_cached(prepared_cell(store, DATA_KEY, cell), ct_cell, DATA_KEY,
                                                                 cell, n_curves, ir))
            except Exception as exc:
                report_error("Half-cell fit failed", exc, debug)
    saved = st.session_state.get("hc")
    if not saved or saved[0] != cell:
        return
    tab, fits = saved[1]
    if not fits:
        st.warning("No curve could be fitted for this battery.")
        return
    last = tab.iloc[-1]
    k = st.columns(4)
    k[0].metric("Loss of lithium inventory", fmt(last["LLI (%)"], ".1f", "%"))
    k[1].metric("LAM positive (LiCoO₂)", fmt(last["LAM_PE (%)"], ".1f", "%"))
    k[2].metric("LAM negative (graphite)", fmt(last["LAM_NE (%)"], ".1f", "%"))
    k[3].metric("Median fit RMSE", fmt(tab["Fit RMSE (mV)"].median(), ".1f", "mV"))
    show(fig_half_cell_modes(tab, P), key="hc_modes", data=tab)
    show(fig_half_cell_fits([fits[0], fits[len(fits) // 2], fits[-1]] if len(fits) > 2 else fits, P), key="hc_fits",
         export=False)
    show_table(tab.set_index("n").style.format("{:.3f}"))
    dom = max(("LLI (%)", "LAM_PE (%)", "LAM_NE (%)"), key=lambda c_: last[c_])
    card("Reading", [f"Dominant mode at n = {int(last['n'])}: {dom.split(' ')[0]}. LLI points to SEI growth or "
                     "plating; LAM_PE to cathode cracking or cobalt dissolution; LAM_NE to graphite exfoliation or "
                     "particle isolation.",
                     "At ~1C the pseudo-OCV still carries kinetic and diffusion overpotentials. Residuals above "
                     "~15 mV or η drifting strongly mean the absolute split is uncertain: compare trends between cells "
                     "rather than trusting single values."])


def _reco_items(view: str) -> List[Tuple[str, str]]:
    ss = st.session_state
    items: List[Tuple[str, str]] = []
    try:
        fs, _ = fleet_cached(ct, DATA_KEY, float(eol_ah), asdict(te.SafetyLimits()))
        n_crit = int((fs["Risk"] == "Critical").sum()) if len(fs) else 0
    except Exception:
        n_crit = 0
    grp = te.condition_groups(ct)["Group"].get(cell, "")
    if view == "diag":
        if n_crit and ss.get("diag_scope", "").endswith("Single battery"):
            items.append(("priority_high", f"{n_crit} batteries are critical: switch the scope to *Selected batteries* "
                                           "and press **Select most critical**."))
        if not ss.get("hc") or ss["hc"][0] != cell:
            items.append(("science", "Quantify the degradation modes of this battery: run **half-cell OCV fitting** below."))
        if grp in ("Reference", ""):
            items.append(("thermostat", "Compare this battery with a **cold (4 °C)** one in the cohort plot: plating shows as fast early fade."))
        items.append(("play_circle", "Open the **Live twin** to watch the models learn this battery cycle by cycle."))
    elif view == "live":
        items.append(("visibility", "Switch on **Reveal future data** and pause at 30–50% of life to judge each forecast."))
        if grp in ("Cold", "Hot", "High current", "Mixed conditions"):
            items.append(("warning", f"This is a **{grp.lower()}** battery: watch the ensemble weights move away from the ECM twin."))
        else:
            items.append(("swap_horiz", "Pick a cold or hot battery in the control bar to see the mechanistic model take over."))
    elif view == "models":
        done = {e["level"] for e in ladder_entries()}
        if not done:
            items.append(("flag", "Start with **Level 1 · Baselines**: every other model must beat the linear trend."))
        elif 2 not in done:
            items.append(("trending_up", "Next: **Level 2 · Machine learning**, then compare in the leaderboard."))
        elif 3 not in done:
            items.append(("trending_up", "Next: **Level 3 · Deep learning**: does it beat Level 2 on the same test?"))
        elif 4 not in done:
            items.append(("trending_up", "Next: **Level 4 · Physics-informed**: does the physics improve the forecast?"))
        items.append(("fact_check", "One battery is an anecdote: confirm the ranking with the **Same operating conditions** "
                                    "or **Across batteries** scheme and several test batteries."))
    elif view == "ops":
        if ss.get("m3_ops") is None:
            items.append(("tune", "Run the **operation study for every battery** to see which current and cut-off "
                                  "pays best across prices."))
        if not ss.get("m3_maint"):
            items.append(("build", "Run the **replacement study**: does the twin beat a fixed replacement age on the "
                                   "real NASA lives?"))
    elif view == "study":
        if not ss.get("study"):
            items.append(("science", "Press **Run cohort study**: it produces the evidence for your results chapter."))
        else:
            items.append(("description", "Export the **study report** and check groups with few cells (labelled underpowered)."))
    return items[:3]


def recommendations(view: str) -> None:
    items = _reco_items(view)
    if not items:
        return
    cards = "".join(f'<div class="bt-reco-item"><span class="bt-reco-dot"></span>{_md_bold(t)}</div>' for _, t in items)
    st.markdown(f'<div class="bt-reco"><div class="bt-reco-h">Recommended next steps</div>{cards}</div>',
                unsafe_allow_html=True)


def _md_bold(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)


def goto(view_label: str) -> None:
    st.session_state["view"] = view_label


GLOSSARY = {
    "SOH": "State of health: present capacity divided by the capacity when new (100% = new).",
    "EOL": "End of life: the capacity at which the battery is considered worn out (default 1.4 Ah).",
    "RUL": "Remaining useful life: cycles left until end of life.",
    "LLI": "Loss of lithium inventory: cyclable lithium trapped in side reactions (SEI growth, plating).",
    "LAM": "Loss of active material: electrode material that no longer stores lithium (cracking, dissolution).",
    "CL": "Conductivity loss: rising internal resistance (power fade).",
    "SEI": "Solid-electrolyte interphase: the film on graphite whose growth consumes lithium.",
    "EKF": "Extended Kalman filter: blends model predictions with measurements to update the twin.",
    "NIS": "Normalised innovation squared: about 1 when the filter's uncertainty is honest.",
    "ICA / DVA": "Incremental capacity (dQ/dV) and differential voltage (dV/dQ): curves whose peaks reveal ageing mechanisms.",
    "n₀": "Forecast origin: the last cycle a model may see before forecasting.",
    "Fade skill": "1 − error / error of assuming no further fade: 1 perfect, 0 useless, < 0 worse than doing nothing.",
    "Coverage": "Share of measurements falling inside a model's uncertainty band (should be close to the band level).",
    "PINN": "Physics-informed neural network: trained to fit data and obey physical equations.",
    "SPM": "Single-particle model: first-principles electrochemical model with one particle per electrode.",
}


def sidebar_guide() -> None:
    ss = st.session_state
    steps = [("Data loaded", True, V_DATA), ("Half-cell modes fitted", bool(ss.get("hc")), V_DATA),
             ("Live twin opened", bool(ss.get("ekf")), V_LIVE), ("Ladder models compared", bool(ss.get("ladder")), V_MODELS),
             ("M3 operation and replacement studies run", bool(ss.get("m3_maint")) and ss.get("m3_ops") is not None,
              V_OPS),
             ("Cohort study run", bool(ss.get("study")), V_STUDY)]
    done = sum(ok for _, ok, _ in steps)
    with st.sidebar:
        st.markdown(f'<div class="bt-guide"><div class="bt-guide-h">Study progress · {done}/{len(steps)}</div>'
                    f'<div class="bt-guide-bar"><div style="width:{100 * done / len(steps):.0f}%"></div></div></div>',
                    unsafe_allow_html=True)
        for k, (t, ok, view_label) in enumerate(steps):
            st.button(("✓ " if ok else "○ ") + t, key=f"guide_{k}", on_click=goto, args=(view_label,),
                      use_container_width=True, type="secondary", help="Go to the view where this step is done.")
        st.markdown("**Display**")
        st.toggle("Show explanations", value=True, key="ui_explain",
                  help="Off: explanations shrink to an ⓘ you can hover.")
        st.toggle("Export menus on charts", value=False, key="ui_export",
                  help="HTML / SVG / CSV export under each chart (off = faster, cleaner pages).")
        st.caption(":material/photo_camera: The camera icon at the top right of every chart saves it as a PNG image, "
                   "named after the chart, battery or group, title and date.")
        with st.expander("Glossary", icon=":material/menu_book:"):
            for k_, v in GLOSSARY.items():
                st.markdown(f"**{k_}**: {v}")


def view_update() -> None:
    """Mission 2 · how the twin's models should be updated (moved from the Live twin page in v5.26)."""
    update_frequency_section()
    frozen_update_section()


def view_home() -> None:
    section("Welcome: self-updating digital twin of Li-ion batteries")
    fs, ev = fleet_cached(ct, DATA_KEY, float(eol_ah), asdict(te.SafetyLimits()))
    groups = te.condition_groups(ct)
    k = st.columns(5)
    k[0].metric("Batteries", len(meta))
    k[1].metric("Discharge cycles", f"{int(ct['n'].count()):,}")
    k[2].metric("Condition groups", groups["Group"].nunique())
    k[3].metric("Critical batteries", int((fs["Risk"] == "Critical").sum()) if len(fs) else 0)
    k[4].metric("Logged events", len(ev))
    reps = []
    for g in te.GROUP_ORDER:
        members = groups[groups["Group"] == g]
        if len(members):
            c = members.sort_values("cycles", ascending=False).index[0]
            inf = battery_info(c, fs, groups)
            reps.append({"id": c, "soh": inf["soh"], "risk": inf["risk"], "sub": g})
    if reps:
        st.markdown("**One battery per operating condition**")
        render_battery_anim(reps[:7], big=False)
    study = st.session_state.get("study")
    status = {"M1 · Health & mechanisms": "answered" if study else "open",
              "M2 · Self-updating twin": "answered" if study and study["res"].get("update") is not None else
              ("partly answered" if study else "open"),
              "M3 · Operation & maintenance": "answered" if st.session_state.get("m3_life") else
              ("partly answered" if (st.session_state.get("m3_maint") or st.session_state.get("m3_ops") is not None)
               else "open")}
    chips = "".join(f'<div class="bt-mchip bt-m-{v.split()[0]}"><b>{m}</b> {html.escape(v)}</div>' for m, v in status.items())
    st.markdown(f'<div class="bt-mstrip">{chips}</div>', unsafe_allow_html=True)
    section("Where to start")
    cards = [("1 · Data & diagnostics", V_DATA, "Mission 1: one battery's health, ageing, mechanisms and operating stress."),
             ("2 · GA-PINN twin", V_TWIN, "Mission 2: the twin. Its state today, its forecast with uncertainty, why it ages."),
             ("3 · Model comparison", V_MODELS, "Mission 2: SOH estimation and forecasting, GA-PINN against the other models."),
             ("4 · Mission 3", V_OPS, "Best operation, operating window, when to replace, lifecycle benefit, and a summary."),
             ("5 · Study results", V_STUDY, "The answers to the three Missions, and the report export."),
             ("Appendix · EKF live twin", V_LIVE, "The Kalman-filter twin on [SOH, R_int, R_ct], replayed cycle by cycle."),
             ("Appendix · Model updating", V_UPDATE, "How often each twin must update, and how.")]
    cols = st.columns(len(cards))
    for c_, (t, v, d) in zip(cols, cards):
        with c_:
            st.markdown(f'<div class="bt-qs"><div class="bt-qs-t">{html.escape(t)}</div>'
                        f'<div class="bt-qs-d">{html.escape(d)}</div></div>', unsafe_allow_html=True)
            st.button("Open", key=f"home_{t}", on_click=goto, args=(v,), use_container_width=True, type="primary")
    with st.expander("How to use the app in 5 steps", expanded=False, icon=":material/route:"):
        st.markdown("1. **Choose a battery** and the end-of-life capacity in the control bar.\n"
                    "2. **Diagnostics → Overview**: health, grade and events; then *Ageing*, *Electrochemistry*, "
                    "*Stress & safety*.\n"
                    "3. **Live twin**: press Play, reveal the future, compare the live models.\n"
                    "4. **Models & forecasting**: pick a training scheme and run the ladder levels; read the leaderboard.\n"
                    "5. **Operations & control** for Mission 3, then **Study results** for the cohort-wide evidence.")


def view_data() -> None:
    global cell, ct_cell, eis_cell, c_bol, soh_eol
    recommendations("diag")
    lim = asdict(te.SafetyLimits())
    fs, ev = fleet_cached(ct, DATA_KEY, float(eol_ah), lim)
    all_cells = list(meta.index)
    labels = [DIAG_SCOPE_LABEL[k] for k in DIAG_SCOPES]
    st.session_state.setdefault("diag_scope", labels[0])
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    with c1:
        try:
            choice = st.segmented_control("Scope", labels, key="diag_scope", label_visibility="collapsed")
        except Exception:
            choice = st.radio("Scope", labels, key="diag_scope", horizontal=True, label_visibility="collapsed")
    choice = choice or labels[0]
    scope = DIAG_SCOPES[labels.index(choice)]
    c2.button("Select most critical", key="pick_crit", icon=":material/priority_high:", on_click=_pick_critical,
              args=(fs,), help="Switches to 'Selected batteries' with the six highest-risk cells.")
    if scope == "single":
        sel = [cell]
        st.caption(f"Analysing the target battery from the control bar: **{cell_label(cell, meta)}**.")
    elif scope == "selected":
        st.session_state.setdefault("diag_cells", [cell])
        sel = st.multiselect("Batteries", all_cells, key="diag_cells", max_selections=8,
                             format_func=lambda c: cell_label(c, meta),
                             help="Up to 8 batteries; every section below follows this selection.") or [cell]
    else:
        sel = all_cells
    st.session_state["_diag_sel"] = sel

    tab = sub_nav("diag_tab", DIAG_TABS)
    if len(sel) > 1 and tab != DIAG_TABS[0]:
        focus = st.selectbox("Focus battery for single-battery analyses on this tab", sel,
                             index=sel.index(cell) if cell in sel else 0, key="diag_focus",
                             format_func=lambda c: cell_label(c, meta),
                             help="Indicator trajectories, ICA / DVA, degradation modes, half-cell fitting and the "
                                  "stress timeline analyse one battery at a time (local to this tab).")
    else:
        focus = cell

    if tab == DIAG_TABS[0]:                                  # ---------------- Overview
        battery_hero(scope, sel, fs, ev)
        section("Health status")
        status_section(scope, sel, fs, ev, lim)
        bad = {c: e for c, e in cell_errors.items()}
        with st.expander(f"Data quality · {len(bad)} skipped cell(s)", icon=":material/fact_check:"):
            for c in sel[:8]:
                issues = validation_cached(store, DATA_KEY, c)
                n_out = int(ct[(ct["Cell_ID"] == c)]["outlier"].sum())
                (st.warning if issues else st.success)(
                    f"{c}: {len(issues)} telemetry issue(s), {n_out} excluded cycle(s)"
                    + ("" if not issues else " · " + "; ".join(issues[:3])))
            if len(sel) > 8:
                st.caption("Per-cell checks are listed for the first eight batteries.")
            if bad:
                show_table(pd.DataFrame({"Cell": list(bad), "Reason": list(bad.values())}).set_index("Cell"))
        return

    if focus != cell:                                        # analyses below follow the focus battery
        cell = focus
        ct_cell = ct[ct["Cell_ID"] == cell].sort_values("n")
        eis_cell = te.valid_eis(imp, cell)
        c_bol = float(meta.loc[cell, "C_bol_Ah"])
        soh_eol = te.soh_eol_for(c_bol, eol_ah)

    if tab == DIAG_TABS[1]:                                  # ---------------- Ageing
        section("Capacity fade: compare batteries")
        fade_explorer()
        section("Cohort by ambient temperature")
        c1, c2 = st.columns(2)
        norm_x = c1.toggle("Normalise x to fraction of recorded life", value=False, key="grid_norm",
                           help="Aligns cells with very different cycle counts.")
        grid_sel = sel if scope != "fleet" else [cell]
        amb = meta["Ambient_C"].round(0)
        groups = sorted(amb.dropna().unique())
        only_sel = scope != "fleet" and c2.toggle("Only groups containing the selection", value=False, key="grid_only")
        show_cells = st.multiselect("Batteries to show (empty = all) · local to this section", list(meta.index),
                                    default=[], key="grid_cells", format_func=lambda c: cell_label(c, meta),
                                    help="Choose which batteries appear in the temperature panels; panels without "
                                         "any chosen battery are hidden.")
        for g in groups:
            members = list(amb[amb == g].index)
            if show_cells:
                members = [c for c in members if c in show_cells]
                if not members:
                    continue
            if only_sel and not set(members) & set(grid_sel):
                continue
            with st.expander(f"Ambient {g:.0f} °C · {len(members)} cells", expanded=True,
                             icon=":material/thermostat:"):
                show(fig_cohort_group(ct, meta, members, grid_sel, P, f"Ambient {g:.0f} °C · {len(members)} cells",
                                      norm_x), key=f"cohort_{int(g)}",
                     data=ct.loc[~ct["outlier"] & ct["Cell_ID"].isin(members), ["Cell_ID", "n", "SOH"]])
        section("Mission 1 · Which parameter best represents health?")
        health_indicator_section()
    elif tab == DIAG_TABS[2]:                                # ---------------- Electrochemistry
        with st.expander("Incremental capacity analysis (dQ/dV)", expanded=True, icon=":material/show_chart:"):
            ica_section()
        with st.expander("Degradation modes: LLI · LAM · conductivity loss", expanded=False, icon=":material/layers:"):
            modes_section()
        with st.expander("Quantitative mode analysis: half-cell OCV fitting", expanded=False, icon=":material/science:"):
            half_cell_section()
    else:                                                    # ---------------- Stress & safety
        section("Operating stress and safety exposure")
        stress_section()


# =============================================================================
# View 2: models & forecasting
# =============================================================================
def methods_panel() -> None:
    with st.expander("Methods and references", expanded=False):
        st.markdown("**ML surrogate (increment strategy).** An autonomous fade-rate law learnt across the cohort "
                    "and integrated from the observed state at the forecast origin; x holds the operating plan "
                    "and early-life descriptors (early fade slope, early resistance growth).")
        st.latex(r"\frac{d\,\mathrm{SOH}}{dn} = g_\theta(\mathrm{SOH}, \mathbf{x}) \le 0,\qquad "
                 r"\widehat{\mathrm{SOH}}_{n+1} = \widehat{\mathrm{SOH}}_n + g_\theta(\widehat{\mathrm{SOH}}_n, \mathbf{x})")
        st.markdown("Bands: cross-cell split conformal, horizon-dependent half-width q(h) from calibration "
                    "cells forecast with the same protocol (valid under cell exchangeability).")
        st.markdown("**Dual time-scale ECM twin.** Per-cycle EKF on θ = [SOH, R_int, R_ct, log k]:")
        st.latex(r"\mathrm{SOH}_{k+1} = \mathrm{SOH}_k - e^{\log k}\, s(T_k)\, A_k,\quad "
                 r"R_{k+1} = R_k + \beta R_0\, e^{\log k} s(T_k) A_k,\quad \log k_{k+1} = \log k_k + w_k")
        st.latex(r"V_j = \mathrm{OCV}\!\left(1 - \tfrac{Q_j}{C_0\,\mathrm{SOH}}\right) + I_j R_\mathrm{int} + R_\mathrm{ct}\, g_j,"
                 r"\qquad s(T) = e^{\frac{E_a}{R}\left(\frac{1}{T_\mathrm{ref}} - \frac{1}{T}\right)}\,[1 + k_c (T_c - T)^+]")
        st.markdown("Voltage is used only over the first part of each discharge (a fixed fraction of the "
                    "beginning-of-life capacity), so the cut-off time never leaks capacity; the load-step "
                    "resistance and, optionally, the full capacity are extra measurements. Forecasts are Monte-Carlo "
                    "propagations of the posterior (SOH, log k).")
        st.markdown("**Hybrid PINN.** Network states with built-in initial conditions and soft physics residuals:")
        st.latex(r"\frac{d\,\mathrm{SOH}}{dn} = -k\, e^{\frac{E_a}{R}\left(\frac{1}{T_\mathrm{ref}}-\frac{1}{T}\right)}"
                 r"\, 2C_0\,\mathrm{SOH}\left(\frac{1-\mathrm{SOH}+\varepsilon}{L_\mathrm{ref}}\right)^{-m},\qquad "
                 r"\Delta V_\mathrm{step} = I(R_\mathrm{int}+R_x) + \frac{2RT}{F}\,\sinh^{-1}\!\frac{I F R_\mathrm{ct}}{2RT}")
        st.markdown("Eₐ is fixed or pooled across cells (ln rate vs 1/T regression with bootstrap CI): a single "
                    "cell at one temperature cannot identify it. Seed ensembles with randomised physical "
                    "initialisation report which parameters the data constrain.")
        st.markdown("**Semi-empirical law** (Wang et al. 2011), in throughput, fitted in log space with a cohort "
                    "ridge prior on the exponent:")
        st.latex(r"\mathrm{SOH} = 1 - B(T, I)\,\mathrm{Ah}^{z},\qquad z \approx 0.5\ \text{(SEI, diffusion-limited)},"
                 r"\quad z \approx 1\ \text{(linear)},\quad z > 1\ \text{(accelerating)}")
        st.markdown("**Hierarchical Bayes** (partial pooling): every cell has its own power-law parameters, drawn "
                    "from a fleet distribution whose mean depends on the operating conditions. The fleet is the prior "
                    "and the battery's own data update it (Laplace posterior):")
        st.latex(r"\mathrm{SOH} = s_0\left[1 - e^{a}\left(\tfrac{\mathrm{Ah}}{100}\right)^{z}\right],\qquad "
                 r"(a, z)_c \sim \mathcal{N}\!\left(G\,x_c,\ \Sigma\right),\quad x_c = \left[1,\ \tfrac{1}{T_\mathrm{ref}} - "
                 r"\tfrac{1}{T_c},\ \ln\tfrac{I_c}{2\,\mathrm{A}}\right]")
        st.markdown("**Particle filter** on the same physics law with the hierarchical prior (Student-t likelihood, "
                    "systematic resampling); **physics-mean GP**: hierarchical-Bayes trend plus a GP residual in "
                    "throughput; **stacked ensemble** weighted by backtest skill on the battery's own history:")
        st.latex(r"w_k^{(i)} \propto w_{k-1}^{(i)}\,p\!\left(y_k \mid \theta^{(i)}\right),\qquad "
                 r"\mathrm{SOH}(\mathrm{Ah}) = m_\theta(\mathrm{Ah}) + f(\mathrm{Ah}),\ f \sim \mathcal{GP}(0, k_\mathrm{RBF}),"
                 r"\qquad \hat y = \sum_i \frac{\mathrm{RMSE}_i^{-2}}{\sum_j \mathrm{RMSE}_j^{-2}}\,\hat y_i")
        st.markdown("**Health-indicator ranking** (Coble & Hines 2009): monotonicity, trendability, prognosability "
                    "and |Spearman ρ| with SOH on beginning-of-life-normalised indicators, plus a leave-one-cell-out "
                    "test; PCA decides whether one parameter suffices. **Degradation modes**: LLI / LAM / CL proxies "
                    "from ICA peak area, capacity and resistance (Birkl et al. 2017); DVA peak spacing. **Knee**: "
                    "two-segment piecewise-linear fit with an F-test.")
        st.markdown("**Mission 3** (rebuilt on the validated twin): lifecycle profit per hour of a battery life, with "
                    "J_op = revenue − energy cost from energy delivered and charged **measured** per tested operation "
                    "(inside each battery, from the raw discharge traces), ageing from the shared physics scaled to the "
                    "battery's measured fade, and J_maint = planned or unplanned replacement plus lead-time downtime. "
                    "Replacement policies are replayed on the **real measured lives** with only what the twin knew at "
                    "each cycle; every data-driven choice (safety margin, fixed age, operation) is made leave-one-out.")
        st.latex(r"\rho = \frac{\sum_{n \le t_\mathrm{rep}} \mathrm{SOH}_n\,(W_\mathrm{out}\,p - W_\mathrm{in}\,p_\mathrm{buy})"
                 r" - C_\mathrm{rep}\,(1 + (k_u - 1)\,\mathbb{1}_\mathrm{unplanned})}"
                 r"{\sum_{n \le t_\mathrm{rep}} h_n + L\,h\,\mathbb{1}_\mathrm{unplanned}}")
        st.markdown(
            "| Model class (Rufino Júnior et al. 2024, Fig. 1) | Implemented as |\n|---|---|\n"
            "| Empirical | Direct ML regression SOH(n) (baseline) |\n"
            "| Semi-empirical | Power law in Ah (Live twin) |\n"
            "| Equivalent-circuit | 1-RC ECM inside the dual EKF twin |\n"
            "| Electrochemical / physics-based | Butler–Volmer and SEI-type fade law in the hybrid PINN |\n"
            "| Data-driven | RF, GBR, GPR, SVR, MLP, Ridge fade-rate models with conformal bands |\n"
            "| Bayesian filtering / inference | Dual EKF (states + rate); particle filter and hierarchical "
            "Bayes on the physics power law; physics-mean GP |\n"
            "| Early-life data-driven | ΔQ(V) elastic net (Severson et al. 2019) |\n"
            "| Model combination | Skill-weighted stacked ensemble |")
        st.markdown("**Metrics.** RMSE on held-out cycles; RUL from the origin with right-censoring; relative "
                    "accuracy, α-λ and prognostic horizon after Saxena et al.; empirical band coverage.")
        st.markdown(
            "References: G. L. Plett, *J. Power Sources* 134 (2004) 252–292 (EKF for BMS, parts 1–3) · "
            "M. Dubarry et al., *J. Power Sources* 219 (2012) 204–216 (ICA degradation modes) · "
            "A. Saxena et al., *Int. J. PHM* 1 (2010) (prognostic metrics) · "
            "K. A. Severson et al., *Nature Energy* 4 (2019) 383–391 (early-life prediction) · "
            "M. Raissi et al., *J. Comput. Phys.* 378 (2019) 686–707 (PINNs) · "
            "A. N. Angelopoulos & S. Bates, arXiv:2107.07511 (conformal prediction) · "
            "B. Saha & K. Goebel, NASA Ames Prognostics Data Repository (battery data set) and *Annual Conf. PHM "
            "Society* (2009) (particle-filter prognostics) · "
            "C. A. Rufino Júnior et al., *Energies* 17 (2024) 3372 (degradation mechanisms, model taxonomy) · "
            "J. S. Menye, M.-B. Camara & B. Dakyo, *Energies* 18 (2025) 342 (degradation and failure mechanisms, "
            "LLI/LAM/CL, stress factors) · "
            "J. Wang et al., *J. Power Sources* 196 (2011) 3942–3948 (semi-empirical cycle-life model) · "
            "C. R. Birkl et al., *J. Power Sources* 341 (2017) 373–386 (degradation diagnostics) · "
            "J. Coble & J. W. Hines, *Annual Conf. PHM Society* (2009) (prognostic-parameter criteria).")


def hyperparam_editor(models: Sequence[str], prefix: str) -> Dict[str, Dict[str, Any]]:
    """One expander per selected model, widgets generated from te.MODEL_SPECS."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in models:
        spec = te.MODEL_SPECS[name]
        with st.expander(f"{name} · {spec.family}", expanded=False, icon=":material/tune:"):
            st.caption(spec.blurb)
            cols = st.columns(min(4, max(len(spec.params), 1)))
            vals: Dict[str, Any] = {}
            for i, h in enumerate(spec.params):
                k = f"{prefix}_{name}_{h.key}"
                c = cols[i % len(cols)]
                has = k in st.session_state                   # tuned or edited value already stored: no default
                if h.kind == "int":
                    if has:
                        st.session_state[k] = int(np.clip(int(st.session_state[k]), int(h.low), int(h.high)))
                    vals[h.key] = int(c.number_input(h.label, int(h.low), int(h.high), key=k, help=h.help or None,
                                                     **({} if has else {"value": int(h.default)})))
                elif h.kind == "float":
                    vals[h.key] = float(c.slider(h.label, float(h.low), float(h.high), key=k,
                                                 **({} if has else {"value": float(h.default)})))
                elif h.kind == "log":
                    extra = [float(f"{float(st.session_state[k]):.3g}")] if has else []
                    if has:
                        st.session_state[k] = extra[0]
                    grid = sorted(set([float(f"{v:.3g}") for v in np.geomspace(h.low, h.high, 25)] + [float(h.default)]
                                      + extra))
                    vals[h.key] = float(c.select_slider(h.label, grid, key=k, format_func=lambda v: f"{v:.3g}",
                                                        **({} if has else {"value": float(h.default)})))
                elif h.kind == "choice":
                    opts = list(h.options)
                    vals[h.key] = c.selectbox(h.label, opts, key=k, **({} if has else {"index": opts.index(h.default)}))
                else:
                    vals[h.key] = c.text_input(h.label, key=k, help=h.help or None,
                                               **({} if has else {"value": str(h.default)}))
            try:
                out[name] = te.validate_params(name, vals)
            except ValueError as exc:
                st.error(str(exc))
                out[name] = te.default_params(name)
    return out


R2_NOTE = ("**Reading the scores.** *Accuracy* = 100 × (1 − mean absolute percentage error). *Fade skill* compares "
           "the forecast with the naive assumption that SOH stops falling at n₀: 1 is perfect, 0 is no better than "
           "doing nothing, negative is worse. *R²* compares with the mean of the held-out SOH; over a short forecast "
           "window SOH hardly varies, so a small bias already makes R² negative. It is shown honestly rather than "
           "clipped, but fade skill and accuracy are the easier scores to read. In the estimation task the test set "
           "spans the whole ageing range and R² behaves as usual.")


def ml_section() -> None:
    section("Levels 1–3 · Machine-learning workbench (trees, forests, boosting)", icon=":material/model_training:")
    st.markdown("Two tasks, 8 curated models, every hyperparameter adjustable. **Forecast**: predict future SOH from the past "
                "(prognosis). **Estimate**: infer the present SOH from operando indicators measured on the same cycle "
                "(diagnosis, no capacity test needed).")
    avail = set(te.available_models())
    models = st.multiselect("Models (L1 simple → L3 boosting)", sorted(te.ML_MODELS, key=lambda m: te.MODEL_SPECS[m].level),
                            default=[m for m in ("Decision Tree", "Random Forest", "SVM", "XGBoost", "LightGBM")
                                     if m in avail],
                            key="ml_models",
                            format_func=lambda m: f"L{te.MODEL_SPECS[m].level} · {m}" + ("" if m in avail else " · not installed"),
                            help="Select any number; the leaderboard ranks them.")
    missing_opt = [n for n in te.OPTIONAL_ML if n not in avail]
    if missing_opt:
        st.warning("Not installed on this server: " + ", ".join(missing_opt) + ". They are listed in requirements.txt; "
                   "reboot the app after deploying it.")
    models = [m for m in models if m in avail]
    params = hyperparam_editor(models, "hp")
    t_fc, t_est = st.tabs([":material/trending_down: Forecast future SOH", ":material/biotech: Estimate SOH from indicators"])
    with t_fc:
        ml_forecast_tab(models, params)
    with t_est:
        ml_estimation_tab(models, params)
    with st.expander("How to read R², accuracy and fade skill", icon=":material/help:"):
        st.markdown(R2_NOTE)


def tuning_block(task: str, models: Sequence[str], params: Dict[str, Dict[str, Any]], run: Callable[[str], Any]
                 ) -> Dict[str, Dict[str, Any]]:
    """Auto-tune button + results; returns the parameters to use (tuned where available and enabled)."""
    store_key = f"tuned_{task}"
    tuned: Dict[str, Any] = st.session_state.setdefault(store_key, {})
    c1, c2, c3 = st.columns([1.4, 1.2, 2])
    n_iter = c2.select_slider("Candidates per model", [8, 12, 16, 24, 32], value=12, key=f"{task}_niter")
    if c1.button("Auto-tune hyperparameters", key=f"{task}_tune", icon=":material/auto_fix_high:",
                 disabled=not models, help="Random search validated without touching the test data."):
        prog = st.progress(0.0)
        for i, m in enumerate(models):
            prog.progress(i / len(models), text=f"Tuning {m}…")
            try:
                tuned[m] = run(m, int(n_iter))
            except Exception as exc:
                st.warning(f"{m}: tuning failed ({exc})")
        prog.empty()
    use = c3.toggle("Use tuned parameters", value=bool(tuned), key=f"{task}_use_tuned",
                    disabled=not tuned, help="Off = the values in the hyperparameter panels above.")
    if tuned:
        rows = [{"Model": m, "Validation (default)": r.default_score, "Validation (tuned)": r.best_score,
                 "Improvement (%)": r.improvement_pct, "Tuned settings": ", ".join(f"{k}={v:.3g}" if isinstance(v, float)
                                                                                    else f"{k}={v}"
                                                                                    for k, v in r.best_params.items()),
                 "Time (s)": r.seconds} for m, r in tuned.items() if m in models]
        if rows:
            with st.expander(f"Tuning results · {next(iter(tuned.values())).validation}", expanded=False,
                             icon=":material/insights:"):
                show_table(pd.DataFrame(rows).set_index("Model").style.format(
                    {"Validation (default)": "{:.4f}", "Validation (tuned)": "{:.4f}", "Improvement (%)": "{:+.0f}",
                     "Time (s)": "{:.1f}"}))
                st.caption("Score = mean + 0.5 SD of the validation RMSE (consistency across batteries is "
                           "rewarded). Defaults are always a candidate, so tuned ≤ default on validation. The "
                           "test data are never used, so tuned settings can still be worse on one particular "
                           "battery: judge them on the leaderboard with several test batteries.")
    out = dict(params)
    if use:
        out.update({m: r.best_params for m, r in tuned.items() if m in models})
    return out


def ml_v2_panel(res: Sequence[Any], target: str) -> None:
    """ML v2 results: forecasts, how much each horizon trusted its correction, and which measurements matter."""
    bl = getattr(res[0], "baseline", "trend")
    st.markdown(f"**ML v2 on {target}** · "
                f"{'fleet-shape baseline' if bl == 'shape' else 'straight-trend baseline'} + a learned correction")
    entries = [{"name": r.model, "level": te.MODEL_SPECS[r.model.split(" · ")[1]].level, "n_grid": r.n_grid,
                "soh": r.soh_pred, "lo": r.soh_lo, "hi": r.soh_hi, "m": r.metrics, "target": target} for r in res]
    if getattr(res[0], "baseline_curve", None) is not None:
        entries.append({"name": "Baseline alone (" + ("fleet shape" if bl == "shape" else "straight trend") + ")",
                        "level": 1, "n_grid": res[0].n_grid, "soh": res[0].baseline_curve, "lo": None, "hi": None,
                        "m": None, "target": target})
    trust = [np.nanmean([v for v in r.correction_weight.values() if np.isfinite(v)] or [0.0]) for r in res]
    if max(trust) < 0.1:
        note("The learned corrections did not beat the baseline on held-out batteries, so the forecasts follow the "
             "baseline. That is the honest outcome with few training batteries; the assessment below quantifies it.")
    show(fig_ladder(entries, res[0].n0, P, "ML v2 forecasts", target), key="ml_v2_fig", export=False)
    rows = []
    for r in res:
        w = {h: v for h, v in r.correction_weight.items() if np.isfinite(v)}
        rows.append({"Model": r.model, "Accuracy (%)": r.metrics.accuracy, "RMSE": r.metrics.rmse,
                     "CV RMSE (held-out batteries)": r.cv_rmse,
                     "Coverage": r.metrics.coverage, "Training batteries": len(r.train_cells),
                     "Correction trusted (short → long horizons)": " · ".join(f"{100 * v:.0f}%" for v in w.values()),
                     "Time (s)": r.fit_seconds})
    show_table(pd.DataFrame(rows).set_index("Model").style.format(
        {"Accuracy (%)": "{:.2f}", "RMSE": "{:.4f}", "CV RMSE (held-out batteries)": "{:.4f}", "Coverage": "{:.0%}",
         "Time (s)": "{:.1f}"}),
        note="'Correction trusted' is how much of the ML correction to the battery's own trend was kept at each "
             "horizon (+10 … +140 cycles), measured on held-out batteries: 0% means the trend alone was more "
             "reliable there. Coverage is the share of test cycles inside the 90% band.")
    ens = (st.session_state.get("ml_v2_ens") or {}).get(target)
    if ens is not None:
        wt = ens.params.get("weights", {})
        st.markdown(f"**ML ensemble (v2)** · accuracy {ens.metrics.accuracy:.2f}% · RMSE {ens.metrics.rmse:.4f} · weights: "
                    + ", ".join(f"{k.split(' · ')[1]} {v:.2f}" for k, v in wt.items()))
        note("The ensemble weights each model by its error on held-out batteries (never on this test battery), so "
             "the most reliable models count most.")
    imp = next((r.importance for r in res if len(r.importance)), None)
    if imp is not None:
        show(fig_importance(imp, P, res[0].model.split(" · ")[1]), key="ml_v2_imp", export=False)
    with st.expander("State features at the forecast origin", icon=":material/list:"):
        f = res[0].features
        show_table(pd.DataFrame({"Feature": [te.V2_FEATURES[k] for k in te.V2_FEATURES],
                                 "Value": [f.get(k, np.nan) for k in te.V2_FEATURES]}).set_index("Feature")
                   .style.format({"Value": "{:.4g}"}, na_rep="—"),
                   note="What v2 knows about the battery at the origin: levels and 10-cycle trends of the health "
                        "indicators, cumulative throughput and the operating conditions in physics form.")


def ml_forecast_tab(models: Sequence[str], params: Dict[str, Dict[str, Any]]) -> None:
    sc = scheme()
    show_t = sc["show"]
    n0_show = sc["n0"][show_t]
    if sc["mode"] == "within":
        source = "cohort" if sc["use_cohort"] else "own"
        st.markdown(f"**Training scheme:** within {show_t}: cycles 1–{n0_show} for training"
                    + (", plus the other batteries' full histories" if sc["use_cohort"] else " only")
                    + " (change it at the top of this view).")
    else:
        source = "cross"
        st.markdown(f"**Training scheme:** across batteries: learn from {len(sc['train'])} batteries, forecast "
                    f"{len(sc['targets'])} test battery(ies) (change it at the top of this view).")
    train_cells: Optional[Tuple[str, ...]] = tuple(sc["train"]) if source == "cross" else None
    d1, d2, d3 = st.columns(3)
    strategy = "increment"
    d1.markdown("**Fade-rate model**  \n*dSOH/dn = g(SOH, conditions, early-life features)*, integrated forward")
    conf = d2.slider("Conformal calibration cells (0 = no band)", 0, 8, 3, key="ml_conf")
    level = d3.select_slider("Band level", [0.8, 0.9, 0.95], value=0.9, key="ml_level")
    use_pop = source == "cohort"
    params = tuning_block(f"fc_{show_t}_{n0_show}_{source}", models, params,
                          lambda m, k: te.tune_ml_forecast(ct, show_t, n0_show, m, n_iter=k,
                                                           use_population=source != "own", train_cells=train_cells))
    ml_cfg = dict(cell=show_t, n0=n0_show, targets=tuple(sc["targets"]), models=tuple(models), source=source,
                  train_cells=train_cells, eol_ah=float(eol_ah), strategy=strategy, conformal_cells=conf,
                  band_level=level, params={m: params.get(m) for m in models})
    version = scheme().get("gen", "v2")
    if version in ("v2", "both"):
        st.radio("ML v2 baseline", ["shape", "trend"], horizontal=True, key="ml_baseline",
                 format_func={"shape": "Fleet shape (typical curve of similar batteries)",
                              "trend": "Straight recent trend"}.get,
                 help="ML v2 forecasts = baseline + a learned correction. The fleet shape bends where batteries of this "
                      "type bend; the straight trend continues the last cycles' slope. Compare both in the assessment.")
    st.caption({"v2": "Generation v2: state features at the origin, the battery's own trend + a learned, "
                      "cross-validated deviation.", "v1": "Generation v1: fade-rate law of SOH and conditions.",
                "both": "Both generations run; compare them on the leaderboard."}[version])
    ready = bool(models) and (source != "cross" or bool(train_cells))
    n_runs = len(models) * len(sc["targets"]) * (2 if version == "both" else 1)
    if st.button(f"Train and forecast ({n_runs} run{'s' if n_runs > 1 else ''})", type="primary", key="ml_go",
                 disabled=not ready, icon=":material/play_arrow:"):
        prog = st.progress(0.0)
        by_target: Dict[str, List[te.MLForecast]] = {}
        v2_by_target: Dict[str, List[Any]] = {}
        k_ = 0
        for t in sc["targets"]:
            for mname in models:
                prog.progress(k_ / max(n_runs, 1), text=f"Training {mname} for {t}…")
                k_ += 1
                lvl = te.MODEL_SPECS[mname].level if mname in te.MODEL_SPECS else 2
                if version in ("v1", "both"):
                    try:
                        r = ml_cached(ct, DATA_KEY, t, sc["n0"][t], mname, use_pop, float(eol_ah), strategy, conf, level,
                                      json.dumps(params.get(mname, {}), sort_keys=True), train_cells)
                        by_target.setdefault(t, []).append(r)
                        ladder_add(f"ML · {mname}", lvl, sc["n0"][t], r.n_grid, r.soh_pred, r.soh_lo, r.soh_hi,
                                   r.metrics, t)
                    except Exception as exc:
                        st.warning(f"{mname} (v1) on {t}: {exc}")
                if version in ("v2", "both"):
                    try:
                        v2 = ml_v2_cached(ct, DATA_KEY, t, sc["n0"][t], mname, float(eol_ah), level,
                                          train_cells if source == "cross" else (None if source == "cohort" else ()),
                                          json.dumps(params.get(mname, {}) if version == "v2" else {}, sort_keys=True),
                                          st.session_state.get("ml_baseline", "shape"))
                        v2_by_target.setdefault(t, []).append(v2)
                        ladder_add(v2.model, lvl, sc["n0"][t], v2.n_grid, v2.soh_pred, v2.soh_lo, v2.soh_hi,
                                   v2.metrics, t)
                    except Exception as exc:
                        st.warning(f"{mname} (v2) on {t}: {exc}")
        for t, mem in v2_by_target.items():                 # skill-weighted ensemble of the v2 models
            if len(mem) >= 2:
                try:
                    e = te.ml_v2_ensemble(mem, ct, float(eol_ah))
                    ladder_add(e.name, 3, sc["n0"][t], e.n_grid, e.soh, e.lo, e.hi, e.metrics, t)
                    st.session_state.setdefault("ml_v2_ens", {})[t] = e
                except Exception as exc:
                    st.warning(f"ML ensemble on {t}: {exc}")
        prog.empty()
        st.session_state["ml"] = {"cfg": ml_cfg, "res": by_target.get(show_t, []), "by_target": by_target}
        st.session_state["ml_v2"] = v2_by_target
    saved = st.session_state.get("ml")
    if saved and saved.get("by_target") and show_t in saved["by_target"] and saved["cfg"]["cell"] != show_t:
        saved = dict(saved, res=saved["by_target"][show_t], cfg=dict(saved["cfg"], cell=show_t, n0=sc["n0"][show_t]))
    v2_res = (st.session_state.get("ml_v2") or {}).get(show_t, [])
    if v2_res:
        ml_v2_panel(v2_res, show_t)
    saved = st.session_state.get("ml")
    if not (saved and saved["res"]):
        if not v2_res:
            placeholder("Press 'Train and forecast': each selected model's forecast, leaderboard and parity plot appear here.")
        return
    if saved["cfg"]["cell"] != show_t:
        st.info("The stored forecasts belong to another battery or scheme. Train again to update them.")
        return
    if {k: v for k, v in saved["cfg"].items() if k not in ("cell", "n0")} != \
            {k: v for k, v in ml_cfg.items() if k not in ("cell", "n0")}:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res = saved["res"]
    n0s = saved["cfg"]["n0"]
    ct_show = ct[ct["Cell_ID"] == show_t].sort_values("n")
    eol_show = te.soh_eol_for(float(meta.loc[show_t, "C_bol_Ah"]), float(eol_ah))
    data = pd.concat([pd.DataFrame({"model": r.model, "n": r.n_grid, "soh": r.soh_pred,
                                    "lo": r.soh_lo if r.soh_lo is not None else np.nan,
                                    "hi": r.soh_hi if r.soh_hi is not None else np.nan}) for r in res])
    show(fig_ml(res, ct_show, n0s, eol_show, P), key="ml_fig", data=data)
    tbl = pd.DataFrame([{"Model": r.model, "Accuracy (%)": r.metrics.accuracy, "Fade skill": r.metrics.fade_skill,
                         "R²": r.metrics.r2, "RMSE": r.metrics.rmse, "MAE": r.metrics.mae,
                         "Coverage": r.metrics.coverage, "RUL true": r.metrics.rul_true, "RUL pred": r.metrics.rul_pred,
                         "RUL error": r.metrics.rul_error, "Fit time (s)": r.fit_seconds} for r in res]).set_index("Model")
    best = _best_key(tbl, "RMSE", lowest=True) or tbl.index[0]
    k = st.columns(4)
    k[0].metric("Best model", best)
    k[1].metric("Accuracy", fmt(tbl.loc[best, "Accuracy (%)"], ".2f", "%"))
    k[2].metric("Fade skill", fmt(tbl.loc[best, "Fade skill"], ".3f"))
    k[3].metric("RMSE", fmt(tbl.loc[best, "RMSE"], ".4f"))
    show(fig_leaderboard(tbl, "Fade skill", P, "Leaderboard: fade skill on the held-out cycles (higher is better)"),
         key="ml_board", data=tbl.reset_index())
    show_table(tbl.style.format(
        {"Accuracy (%)": "{:.2f}", "Fade skill": "{:.3f}", "R²": "{:.3f}", "RMSE": "{:.4f}", "MAE": "{:.4f}",
         "Coverage": "{:.2f}", "RUL true": "{:.0f}", "RUL pred": "{:.0f}", "RUL error": "{:+.0f}",
         "Fit time (s)": "{:.2f}"}, na_rep="—")
        .highlight_min(subset=["RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    note("Forecast scores of each ML model on the cycles after the origin: accuracy, fade skill, R², errors, band coverage and remaining-life error.")
    good = ct_show[(~ct_show["outlier"]) & (ct_show["n"] > n0s)]
    par = pd.concat([pd.DataFrame({"model": r.model, "SOH": good["SOH"].to_numpy(),
                                   "SOH_pred": np.interp(good["n"], r.n_grid, r.soh_pred)}) for r in res])
    if len(par):
        show(fig_parity(par, "model", P, "Parity on the test cycles: forecast vs measured SOH"), key="ml_parity",
             data=par)
    cal = res[0].calibration_cells
    if cal:
        st.caption(f"Conformal calibration cells: {', '.join(cal)}. Coverage below the nominal level signals that "
                   "this battery ages differently from its calibration partners.")
    manifest_button(ml_cfg, "ml", DATA_KEY)


def ml_estimation_tab(models: Sequence[str], params: Dict[str, Dict[str, Any]]) -> None:
    all_cells = list(meta.index)
    avail = [k for k in te.HI_CATALOG if k not in te.CAPACITY_LEAKS and k in ct.columns or k in ("Re_ohm", "Rct_ohm")]
    avail += [k for k in ("T_mean_C", "I_dis_A", "n") if k in ct.columns]
    labels = {**{k: te.HI_CATALOG[k].label for k in te.HI_CATALOG}, "T_mean_C": "Mean cell temperature",
              "I_dis_A": "Discharge current", "n": "Cycle number"}
    c1, c2 = st.columns([3, 1])
    feats = c1.multiselect("Input indicators (capacity-derived ones are excluded: they would leak SOH)", avail,
                           default=[k for k in te.DEFAULT_EST_FEATURES if k in avail], key="est_feats",
                           format_func=lambda k: labels.get(k, k))
    normalise = c2.toggle("Normalise to beginning of life", value=True, key="est_norm",
                          help="Divides each indicator by its early-life value, so models transfer between batteries.")
    d1, d2 = st.columns(2)
    split = d1.radio("Train / test split", list(te.ESTIMATION_SPLITS), key="est_split",
                     format_func={"random": "Random cycles (interpolation, optimistic)",
                                  "chronological": "Late life of every battery (extrapolation in time)",
                                  "by_cell": "Train on some batteries, test on others (transfer)"}.get)
    test_frac = d2.slider("Test fraction", 0.1, 0.8, 0.3, 0.05, key="est_frac", disabled=split == "by_cell")
    tr_cells: Tuple[str, ...] = ()
    te_cells: Tuple[str, ...] = ()
    if split == "by_cell":
        e1, e2 = st.columns(2)
        te_cells = tuple(e2.multiselect("Test batteries", all_cells, default=[cell], key="est_test_cells",
                                        format_func=lambda c: cell_label(c, meta)))
        rest = [c for c in all_cells if c not in te_cells]
        tr_cells = tuple(e1.multiselect("Train batteries", rest, default=rest, key="est_train_cells",
                                        format_func=lambda c: cell_label(c, meta)))
    params = tuning_block(f"est_{split}_{test_frac}_{hash((tr_cells, te_cells, tuple(feats))) % 10**6}", models, params,
                          lambda m, k: te.tune_soh_estimator(ct, imp, m, tuple(feats), split, float(test_frac),
                                                             list(tr_cells) or None, list(te_cells) or None, normalise,
                                                             n_iter=k))
    cfg = dict(models=tuple(models), feats=tuple(feats), split=split, test_frac=test_frac, train=tr_cells,
               test=te_cells, normalise=normalise, params={m: params.get(m) for m in models})
    ready = bool(models) and bool(feats) and (split != "by_cell" or (tr_cells and te_cells))
    if st.button("Train estimators", type="primary", key="est_go", disabled=not ready, icon=":material/play_arrow:"):
        prog = st.progress(0.0)
        out: Dict[str, te.EstimationResult] = {}
        for i, mname in enumerate(models):
            prog.progress(i / max(len(models), 1), text=f"Training {mname}…")
            try:
                out[mname] = est_cached(ct, imp, DATA_KEY, mname, json.dumps(params.get(mname, {}), sort_keys=True),
                                        tuple(feats), split, float(test_frac), tr_cells, te_cells, normalise)
            except Exception as exc:
                st.warning(f"{mname}: {exc}")
        prog.empty()
        st.session_state["est"] = {"cfg": cfg, "res": out}
    saved = st.session_state.get("est")
    if not (saved and saved["res"]):
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res: Dict[str, te.EstimationResult] = saved["res"]
    tbl = pd.DataFrame([{"Model": m, "Test R²": r.metrics.loc["test", "R²"], "Test RMSE": r.metrics.loc["test", "RMSE"],
                         "Test accuracy (%)": r.metrics.loc["test", "Accuracy (%)"],
                         "Train R²": r.metrics.loc["train", "R²"], "Train RMSE": r.metrics.loc["train", "RMSE"],
                         "Overfit gap (RMSE)": r.metrics.loc["test", "RMSE"] - r.metrics.loc["train", "RMSE"],
                         "Fit time (s)": r.fit_seconds} for m, r in res.items()]).set_index("Model")
    best = _best_key(tbl, "Test RMSE", lowest=True) or tbl.index[0]
    k = st.columns(4)
    k[0].metric("Best model", best)
    k[1].metric("Test R²", fmt(tbl.loc[best, "Test R²"], ".3f"))
    k[2].metric("Test accuracy", fmt(tbl.loc[best, "Test accuracy (%)"], ".2f", "%"))
    k[3].metric("Test RMSE", fmt(tbl.loc[best, "Test RMSE"], ".4f"))
    show(fig_leaderboard(tbl, "Test R²", P, "Leaderboard: R² on the test set (higher is better)"), key="est_board",
         data=tbl.reset_index())
    show_table(tbl.style.format({"Test R²": "{:.3f}", "Test RMSE": "{:.4f}", "Test accuracy (%)": "{:.2f}",
                                 "Train R²": "{:.3f}", "Train RMSE": "{:.4f}", "Overfit gap (RMSE)": "{:+.4f}",
                                 "Fit time (s)": "{:.2f}"}, na_rep="—")
               .highlight_min(subset=["Test RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    note("Estimation scores on the test cycles and on the training cycles: a large gap between them means the model memorises instead of learning.")
    pick = st.selectbox("Inspect model", list(res), index=list(res).index(best), key="est_pick")
    r = res[pick]
    test = r.predictions[r.predictions["set"] == "test"]
    show(fig_parity(test.assign(model=pick), "Cell_ID", P, f"{pick}: estimated vs measured SOH on the test set",
                    all_cells), key="est_parity", data=test)
    show(fig_est_traj(r.predictions[r.predictions["Cell_ID"].isin(test["Cell_ID"].unique())], P, pick, all_cells),
         key="est_traj", data=r.predictions)
    show(fig_importance(r.importance, P, pick), key="est_imp", data=r.importance)
    st.caption(f"Train batteries: {', '.join(r.train_cells)} · Test batteries: {', '.join(r.test_cells)}. "
               "A large overfit gap (test RMSE ≫ train RMSE) means the model memorises; raise regularisation or "
               "reduce depth in its hyperparameter panel.")
    manifest_button(cfg, "estimation", DATA_KEY)


def pinn_equations_panel() -> None:
    with st.expander("Governing equations of the mechanism-resolved PINN", icon=":material/functions:"):
        st.markdown("The network $\\mathcal{N}(n)$ outputs three latent capacity losses (fractions of the "
                    "beginning-of-life capacity) and two resistances, each with its initial condition built in. "
                    "Collocation points span 1.5× the horizon, so the forecast obeys the kinetics beyond the data. "
                    "$A(T;E) = \\exp\\left[\\tfrac{E}{R}\\left(\\tfrac{1}{T_\\mathrm{ref}} - \\tfrac{1}{T}\\right)\\right]$, "
                    "$A_n = 2\\,C_0\\,\\mathrm{SOH}$ is the charge throughput of cycle $n$.")
        st.markdown("**1 · Capacity balance** (loss of lithium inventory = SEI + plating; loss of active material)")
        st.latex(r"\mathrm{SOH}(n) = 1 - Q_\mathrm{SEI}(n) - Q_\mathrm{pl}(n) - Q_\mathrm{LAM}(n)")
        st.markdown("**2 · SEI growth**: solvent reduction at the graphite surface, reaction-limited while the film "
                    "is thin and diffusion-limited as it thickens (Ploehn 2004; Pinson & Bazant 2013)")
        st.latex(r"\frac{dQ_\mathrm{SEI}}{dn} = k_\mathrm{SEI}\,A(T;E_\mathrm{SEI})\,\frac{A_n}{C_0}\,"
                 r"\frac{1}{1 + Q_\mathrm{SEI}/\delta}\;\;\Rightarrow\;\; Q_\mathrm{SEI} \propto n\ (\text{thin}),"
                 r"\quad \propto \sqrt{n}\ (\text{thick})")
        st.markdown("**3 · Lithium plating**: favoured by cold and by charge current, and triggered at any "
                    "temperature once LAM clogs the pores (Waldmann 2014; Yang et al. 2017), which produces the knee")
        st.latex(r"\frac{dQ_\mathrm{pl}}{dn} = k_\mathrm{pl}\,e^{\frac{E_\mathrm{pl}}{R}\left(\frac1T - "
                 r"\frac1{T_\mathrm{ref}}\right)}\,\frac{I_\mathrm{ch}}{C_0}\left[g_\mathrm{cold}(T) + "
                 r"\kappa\,\frac{Q_\mathrm{LAM}}{0.05}\right],\qquad g_\mathrm{cold} = "
                 r"\frac{1}{1 + e^{(T - T_\mathrm{onset})/3\,\mathrm{K}}}")
        st.markdown("**4 · Loss of active material**: particle cracking and dissolution, driven by C-rate and "
                    "self-accelerating as the remaining material carries more current (Laresgoiti 2015)")
        st.latex(r"\frac{dQ_\mathrm{LAM}}{dn} = k_\mathrm{LAM}\,A(T;E_\mathrm{LAM})\left(\frac{I}{C_0}\right)^{\beta}"
                 r"\frac{A_n}{C_0}\left(1 + \frac{Q_\mathrm{LAM}}{\varepsilon}\right)")
        st.markdown("**5 · Resistance from the mechanisms**: the SEI film adds ohmic resistance, while active-area "
                    "loss and surface films raise the charge-transfer resistance")
        st.latex(r"\frac{dR_\mathrm{int}}{dn} = R_\mathrm{int,0}\,\rho_\mathrm{SEI}\,\frac{1}{0.1}\frac{dQ_\mathrm{SEI}}{dn},"
                 r"\qquad \frac{dR_\mathrm{ct}}{dn} = R_\mathrm{ct,0}\,\frac{1}{0.1}\left(\gamma_\mathrm{LAM}"
                 r"\frac{dQ_\mathrm{LAM}}{dn} + \gamma_\mathrm{SEI}\frac{dQ_\mathrm{SEI}}{dn}\right)")
        st.markdown("**6 · Electrochemistry**: Butler–Volmer charge transfer (α = 0.5) with Arrhenius kinetics, "
                    "constraining the load-step voltage drop and the mean discharge voltage")
        st.latex(r"\eta_\mathrm{ct} = \frac{2RT}{F}\,\sinh^{-1}\!\left(\frac{I\,F\,R_\mathrm{ct}(T)}{2RT}\right),\qquad "
                 r"R_\mathrm{ct}(T) = R_\mathrm{ct}\,e^{\frac{E_\mathrm{ct}}{R}\left(\frac1T - \frac1{T_\mathrm{ref}}\right)},"
                 r"\qquad i_0 = \frac{RT}{F\,R_\mathrm{ct}}")
        st.latex(r"\Delta V_\mathrm{step} = I\,(R_\mathrm{int} + R_x) + \eta_\mathrm{ct},\qquad "
                 r"\bar V_\mathrm{dis} = \bar U - I\,(R_\mathrm{int} + R_x) - \eta_\mathrm{ct}")
        st.markdown("**7 · Loss**")
        st.latex(r"\mathcal{L} = \lambda_\mathrm{d}\,\mathcal{L}_\mathrm{SOH} + \lambda_\mathrm{p}\sum_{m}\|r_m\|^2 "
                 r"+ \lambda_\mathrm{BV}\,\mathcal{L}_{\Delta V} + \lambda_V\,\mathcal{L}_{\bar V} + "
                 r"\lambda_\mathrm{EIS}\,\mathcal{L}_\mathrm{EIS} + \lambda_\pi \sum_m \left(\frac{\ln k_m - "
                 r"\ln k_m^0}{2}\right)^2")
        st.caption("Weak log-normal priors on the rate constants keep the mechanism split identifiable on one "
                   "cell; with several ensemble members the identifiability table shows which parameters the data "
                   "actually pin down. From capacity alone, SEI and LAM are only partly separable. EIS and "
                   "voltage terms help, and plating is only resolved on cold or knee-bearing cells.")


def ablation_section() -> None:
    section("Does voltage feedback add information? Measurement ablation")
    st.markdown("The dual twin is re-run with measurement subsets. *Open loop* uses only the cohort prior; "
                "the gap between it and the voltage-only observer is the information carried by the voltage signal.")
    frac = st.slider("Forecast origin for the ablation forecasts", 0.2, 0.8, 0.4, 0.05, key="abl_frac")
    n0 = int(max(5, round(frac * meta.loc[cell, "cycles"])))
    cfg = dict(cell=cell, n0=n0, eol_ah=float(eol_ah))
    if st.button("Run ablation", key="abl_go"):
        with st.spinner("Running four observer configurations…"):
            try:
                tab, results = te.twin_ablation(cell_frame(store, DATA_KEY, cell), ct, imp, cell, te.TwinParameters(),
                                                None, te.similar_cells(meta, cell), n0, soh_eol)
                st.session_state["abl"] = {"cfg": cfg, "tab": tab, "res": results}
            except Exception as exc:
                report_error("Ablation failed", exc, debug)
    saved = st.session_state.get("abl")
    if saved and saved["cfg"]["cell"] == cell:
        if saved["cfg"] != cfg:
            st.warning("Settings changed since the last run; the results below use the previous settings.")
        show(fig_ablation(saved["res"], ct_cell, P), key="abl_fig", data=saved["tab"].reset_index())
        show_table(saved["tab"].style.format({"Tracking RMSE": "{:.4f}", "Tracking 90% coverage": "{:.2f}",
                                              "Mean NIS / dof": "{:.2f}", "k personalised / prior": "{:.2f}",
                                              "Forecast RMSE": "{:.4f}", "RUL error": "{:+.0f}",
                                              "Forecast coverage": "{:.2f}", "Info gain SOH (nats/100 cyc)": "{:.2f}",
                                              "Info gain log k (nats/100 cyc)": "{:.2f}"}, na_rep="—"))
        st.caption("Information gain = entropy reduction of the SOH and degradation-rate posteriors per 100 cycles: "
                   "the measurement set with the largest gain is the most informative (Mission 2). "
                   "Tracking RMSE compares the causal estimate with measured SOH over the whole life. When "
                   "capacity is among the measurements this is partly circular; the operando rows are the fair test.")


def cross_cell_section() -> None:
    section("Cross-cell benchmark: forecast-origin sweep")
    st.markdown("A single cell at a single origin is an anecdote. This section evaluates every paradigm on every "
                "cell at several origins. Run the full sweep offline with `python benchmark.py` "
                "(writes `results/benchmark.parquet`) or a quick subset here.")
    src = st.radio("Source", ["Run a quick sweep here", "Load results file"], horizontal=True, key="bench_src")
    if src == "Load results file":
        default = RESULTS_DIR / "benchmark.parquet"
        up_b = st.file_uploader("Benchmark results (.parquet / .csv)", type=["parquet", "csv"], key="up_bench")
        up_r = st.file_uploader("Residuals (optional, *_residuals)", type=["parquet", "csv"], key="up_resid")
        try:
            if up_b is not None:
                bench = read_uploaded_table(up_b)
                resid = read_uploaded_table(up_r) if up_r is not None else None
                st.session_state["bench"] = {"bench": bench, "resid": resid, "cfg": {"source": up_b.name}}
            elif default.exists() and "bench" not in st.session_state:
                bench = load_bench_file(str(default), default.stat().st_mtime)
                rp = RESULTS_DIR / "benchmark_residuals.parquet"
                resid = load_bench_file(str(rp), rp.stat().st_mtime) if rp.exists() else None
                st.session_state["bench"] = {"bench": bench, "resid": resid, "cfg": {"source": str(default)}}
        except Exception as exc:
            report_error("Could not read the benchmark file", exc, debug)
    else:
        c1, c2, c3 = st.columns(3)
        pool = [c for c in cells if int(meta.loc[c, "cycles"]) >= 30]
        sel = c1.multiselect("Cells", pool, default=pool[: min(6, len(pool))], key="bench_cells")
        fracs = c2.multiselect("Origins (fraction of life)", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], default=[0.3, 0.5],
                               key="bench_fracs")
        pars = c3.multiselect("Paradigms", [p_ for p_ in te.BENCH_PARADIGMS if p_ != "PINN"],
                              default=[p_ for p_ in ("ML", "Twin", "HB") if p_ in te.BENCH_PARADIGMS],
                              key="bench_pars", format_func={"ML": "ML surrogate", "Twin": "ECM twin (dual EKF)",
                                                             "PINN": "Hybrid PINN", "SemiEmp": "Semi-empirical",
                                                             "PF": "Particle filter", "HB": "Hierarchical Bayes",
                                                             "GP": "Physics-mean GP"}.get,
                              help="The PINN adds ~seconds per cell and origin.")
        cfg = te.BenchmarkConfig(fracs=tuple(sorted(fracs)) or (0.4,), paradigms=tuple(pars) or ("Twin",),
                                 eol_ah=float(eol_ah), pinn_epochs=500, conformal_cells=3)
        est = len(sel) * len(cfg.fracs) * (0.5 + 1.5 * ("ML" in pars) + 3 * ("PINN" in pars) + 0.5 * ("PF" in pars))
        if st.button(f"Run sweep (≈ {est:.0f} s)", key="bench_go", disabled=not sel):
            bar = st.progress(0.0, text="Starting…")
            try:
                bench, resid = te.run_benchmark(store, ct, imp, cfg, cells=sel,
                                                progress=lambda f, m: bar.progress(f, text=m))
                st.session_state["bench"] = {"bench": bench, "resid": resid, "cfg": asdict(cfg) | {"cells": sel}}
            except Exception as exc:
                report_error("Benchmark sweep failed", exc, debug)
            finally:
                bar.empty()

    saved = st.session_state.get("bench")
    if not saved:
        return
    bench, resid = saved["bench"], saved["resid"]
    alpha = float(saved["cfg"].get("alpha", 0.2)) if isinstance(saved["cfg"], dict) else 0.2
    level = float(saved["cfg"].get("band_level", 0.9)) if isinstance(saved["cfg"], dict) else 0.9
    if "error" in bench.columns and bench["error"].notna().any():
        with st.expander(f"{int(bench['error'].notna().sum())} failed forecast(s)"):
            show_table(bench[bench["error"].notna()][["cell", "frac", "paradigm", "error"]])
    summ = te.benchmark_summary(bench, alpha)
    if summ.empty:
        st.warning("No successful forecasts in these results.")
        return
    show_table(summ.style.format({"Median RMSE": "{:.4f}", "Mean RMSE": "{:.4f}", "Mean RA": "{:.2f}",
                                  "α-λ hit rate": "{:.2f}", "Mean coverage": "{:.2f}", "Mean band width": "{:.4f}",
                                  "Mean PH (cycles)": "{:.0f}"}, na_rep="—"))
    a = b = st.container()  # stacked full width
    with a:
        show(fig_bench_parity(bench, alpha, P), key="bench_parity", data=_bench_ok(bench))
    with b:
        show(fig_alpha_lambda(bench, alpha, P), key="bench_al", data=_bench_ok(bench))
    if resid is not None and len(resid):
        cov = te.coverage_by_horizon(resid)
        show(fig_horizon(cov, level, P), key="bench_horizon", data=cov)
    with st.expander("Prognostic horizon per cell"):
        show_table(te.prognostic_horizon(_bench_ok(bench), alpha).pivot(index="cell", columns="paradigm",
                                                                         values="PH_cycles"))
    download("Benchmark rows (CSV)", bench.to_csv(index=False), "benchmark.csv", "text/csv", key="dl_bench")
    manifest_button(saved["cfg"] if isinstance(saved["cfg"], dict) else {}, "benchmark", DATA_KEY)


def update_frequency_section() -> None:
    section("Mission 2 · How often should the twin update?")
    st.markdown("The dual twin assimilates measurements only every *m*-th discharge and runs open loop on its "
                "calibrated fade law in between. The recommended interval is the sparsest schedule whose tracking "
                "error stays within 25% of updating every cycle or below 0.5% SOH (about the repeatability of a "
                "capacity measurement): it sets the diagnostic cost of the twin in operation.")
    c1, c2 = st.columns(2)
    ivals = c1.multiselect("Update intervals (cycles)", [1, 2, 3, 5, 10, 20, 50], default=[1, 2, 5, 10, 20, 50],
                           key="uf_ivals")
    frac = c2.slider("Forecast origin", 0.2, 0.8, 0.4, 0.05, key="uf_frac")
    n0 = int(max(5, round(frac * meta.loc[cell, "cycles"])))
    cfg = dict(cell=cell, n0=n0, intervals=sorted(ivals), eol_ah=float(eol_ah))
    if st.button("Run update-frequency study", key="uf_go", disabled=not ivals):
        with st.spinner("Replaying the twin with sparse updates…"):
            try:
                tab, results = te.update_frequency_study(cell_frame(store, DATA_KEY, cell), ct, imp, cell,
                                                         te.TwinParameters(), None, n0, soh_eol,
                                                         intervals=tuple(sorted(ivals)))
                st.session_state["uf"] = {"cfg": cfg, "tab": tab, "res": results}
            except Exception as exc:
                report_error("Update-frequency study failed", exc, debug)
    saved = st.session_state.get("uf")
    if saved and saved["cfg"]["cell"] == cell:
        if saved["cfg"] != cfg:
            st.warning("Settings changed since the last run; the results below use the previous settings.")
        tab = saved["tab"]
        show(fig_update_freq(tab, saved["res"], ct_cell, P), key="uf_fig", data=tab.reset_index())
        show_table(tab.style.format({"Updates per 100 cycles": "{:.0f}", "Tracking RMSE": "{:.4f}",
                                     "Worst |error|": "{:.4f}", "Tracking 90% coverage": "{:.2f}",
                                     "Info gain per update (nats)": "{:.3f}", "Forecast RMSE": "{:.4f}",
                                     "RUL error": "{:+.0f}", "Forecast coverage": "{:.2f}"}, na_rep="—"))
        rec = tab.attrs.get("recommended")
        if rec:
            card("Mission 2 answer", [f"Recommended update interval for {cell}: every {rec} discharge cycle(s) "
                                      f"({tab.loc[rec, 'Updates per 100 cycles']:.0f} updates per 100 cycles, tracking "
                                      f"RMSE {tab.loc[rec, 'Tracking RMSE']:.4f} vs "
                                      f"{tab['Tracking RMSE'].iloc[0]:.4f} when updating every cycle)",
                                      "Information gain per update rises as updates get sparser (each measurement "
                                      "corrects more drift), but total information per 100 cycles falls."])
        manifest_button(cfg, "update_frequency", DATA_KEY)


FZ_ANCHOR_COLORS = {"none": "#999999", "capacity": "#0072B2", "estimate": "#D55E00"}


def fig_frozen_update(summ: pd.DataFrame, P: Palette, title: str) -> go.Figure:
    """Forecast error against update cost: one line per anchor, one point per retraining interval."""
    fig = go.Figure()
    for a_key, d in summ.groupby("anchor_key", sort=False):
        d = d.sort_values("Refits per 100 cycles")
        fig.add_trace(go.Scatter(
            x=d["Refits per 100 cycles"], y=d["Forecast RMSE (SOH pts)"], mode="lines+markers+text",
            name=te.FROZEN_ANCHORS[a_key].split(" (")[0], text=[f"{v}" for v in d["Retrain every"]],
            textposition="top center", line=dict(color=FZ_ANCHOR_COLORS.get(a_key, P.text), width=2.4),
            marker=dict(size=9), hovertemplate="retrain every %{text}: %{y:.2f} SOH pts at %{x:.1f} refits / 100 "
                                                "cycles<extra>" + html.escape(te.FROZEN_ANCHORS[a_key]) + "</extra>"))
    fig.update_xaxes(title_text="Retrainings per 100 cycles (cost)")
    fig.update_yaxes(title_text="Forecast RMSE, next cycles (SOH pts)", rangemode="tozero")
    return style_fig(fig, P, 440, title)


def frozen_update_section() -> None:
    section("Mission 2 · How often should the frozen model update?")
    st.markdown("The frozen assessment winner of the battery's group runs live from 30 % of life (the assessment's "
                "origin). Two kinds of update are compared: **retraining** every N cycles (on the group's other "
                "batteries plus this battery's history up to now) and **re-anchoring** every cycle, which shifts the "
                "forecast to the current SOH, taken from the measured capacity (lab) or estimated by the frozen "
                "SOH-now model (field: no capacity measured after the start). 'Never retrain, no anchor' is exactly the "
                "assessment forecast. Scores are causal: each forecast is compared only with cycles measured after it.")
    grp_c, fz, train_c = frozen_context(cell, core=TWIN_CORES[0])   # Mission 2 study: always the frozen model
    why_not = frozen_unavailable_reason(cell, grp_c, fz, train_c)
    if why_not:
        st.info(why_not + " Select a battery of another group in the control bar.", icon=":material/info:")
        return
    st.caption(f"{grp_c} group · forecast: {model_label(fz[0])} · SOH-now: {model_label(fz[1])} · "
               f"default settings.")
    c1, c2, c3, c4 = st.columns([1.4, 1.6, 0.8, 1.3])
    ivals = c1.multiselect("Retrain every (cycles)", [5, 10, 20, 50, 0], default=[5, 10, 20, 50, 0], key="fzu_ivals",
                           format_func=lambda v: "never" if v == 0 else str(v))
    anchors = c2.multiselect("Re-anchor to", list(te.FROZEN_ANCHORS), default=list(te.FROZEN_ANCHORS), key="fzu_anch",
                             format_func=lambda k_: te.FROZEN_ANCHORS[k_].split(" (")[0])
    horizon = c3.selectbox("Score the next", [10, 20, 40], index=1, key="fzu_h", format_func=lambda v: f"{v} cycles")
    scope = c4.radio("Batteries", [f"This battery ({cell})", f"Whole {grp_c} group"], key="fzu_scope")
    cells_run = [cell] if scope.startswith("This") else [cell] + train_c
    # same data rule as the accuracy assessment, whatever the control-bar toggle says: Mixed conditions at each
    # battery's reference condition (capacity jumps at load / temperature changes removed), every other group as recorded
    if grp_c == "Mixed conditions":
        ct_s, dkey_s = ref_ct_cached(CT_RAW, DATA_KEY_RAW), DATA_KEY_RAW + "|reference-condition"
        st.caption("Mixed conditions: scored at each battery's reference condition, as in the accuracy assessment "
                   "(applied automatically; the control-bar toggle does not change this study).")
    else:
        ct_s, dkey_s = CT_RAW, DATA_KEY_RAW
    cfg = (dkey_s, grp_c, tuple(sorted(cells_run)), tuple(sorted(ivals)), tuple(anchors), int(horizon), float(eol_ah))
    n_fast = min([v for v in ivals if v] or [50])
    st.caption(f"{len(cells_run)} battery(ies) × {len(ivals) * len(anchors)} settings. Fits are shared between "
               f"settings, so the cost is set by retraining every {n_fast} cycles (the forecast model is refitted about "
               f"{max(1, round(70 / n_fast))} times per 100 cycles, twice when field mode is included). Finished "
               "batteries are kept: an interrupted run continues where it stopped.")
    if st.button("Run the frozen-model update study", key="fzu_go", type="primary", icon=":material/update:",
                 disabled=not ivals or not anchors):
        prog = st.progress(0.0)
        parts = []
        groups = te.condition_groups(CT_RAW)
        for i, c in enumerate(cells_run):
            prog.progress(i / len(cells_run), text=f"{c} ({i + 1}/{len(cells_run)}) · retraining as it streams…")
            g_ = ct_s[(ct_s["Cell_ID"] == c) & ~ct_s["outlier"]]
            if g_.empty or c not in groups.index:
                continue
            soh_eol_c = te.soh_eol_for(float(g_["C_bol_Ah"].iloc[0]), eol_ah)
            tr_c = tuple(x for x in [cell] + train_c if x != c)
            try:
                parts.append(frozen_study_cached(ct_s, imp, dkey_s, c, tr_c, grp_c, float(soh_eol_c),
                                                 tuple(sorted(ivals)), tuple(anchors), int(horizon), float(eol_ah)))
            except Exception as exc:
                report_error(f"Frozen-model study failed on {c}", exc, debug)
            st.session_state["fzu"] = {"cfg": cfg, "tab": pd.concat(parts, ignore_index=True) if parts else None}
        prog.empty()
    saved = st.session_state.get("fzu")
    if not saved or saved.get("tab") is None or saved["cfg"][1] != grp_c:
        placeholder("Press 'Run the frozen-model update study': the error-versus-cost chart and the table appear here.")
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    tab = saved["tab"]
    ok = tab[tab["status"] == "ok"]
    if ok.empty:
        st.error("Every setting failed: " + "; ".join(tab["status"].astype(str).unique()[:3]))
        return
    nb = ok["Cell_ID"].nunique()
    summ = (ok.groupby(["anchor_key", "Anchor", "Retrain every"], sort=False)
              .agg(**{"Forecast RMSE (SOH pts)": ("Forecast RMSE (SOH pts)", "mean"),
                      "SOH-now RMSE (SOH pts)": ("SOH-now RMSE (SOH pts)", "mean"),
                      "EOL |error| (cycles)": ("EOL |error| (cycles)", "mean"),
                      "Refits per 100 cycles": ("Refits per 100 cycles", "mean"),
                      "Batteries": ("Cell_ID", "nunique")}).reset_index())
    show(fig_frozen_update(summ, P, f"Frozen {grp_c} model: forecast error versus update cost"
                                    + (f" (mean of {nb} batteries)" if nb > 1 else f" ({cell})")),
         key="fzu_fig", data=summ)
    show_table(summ.drop(columns=["anchor_key"]).set_index(["Anchor", "Retrain every"]).style.format(
        {"Forecast RMSE (SOH pts)": "{:.2f}", "SOH-now RMSE (SOH pts)": "{:.2f}", "EOL |error| (cycles)": "{:.1f}",
         "Refits per 100 cycles": "{:.1f}"}, na_rep="—"),
        note=f"Forecast RMSE: error of each forecast over the next {horizon} measured cycles, averaged over every cycle "
             "from 30 % of life on" + (", then over batteries" if nb > 1 else "") + ". SOH-now RMSE: field mode only, "
             "the estimated against the measured SOH. EOL error: only where end of life is reached after the start.")
    fails = tab[tab["status"] != "ok"]
    if len(fails):
        st.warning(f"{len(fails)} battery-setting combination(s) failed: "
                   + "; ".join(sorted(fails["status"].astype(str).str[:70].unique())[:3]))
    best = summ.loc[summ["Forecast RMSE (SOH pts)"].idxmin()]
    never = summ[(summ["Retrain every"] == "never") & (summ["anchor_key"] == "none")]
    lines = [f"Lowest forecast error: retrain every {best['Retrain every']} with "
             f"{te.FROZEN_ANCHORS[best['anchor_key']].split(' (')[0].lower()}: {best['Forecast RMSE (SOH pts)']:.2f} SOH "
             f"pts at {best['Refits per 100 cycles']:.1f} retrainings per 100 cycles."]
    if len(never):
        lines.append(f"Frozen once and never updated (the assessment forecast): "
                     f"{float(never['Forecast RMSE (SOH pts)'].iloc[0]):.2f} SOH pts.")
    if nb < 3:
        lines.append("Run the whole group before drawing a conclusion: one battery is an anecdote.")
    card("Mission 2 · update frequency of the frozen model", lines)
    stamp = {"app version": te.ENGINE_VERSION, "group": grp_c, "forecast model": fz[0], "SOH-now model": fz[1],
             "data": "reference condition" if saved["cfg"][0].endswith("|reference-condition") else "as recorded",
             "horizon (cycles)": int(saved["cfg"][5]),
             "EOL setting": f"{eol_ah:.2f} Ah" if eol_ah > 1 else f"{100 * eol_ah:.0f} % of initial capacity"}
    download("Download the update study (CSV)", tab.assign(**stamp).to_csv(index=False).encode(),
             f"frozen_update_{grp_c.replace(' ', '-')}__v{te.ENGINE_VERSION}.csv", "text/csv", key="fzu_csv")


def early_life_section() -> None:
    section("Early-life lifetime prediction from ΔQ(V)")
    st.markdown("Severson et al. (*Nature Energy* 4, 2019) showed that the variance of ΔQ(V), the change of the "
                "discharge curve between an early and a slightly later cycle, predicts cycle life long before capacity "
                "fades visibly. Here the same idea is applied to your cohort, with a leave-one-cell-out elastic net. "
                "The ΔQ(V) statistics also feed the ML fade-rate models as early-life descriptors.")
    c1, c2 = st.columns(2)
    n_a = c1.slider("Reference cycle", 1, 10, 2, key="el_a")
    n_b = c2.slider("Comparison cycle (early window end)", n_a + 5, 80, max(n_a + 5, 20), key="el_b")
    el = early_life_cached(ct, DATA_KEY, float(eol_ah), int(n_a), int(n_b))
    if el["table"].empty:
        st.info("Not enough cells with this many cycles.")
        return
    show(fig_dq_curves(ct, el, P), key="el_dq", data=el["table"])
    if not el["available"]:
        st.info("Fewer than five cells reached end of life with usable ΔQ(V): the lifetime regression is not "
                "available (lower the end-of-life capacity in the control bar to include more cells).")
        return
    k = st.columns(4)
    k[0].metric("Cells with known life", el["n_cells"])
    k[1].metric("corr(log var ΔQ, log life)", fmt(el["corr_logvar_loglife"], ".2f"))
    k[2].metric("LOCO lifetime error", fmt(el["mape_pct"], ".1f", "%"))
    k[3].metric("Cohort-mean baseline", fmt(el["baseline_mape_pct"], ".1f", "%"))
    show(fig_lifetime_parity(el, P, list(meta.index)), key="el_parity", data=el["loco"])
    st.caption("NASA cells live 40–200 cycles and the cohort mixes temperatures, currents and cut-offs. The early "
               "window is therefore much shorter than Severson's (cycle 10 → 100), and the elastic net also sees the "
               "operating conditions. A strongly negative correlation means ΔQ(V) carries lifetime information.")


def ml_methods_panel() -> None:
    with st.expander("Methods: machine-learning models", icon=":material/menu_book:"):
        st.markdown("**Fade-rate forecasting.** A regressor learns the degradation rate as a function of the current "
                    "state, the operating conditions and early-life descriptors, and is integrated forward from the "
                    "forecast origin, so tree models keep extrapolating beyond the longest training life:")
        st.latex(r"\frac{d\,\mathrm{SOH}}{dn} = g_\phi\!\left(\mathrm{SOH},\ T,\ I,\ V_\mathrm{cut},\ "
                 r"\text{early slope},\ \text{early } R \text{ growth},\ \log\operatorname{var}\Delta Q(V)\right) \le 0")
        st.markdown("**Uncertainty.** Split-conformal bands calibrated on the batteries closest in operating conditions, "
                    "widening with the forecast horizon. **Estimation.** Present SOH from operando indicators measured on "
                    "the same cycle, with random, chronological or by-battery splits. **Tuning.** Random search validated "
                    "by forecast backtests on other batteries (forecasting) or grouped cross-validation by battery "
                    "(estimation). **Early life.** Elastic net on ΔQ(V) statistics (Severson et al. 2019).")
        st.markdown("**Scores.** Accuracy = 100 × (1 − MAPE); fade skill = 1 − SSE / SSE of a 'no further fade' "
                    "forecast; R², RMSE and MAE on held-out cycles; RUL error and α-λ accuracy (Saxena et al. 2010). "
                    "Physics-based and mechanistic models (twin, particle filters, PINN) live in the Live twin view.")


LEVEL_INFO = {
    1: ("Baselines", "Persistence, linear trend and fleet shape: references that need no learning."),
    2: ("Machine learning", "SOH of every cycle learned from all training batteries: conditions and health parameters."),
    3: ("Deep learning", "MLP on the Level-2 inputs, GRU and Transformer over the last cycles: same question, neural networks."),
    4: ("Physics-informed", "Neural network constrained by the electrochemistry and degradation-mechanism equations."),
    5: ("Hybrid & physics-informed", "Mechanistic PINN and hierarchical Bayes: equations plus learning."),
    6: ("First principles", "Single-particle electrochemical model with SEI growth: physics only, calibrated."),
}
LEVEL_COLORS = {1: "#8C8C8C", 2: "#0072B2", 3: "#332288", 4: "#AA4499", 5: "#56B4E9", 6: "#882255"}


def scheme() -> Dict[str, Any]:
    return st.session_state.get("_scheme") or {"mode": "within", "targets": [cell], "train": None, "gen": "v2",
                                               "n0": {cell: int(max(5, round(0.4 * meta.loc[cell, "cycles"])))},
                                               "use_cohort": True, "show": cell}


def exclusive_pickers(c1: Any, c2: Any, label_a: str, label_b: str, options: Sequence[str], default_a: Sequence[str],
                      default_b: Sequence[str], key_a: str, key_b: str, fmt: Callable[[str], str]) -> Tuple[List[str], List[str]]:
    """Two multiselects that never share an item: each offers only what is not selected in the other, so choosing a
    battery in one list removes it from the other's options immediately."""
    ss = st.session_state
    a = [x for x in ss.get(key_a, default_a) if x in options]
    b = [x for x in ss.get(key_b, default_b) if x in options and x not in a]
    for k_, v in ((key_a, a), (key_b, b)):
        if k_ in ss:
            ss[k_] = v
    kw_a = {} if key_a in ss else {"default": a}
    kw_b = {} if key_b in ss else {"default": b}
    a = c1.multiselect(label_a, [x for x in options if x not in b], key=key_a, format_func=fmt, **kw_a)
    b = c2.multiselect(label_b, [x for x in options if x not in a], key=key_b, format_func=fmt, **kw_b)
    return list(a), list(b)


def scheme_controls() -> Dict[str, Any]:
    """One training scheme for every level of the ladder."""
    all_cells = list(meta.index)
    mode = st.radio("Training scheme", ["within", "same", "across"], horizontal=True, key="sch_mode",
                    format_func={"within": ":material/call_split: Within a battery: first part → rest",
                                 "same": ":material/thermostat: Same operating conditions: batteries of one group",
                                 "across": ":material/swap_horiz: Across batteries: any batteries → others"}.get)
    if mode == "within":
        c1, c2 = st.columns([3, 2])
        frac = c1.slider("Training part of the battery's life (the rest is the test)", 0.1, 0.9, 0.4, 0.05, key="ml_frac")
        use_cohort = c2.toggle("Models may also learn from the other batteries' full histories", value=True,
                               key="sch_cohort", help="Off = strictly this battery only (fewer data, harder for ML).")
        n0 = int(max(5, round(frac * meta.loc[cell, "cycles"])))
        sc = {"mode": "within", "targets": [cell], "train": None if use_cohort else [], "use_cohort": use_cohort,
              "n0": {cell: n0}, "show": cell, "frac": frac}
        st.caption(f"Battery {cell}: training on cycles 1–{n0}, testing on cycles {n0 + 1}–{int(meta.loc[cell, 'cycles'])}.")
    elif mode == "same":
        grp = te.condition_groups(ct)
        order = [g for g in te.GROUP_ORDER if g in set(grp["Group"])]
        my = grp.loc[cell, "Group"] if cell in grp.index else order[0]
        g_sel = st.selectbox("Operating-condition group", order, index=order.index(my), key="sch_group",
                             format_func=lambda g: f"{g} · {int((grp['Group'] == g).sum())} batteries")
        members = list(grp.index[grp["Group"] == g_sel])
        lab = lambda c: (f"{c} · {grp.loc[c, 'Ambient_C']:.0f} °C · {grp.loc[c, 'I_dis_A']:.1f} A · "
                         f"cut-off {meta.loc[c, 'V_cut_V']:.2f} V · {int(grp.loc[c, 'cycles'])} cycles")
        c1, c2 = st.columns(2)
        test_default = [cell] if cell in members else members[-1:]
        train, test = exclusive_pickers(c1, c2, "Training batteries (same group)", "Test batteries (same group)", members,
                                        [c for c in members if c not in test_default], test_default,
                                        "sch_same_train", "sch_same_test", lab)
        frac = st.slider("Start of each test battery the models may see (the rest is forecast)", 0.05, 0.6, 0.15, 0.05,
                         key="sch_same_seen")
        n0 = {t: int(max(10, round(frac * meta.loc[t, "cycles"]))) for t in test}
        show_t = st.selectbox("Battery to display in the charts", test or [cell], key="sch_same_show",
                              format_func=lambda c: cell_label(c, meta)) if test else cell
        sc = {"mode": "across", "variant": "same", "group": g_sel, "targets": test, "train": train,
              "use_cohort": False, "n0": n0, "show": show_t, "frac": frac}
        if len(members) < 2:
            st.warning(f"The {g_sel} group has a single battery: choose another group or another scheme.")
        elif not train or not test:
            st.warning("Choose at least one training and one test battery in this group.")
        else:
            st.caption(f"{g_sel}: learning from {len(train)} battery(ies) to forecast {len(test)} under the same "
                       "operating conditions. This isolates cell-to-cell variation from the effect of conditions.")
    else:
        c1, c2 = st.columns(2)
        near = te.calibration_partners(meta, cell, 4)
        train, test = exclusive_pickers(c1, c2, "Training batteries", "Test batteries (one or more)", list(all_cells),
                                        [c for c in near if c != cell], [cell], "sch_train", "sch_test",
                                        lambda c: cell_label(c, meta))
        frac = st.slider("Start of each test battery the models may see (the rest is forecast)", 0.05, 0.6, 0.15, 0.05,
                         key="sch_seen", help="Models need a few cycles of the new battery to know its current state.")
        n0 = {t: int(max(10, round(frac * meta.loc[t, "cycles"]))) for t in test}
        show_t = st.selectbox("Battery to display in the charts", test or [cell], key="sch_show",
                              format_func=lambda c: cell_label(c, meta)) if test else cell
        sc = {"mode": "across", "targets": test, "train": train, "use_cohort": False, "n0": n0, "show": show_t,
              "frac": frac}
        if not train or not test:
            st.warning("Choose at least one training and one test battery.")
        else:
            st.caption(f"Learning from {len(train)} batteries, forecasting {len(test)}: each test battery is seen for its "
                       f"first {100 * frac:.0f}% of life. Baselines, the PINN and the first-principles model only need "
                       "the test battery itself; the other levels learn from the training batteries.")
    gen = "v2"
    sc["gen"] = gen
    notes = []
    for t_ in list(sc.get("n0", {})):
        n_new, note = te.safe_origin(ct, t_, int(sc["n0"][t_]))
        sc["n0"][t_] = n_new
        if note:
            notes.append(note)
    if notes:
        st.info("Origins adjusted so that at least " f"{te.MIN_VALID_BEFORE_ORIGIN} valid cycles come before them: "
                + " ".join(notes))
    st.session_state["_scheme"] = sc
    return sc


def ladder_add(name: str, level: int, n0: int, n_grid: np.ndarray, soh: np.ndarray, lo: Optional[np.ndarray],
               hi: Optional[np.ndarray], metrics: Any, target: Optional[str] = None) -> None:
    store_ = st.session_state.setdefault("ladder", {})
    store_[(target or cell, int(n0), name)] = {"name": name, "level": int(level), "n_grid": np.asarray(n_grid),
                                               "soh": np.asarray(soh), "lo": None if lo is None else np.asarray(lo),
                                               "hi": None if hi is None else np.asarray(hi), "m": metrics,
                                               "target": target or cell}


def ladder_entries(n0: Optional[int] = None, target: Optional[str] = None) -> List[Dict[str, Any]]:
    sc = scheme()
    t = target or sc["show"]
    n = n0 if n0 is not None else sc["n0"].get(t)
    return sorted([v for (c, nn, _), v in st.session_state.get("ladder", {}).items() if c == t and nn == n],
                  key=lambda v: (v["level"], v["name"]))


def all_target_entries() -> List[Dict[str, Any]]:
    sc = scheme()
    out = []
    for t in sc["targets"]:
        out += ladder_entries(sc["n0"][t], t)
    return out


def fig_ladder(entries: Sequence[Dict[str, Any]], n0: int, P: Palette, title: str, target: Optional[str] = None) -> go.Figure:
    t = target or scheme()["show"]
    ctt = ct[(ct["Cell_ID"] == t) & ~ct["outlier"]]
    eol_t = te.soh_eol_for(float(meta.loc[t, "C_bol_Ah"]), float(eol_ah))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctt["n"], y=ctt["SOH"], mode="markers", name=f"Measured SOH · {t}",
                             marker=dict(color=P.measured, size=5, opacity=0.75), hovertemplate="%{y:.2%}<extra>measured</extra>"))
    for i, e in enumerate(entries):
        col = LEVEL_COLORS.get(e["level"], P.text)
        sel = e["n_grid"] >= n0
        if e["lo"] is not None and e["hi"] is not None and len(entries) <= 3:
            add_band(fig, e["n_grid"][sel], e["lo"][sel], e["hi"][sel], col, f"{e['name']} band", group=e["name"], alpha=0.12)
        fig.add_trace(go.Scatter(x=e["n_grid"][sel], y=e["soh"][sel], mode="lines", name=f"L{e['level']} · {e['name']}",
                                 legendgroup=e["name"], line=dict(color=col, width=2.6, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.2%}<extra>" + html.escape(e["name"]) + "</extra>"))
    fig.add_vline(x=n0, line_dash="dot", line_color=P.muted, annotation_text="training | test",
                  annotation_font=dict(color=P.muted))
    fig.add_hline(y=eol_t, line_dash="dash", line_color=P.eol, annotation_text="End of life",
                  annotation_font=dict(color=P.eol))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="SOH (%)", tickformat=".0%")
    return style_fig(fig, P, 520, f"{title} · {t}")


def _ladder_table(entries: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Per model: metrics averaged over the test batteries (one row per model)."""
    entries = [e for e in entries if e.get("m") is not None]
    rows = [{"Level": f"L{e['level']} · {LEVEL_INFO[e['level']][0]}", "Model": e["name"], "Battery": e["target"],
             "Accuracy (%)": e["m"].accuracy, "RMSE": e["m"].rmse, "Fade skill": e["m"].fade_skill,
             "Coverage": e["m"].coverage, "RUL error": e["m"].rul_error} for e in entries if e["m"] is not None]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    agg = d.groupby(["Level", "Model"], as_index=False).agg(
        **{"Test batteries": ("Battery", "nunique"), "Accuracy (%)": ("Accuracy (%)", "mean"), "RMSE": ("RMSE", "mean"),
           "Fade skill": ("Fade skill", "mean"), "Coverage": ("Coverage", "mean"),
           "RUL error": ("RUL error", lambda x: float(np.nanmean(np.abs(x))) if x.notna().any() else np.nan)})
    return agg


TABLE_FMT = {"Accuracy (%)": "{:.2f}", "RMSE": "{:.4f}", "Fade skill": "{:.3f}", "Coverage": "{:.0%}", "RUL error": "{:.0f}"}


def ladder_level_block(level: int, key: str) -> None:
    sc = scheme()
    shown = [e for e in ladder_entries() if e["level"] == level]
    allt = [e for e in all_target_entries() if e["level"] == level]
    if not allt:
        placeholder(f"Press the button above to run Level {level} · {LEVEL_INFO[level][0]} on the test "
                    f"battery(ies); the forecast chart and scores will appear here.")
        return
    if shown:
        show(fig_ladder(shown, sc["n0"][sc["show"]], P, f"Level {level} · {LEVEL_INFO[level][0]}"), key=key, export=False)
    if allt:
        multi = len(sc["targets"]) > 1
        show_table(_ladder_table(allt).set_index("Model").style.format(TABLE_FMT, na_rep="—"),
                   note=("Scores averaged over the test batteries (RUL error = mean absolute error in cycles)." if multi else
                         "Scores on the test cycles of this battery."))


def gens() -> List[str]:
    g = scheme().get("gen", "v2")
    return ["v1", "v2"] if g == "both" else [g]


def _run_targets(fn: Callable[[str, int], Any], label: str) -> None:
    """Run a forecaster for every test battery of the current scheme, with progress and per-battery errors."""
    sc = scheme()
    prog = st.progress(0.0)
    for i, t in enumerate(sc["targets"]):
        prog.progress(i / max(len(sc["targets"]), 1), text=f"{label}: {t}")
        try:
            fn(t, sc["n0"][t])
        except Exception as exc:
            st.warning(f"{label} on {t}: {exc}")
    prog.empty()


def ladder_intro() -> None:
    done = {e["level"] for e in ladder_entries()}
    steps = "".join(
        f'<div class="bt-step{" bt-step-done" if lv in done else ""}"><div class="bt-step-n">{lv}</div>'
        f'<div class="bt-step-t">{html.escape(LEVEL_INFO[lv][0])}</div></div>' for lv in ACTIVE_LEVELS)
    st.markdown(f'<div class="bt-ladder">{steps}</div>', unsafe_allow_html=True)
    st.caption("Two independent levels: no model uses another level's output. The leaderboard and the accuracy "
               "assessment only compare them on the same test cycles. Advanced levels (deep learning, hybrid, first "
               "principles) are added one at a time once these are validated.")


def ladder_baselines() -> None:
    section("Level 1 · Baselines: the references every model must beat")
    st.markdown("Three forecasts that need no learning at all: **persistence** (SOH stays), **linear trend** (the "
                "recent slope continues) and **fleet shape** (the typical fade curve of similar batteries, placed at this "
                "battery's state and speed). If a model cannot beat the best of them, its complexity is not paying off.")
    _names = {"persistence": "Persistence", "trend": "Linear trend", "shape": "Fleet shape"}
    _cols = st.columns(3)
    for _c, (_k, _lab) in zip(_cols, _names.items()):
        with (_c.popover(_lab, icon=":material/functions:", use_container_width=True) if hasattr(st, "popover")
              else _c.expander(_lab, icon=":material/functions:")):
            for text, tex in MODEL_EQUATIONS[_k]:
                if text:
                    st.caption(text)
                st.latex(tex)
    if st.button("Run baselines", key="lad_b_go", icon=":material/play_arrow:", type="primary"):
        sc_b = scheme()
        def run(t, n0):
            for kind in ("persistence", "trend"):
                for g in gens():
                    f = (te.baseline_forecast_v2 if g == "v2" else te.baseline_forecast)(ct, t, n0, kind, float(eol_ah),
                                                                                       horizon_factor=1.0)
                    ladder_add(f.name, 1, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
            f = te.fleet_shape_forecast(ct, t, n0, float(eol_ah),
                                        list(sc_b["train"]) if sc_b["mode"] == "across" and sc_b.get("train") else None,
                                        horizon_factor=1.0)
            ladder_add(f.name, 1, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
        _run_targets(run, "Baselines")
    ladder_level_block(1, "lad_base")


def ladder_deep() -> None:
    section("Level 4 · Deep learning: recurrent network and Transformer")
    st.markdown("Sequence models read a window of past SOH values (plus temperature and current) and predict the next "
                "cycle; feeding each prediction back builds the whole forecast. Three networks with different seeds "
                "form a small ensemble for the band. They are trained on all other batteries and this battery's past.")
    c = st.columns(4)
    kinds = c[0].multiselect("Networks", list(te.SEQ_MODELS), default=list(te.SEQ_MODELS), key="lad_d_kinds")
    L = c[1].select_slider("Window (cycles)", [5, 8, 10, 15, 20], value=10, key="lad_d_L")
    ep = c[2].select_slider("Epochs", [100, 200, 300, 500], value=200, key="lad_d_ep")
    mem = c[3].select_slider("Ensemble members", [1, 2, 3, 5], value=3, key="lad_d_mem")
    if st.button(f"Train deep models (≈ {int(len(kinds) * mem * ep / 60 * max(len(scheme()['targets']), 1)) + 2} s)", key="lad_d_go",
                 icon=":material/neurology:", type="primary", disabled=not kinds):
        sc = scheme()
        def run(t, n0):
            for k in kinds:
                for g in gens():
                    if g == "v2":
                        f = te.seq_forecast_v2(ct, t, n0, k, float(eol_ah), L=int(L), epochs=int(ep),
                                               n_members=max(1, int(mem) - 1), train_cells=sc["train"])
                    else:
                        f = te.seq_forecast(ct, t, n0, k, float(eol_ah), L=int(L), epochs=int(ep), n_members=int(mem),
                                            train_cells=sc["train"])
                    ladder_add(f.name, 4, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
        _run_targets(run, "Deep models")
    ladder_level_block(4, "lad_deep")
    with st.expander("Why no large language model (LLM) forecaster?", icon=":material/help:"):
        st.markdown("LLMs and time-series foundation models are trained on text or on millions of generic series; "
                    "with ~30 batteries they add no physical knowledge and cannot be validated against it, and "
                    "running them needs large downloads or paid APIs. They are useful *around* the twin instead: "
                    "explaining results, drafting reports, or answering operator questions from the twin's outputs. "
                    "The deep models here (GRU, Transformer) are the same architectures at a size these data can "
                    "support.")


def ladder_hybrid() -> None:
    section("Level 5 · Hybrid & physics-informed models")
    st.markdown("**Mechanistic PINN**: a neural network trained to fit the data *and* obey the SEI, plating and "
                "loss-of-active-material equations. **Hierarchical Bayes**: a physics fade law whose parameters "
                "start from what the whole fleet taught us and are updated by this battery's data.")
    c = st.columns(3)
    run_p = c[0].toggle("Mechanistic PINN", value=True, key="lad_h_pinn")
    ep = c[1].select_slider("PINN epochs", [500, 1000, 1500, 2500], value=1000, key="lad_h_ep")
    run_hb = c[2].toggle("Hierarchical Bayes", value=True, key="lad_h_hb")
    if st.button("Run hybrid models", key="lad_h_go", icon=":material/hub:", type="primary"):
        sc = scheme()
        def run(t, n0):
            pop = sc["train"] if sc["mode"] == "across" else None
            cfg = te.PINNConfig(epochs=int(ep), physics="mechanistic")
            for g in gens():
                if run_p:
                    if g == "v2":
                        f = te.pinn_forecast_v2(ct, imp, t, n0, cfg, float(eol_ah), train_cells=pop)
                        ladder_add(f.name, 5, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
                    else:
                        r = te.train_pinn(ct, imp, t, n0, cfg, float(eol_ah))
                        ladder_add("Mechanistic PINN", 5, n0, r.n_grid, r.soh, r.soh_lo, r.soh_hi, r.metrics, t)
                if run_hb:
                    f = (te.hierarchical_bayes_forecast_v2 if g == "v2" else te.hierarchical_bayes_forecast)(
                        ct, t, n0, float(eol_ah), train_cells=pop)
                    ladder_add(f.name if g == "v2" else "Hierarchical Bayes", 5, n0, f.n_grid, f.soh, f.lo, f.hi,
                               f.metrics, t)
        _run_targets(run, "Hybrid models")
    ladder_level_block(5, "lad_hybrid")


def fig_spm_curves(P: Palette) -> go.Figure:
    p = te.SPMParams()
    I = float(meta.loc[cell, "I_dis_A"])
    T = float(meta.loc[cell, "T_mean_C"])
    fig = go.Figure()
    for name, kw, col, dash in (("New cell", dict(), P.accent, "solid"), ("10% lithium lost", dict(lli=0.10), "#E69F00", "dash"),
                                ("20% lithium lost", dict(lli=0.20), "#D55E00", "dot"),
                                ("2× current", dict(I_A=2 * I), "#AA4499", "dashdot"), ("Cold (4 °C)", dict(T_C=4.0), "#56B4E9", "longdash")):
        args = dict(p=p, I_A=I, T_C=T)
        args.update(kw)
        r = te.spm_discharge(**args)
        fig.add_trace(go.Scatter(x=r["q_Ah"], y=r["V"], mode="lines", name=f"{name} · {r['capacity_Ah']:.2f} Ah",
                                 line=dict(color=col, width=2.6, dash=dash), hovertemplate="%{y:.3f} V<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Discharged capacity (Ah)")
    fig.update_yaxes(title_text="Terminal voltage (V)")
    return style_fig(fig, P, 470, f"Single-particle model: discharge at {I:.1f} A, {T:.0f} °C")


def ladder_first_principles() -> None:
    section("Level 6 · First principles: single-particle electrochemical model")
    st.markdown("Each electrode is one spherical particle: lithium diffuses inside it (Fick's law), crosses the "
                "surface with Butler–Volmer kinetics, and the voltage is the difference of the LiCoO₂ and graphite "
                "potentials minus the losses. Ageing is SEI growth consuming lithium, "
                "LLI = a·Ah + b·√Ah (reaction- plus diffusion-limited), fitted to this battery's capacity history.")
    show(fig_spm_curves(P), key="lad_spm_curves", export=False)
    with st.expander("Governing equations", icon=":material/functions:"):
        st.latex(r"\frac{\partial \theta}{\partial t} = \frac{D}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial \theta}{\partial r}\right),"
                 r"\qquad D\,\frac{\partial \theta}{\partial r}\Big|_{r=R} = \mp\frac{I}{3600\,C}\,\frac{R}{3}")
        st.latex(r"V = U_p(\theta_{p,s}) - U_n(\theta_{n,s}) - \frac{2RT}{F}\left[\sinh^{-1}\frac{I}{2 i_{0,p}} + "
                 r"\sinh^{-1}\frac{I}{2 i_{0,n}}\right] - I\,(R_\Omega + R_\mathrm{film}),\quad i_0 \propto \sqrt{\theta(1-\theta)}")
        st.latex(r"\mathrm{LLI}(\mathrm{Ah}) = a\,\mathrm{Ah} + b\,\sqrt{\mathrm{Ah}},\qquad x_0 \rightarrow x_0 - \mathrm{LLI},"
                 r"\qquad R_\mathrm{film} \propto \mathrm{LLI}")
    if st.button("Run first-principles forecast", key="lad_s_go", icon=":material/science:", type="primary"):
        sc = scheme()
        def run(t, n0):
            for g in gens():
                if g == "v2":
                    f = te.spm_forecast_v2(ct, t, n0, float(eol_ah), train_cells=sc["train"] if sc["mode"] == "across" else None)
                else:
                    f = te.spm_forecast(ct, t, n0, float(eol_ah))
                ladder_add(f.name if g == "v2" else "SPM + SEI", 6, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
                st.session_state["spm_params"] = f.params
        _run_targets(run, "First-principles model")
    ladder_level_block(6, "lad_spm")
    if st.session_state.get("spm_params"):
        show_table(pd.Series(st.session_state["spm_params"], name="Value").to_frame().style.format("{:.4g}"),
                   note="Fitted SEI kinetics: a large reaction term means near-linear fade, a large diffusion term "
                        "means fade that slows down as the SEI film thickens.")


def ladder_leaderboard() -> None:
    section("Leaderboard: every level on the same test cycles")
    sc = scheme()
    shown = ladder_entries()
    allt = all_target_entries()
    if not allt:
        st.info("Run at least one level above; results appear here side by side.")
        return
    if shown:
        show(fig_ladder(shown, sc["n0"][sc["show"]], P, "All models"), key="lad_board", export=False)
    tab = _ladder_table(allt).sort_values("RMSE")
    best = tab.iloc[0]
    medals = ("🥇", "🥈", "🥉")
    pod = "".join(f'<div class="bt-pod bt-pod-{i}"><div class="bt-pod-m">{medals[i]}</div>'
                  f'<div class="bt-pod-n">{html.escape(r["Model"])}</div><div class="bt-pod-l">{html.escape(r["Level"])}</div>'
                  f'<div class="bt-pod-v">{r["Accuracy (%)"]:.2f}% · RMSE {r["RMSE"]:.4f}</div></div>'
                  for i, (_, r) in enumerate(tab.head(3).iterrows()))
    st.markdown(f'<div class="bt-podium">{pod}</div>', unsafe_allow_html=True)
    k = st.columns(3)
    k[0].metric("Best model", best["Model"], delta=best["Level"], delta_color="off")
    k[1].metric("Mean accuracy", fmt(best["Accuracy (%)"], ".2f", "%"),
                help=f"Averaged over {int(best['Test batteries'])} test battery(ies).")
    base = tab[tab["Model"] == "Baseline · linear trend"]
    if len(base):
        k[2].metric("Gain over linear trend", fmt(100 * (1 - best["RMSE"] / base["RMSE"].iloc[0]), ".0f", "%"),
                    help="How much lower the best model's error is than the simplest credible baseline.")
    incomplete = tab[tab["Test batteries"] < len(sc["targets"])]
    show_table(tab.set_index("Model").style.format(TABLE_FMT, na_rep="—")
               .highlight_min(subset=["RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"),
               note="Sorted by error on the test cycles" + (" and averaged over the test batteries" if len(sc["targets"]) > 1
                                                             else "") + ". More complexity is only worth it when it "
                    "clearly beats the lower levels and keeps honest uncertainty (coverage near 90%).")
    if len(incomplete):
        st.warning("Not all models ran on every test battery: " + ", ".join(
            f"{m} ({int(n)}/{len(sc['targets'])})" for m, n in zip(incomplete["Model"], incomplete["Test batteries"])) +
                   ". Compare them with care.")
    if st.button("Clear leaderboard", key="lad_clear", icon=":material/delete:"):
        st.session_state["ladder"] = {}
        st.rerun()


ACTIVE_LEVELS = (1, 2, 3, 4)     # independent levels, compared only in the leaderboard and the assessment


FAMILY_COLORS = {"Baselines": "#8C8C8C", "Machine learning (forecast)": "#0072B2",
                 "Machine learning (measured)": "#56B4E9", "Deep learning (forecast)": "#AA4499",
                 "Deep learning (measured)": "#CC79A7", "Physics-informed (PINN)": "#882255"}


def fig_scoreboard(sb: pd.DataFrame, P: Palette) -> go.Figure:
    d = sb.iloc[::-1]
    fig = go.Figure(go.Bar(y=d.index, x=d["RMSE (SOH pts)"], orientation="h",
                           marker=dict(color=[FAMILY_COLORS.get(f, P.text) for f in d["Family"]]),
                           # an error cannot be negative: the lower bar stops at zero
                           error_x=dict(type="data", array=d["95% CI ±"].fillna(0),
                                        arrayminus=np.minimum(d["95% CI ±"].fillna(0), d["RMSE (SOH pts)"]),
                                        symmetric=False, color=P.muted),
                           customdata=np.column_stack([d["Family"], d["Batteries"], d["vs best baseline"]]),
                           hovertemplate="%{y}<br>%{x:.2f} SOH points<br>%{customdata[0]} · %{customdata[1]} batteries"
                                         "<br>%{customdata[2]}<extra></extra>"))
    bb = sb.attrs.get("best_baseline")
    if bb in sb.index:
        fig.add_vline(x=float(sb.loc[bb, "RMSE (SOH pts)"]), line_dash="dash", line_color=P.muted,
                      annotation_text=f"best baseline: {bb}", annotation_font=dict(color=P.muted))
    fig.update_xaxes(title_text="Forecast error, RMSE (SOH points), mean over held-out batteries ± 95% CI", rangemode="tozero")
    return style_fig(fig, P, 140 + 36 * len(sb), "Prediction-accuracy assessment: same batteries, same protocol",
                     hovermode="closest")




ACC_SCOPES = ("Best 2 per level (chosen by CV in each fold)", "Models I choose")
ACC_EXCLUDED_GROUPS = ("Corrupted logging",)          # NASA: logging failed, the measured SOH cannot be trusted
ACC_EXPLORATORY = {
    "Pulsed load": "Exploratory group. These batteries were discharged in pulses with rests, so the capacity of a cycle "
                   "is measured differently from a continuous discharge: compare models within this group only, and "
                   "report its results as exploratory.",
    "Mixed conditions": "Exploratory group. The discharge current or ambient temperature of these batteries changes "
                        "during their test, which makes measured capacity jump independently of ageing. Here every "
                        "battery is therefore scored at its reference condition only (cycles at its initial current "
                        "and ambient; the Reference-condition SOH is applied automatically). The batteries come from "
                        "different campaigns, and cells of one campaign share a test schedule that models can learn: "
                        "report these results as exploratory.",
}
ASSESS_SECONDS = {"PINN": 8.0, "PINN-ablation": 4.0, "PINN-full": 9.0, "PINN-full-ablation": 5.0, "persistence": 0.3, "trend": 0.3, "shape": 1.0, "Bayesian Ridge": 1.5, "Decision Tree": 1.5,
                  "Random Forest": 3.0, "SVM": 2.0, "Gaussian Process": 12.0, "XGBoost": 3.0, "LightGBM": 2.0,
                  "MLP": 2.0, "GRU": 6.0, "Transformer": 4.0, "Shared physics v2": 6.0, "Hybrid": 15.0, "Shared physics v3": 6.0,
                  "Hybrid v3": 15.0, "Physics v2 + ageing speed": 6.0, "Hybrid v2 + ageing speed": 15.0,
                  "Physics v3 + ageing speed": 6.0, "Hybrid v3 + ageing speed": 15.0}


GA_ASSESS_KEY = "pinn:GA-PINN"            # virtual key: the held-out battery's own GA winner (STATUS 5cf)


def ga_search_key(ct_: pd.DataFrame, data_key: str, c: str, n0: int, train: Sequence[str],
                  note: Optional[Any] = None) -> Optional[str]:
    """GA-PINN for ANY battery, forecast start and training batteries (STATUS 5ci): reuse a saved search when it is
    the same situation, otherwise run the search now (population and generations of the GA-PINN section, default
    6 x 4; every evaluation saved on disk, so a re-run is instant). Nested fitness: only the training batteries'
    futures are scored. Returns the winner's key."""
    train_t = tuple(sorted(x for x in train if x != c))
    saved = (st.session_state.get("ga_pinn") or {}).get(c) or {}
    if saved.get("done") and saved.get("n0") == int(n0) and tuple(sorted(saved.get("train") or ())) == train_t:
        return saved.get("best_key")
    runs = st.session_state.setdefault("ga_runs", {})
    rk = (data_key, c, int(n0), train_t)
    if rk in runs and runs[rk].get("done"):
        return runs[rk]["best_key"]
    pop = int(st.session_state.get("ga_pop", 6))
    gens = int(st.session_state.get("ga_gens", 4))
    count = {"n": 0}

    def evaluate(g: Dict[str, Any]) -> float:
        count["n"] += 1
        if note is not None:
            note.caption(f"GA-PINN search for {c} (forecast start {int(n0)}): evaluation {count['n']} "
                         f"(about {pop + 2 * (gens - 1) + 3} in total; saved, so a re-run is instant)")
        return ga_fitness_cached(ct_, imp, data_key, c, train_t, int(n0), float(eol_ah),
                                 tuple(sorted(g.items())))["fitness"]
    run = te.ga_pinn_run(evaluate, pop, gens, seed=0)
    runs[rk] = run
    if note is not None:
        note.empty()
    return run.get("best_key") if run.get("done") else None


def assess_ga_task(store: Any, ct_run: pd.DataFrame, dkey: str, cells_g: Tuple[str, ...], origin: float, c: str
                   ) -> pd.DataFrame:
    """GA-PINN in the assessment: run the held-out battery's own GA winner (chosen without it) and relabel the rows
    so all batteries count as one model. Empty when the battery has no GA result."""
    good = ct_run[(ct_run["Cell_ID"] == c) & ~ct_run["outlier"]].sort_values("n")
    n0, _ = te.safe_origin(ct_run, c, int(max(10, round(float(origin) * int(good["n"].max())))))
    gk = ga_search_key(ct_run, dkey, c, int(n0), [x for x in cells_g if x != c], st.empty())
    if not gk:
        return pd.DataFrame()
    r = assess_task_cached(store, ct_run, imp, dkey, cells_g, gk, float(origin), c, float(eol_ah)).copy()
    if len(r):
        r["key"], r["model"], r["family"] = GA_ASSESS_KEY, "GA-PINN (each battery's GA winner)", "Physics-informed (PINN)"
    return r


@st.cache_data(show_spinner=False, max_entries=20000)
def assess_task_cached(_store: Any, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cells: Tuple[str, ...],
                       mkey: str, origin: float, held: str, eol: float) -> pd.DataFrame:
    """One forecast of the assessment, cached as soon as it finishes: an interrupted run resumes where it stopped."""
    return te.run_protocol_task(_store, _ct, _imp, cells, mkey, origin, held, eol)


def _per_battery_table(res: pd.DataFrame) -> pd.DataFrame:
    ok = res[res["status"] == "ok"]
    piv = ok.groupby(["model", "Cell_ID"])["RMSE (SOH pts)"].mean().unstack()
    return piv


def assessment_section() -> None:
    section("Accuracy assessment: every model, each battery held out in turn")
    st.markdown("One forecast on one battery is an anecdote. Here each battery of a condition group is held out in turn: "
                "the models learn from the other batteries of the group plus the held-out battery's own cycles up to the "
                "forecast origin, and are scored on all its later measured cycles. Each forecast is saved as soon as it "
                "finishes, so the scoreboard fills in live and an interrupted run continues where it stopped.")
    groups = te.condition_groups(CT_RAW)       # membership from the recorded test conditions, whatever the toggles say
    usable = [g for g in te.GROUP_ORDER if g in set(groups["Group"]) and g not in ACC_EXCLUDED_GROUPS]
    if not usable:
        st.info("No usable condition group (batteries with corrupted logging are always excluded).")
        return
    my = groups.loc[cell, "Group"] if cell in groups.index and groups.loc[cell, "Group"] in usable else usable[0]
    c1, c2 = st.columns(2)
    grp = c1.selectbox("Condition group", usable, index=usable.index(my), key="acc_group",
                       format_func=lambda x: f"{x}, {len(te.protocol_cells(CT_RAW, x))} batteries"
                                         + (" (exploratory)" if x in ACC_EXPLORATORY else ""))
    if grp in ACC_EXPLORATORY:
        st.warning(ACC_EXPLORATORY[grp], icon=":material/science:")
    # data the forecasts use: mixed conditions always at each battery's reference condition (SOH = ageing only);
    # every other group as recorded, so its results do not depend on the Reference-condition toggle either
    if grp == "Mixed conditions":
        ct_run, dkey = ref_ct_cached(CT_RAW, DATA_KEY_RAW), DATA_KEY_RAW + "|reference-condition"
    else:
        ct_run, dkey = CT_RAW, DATA_KEY_RAW
    origins = c2.multiselect("Forecast origins (share of each battery's life)", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                             default=[0.3], key="acc_origins", format_func=lambda v: f"{100 * v:.0f}%")
    av = te.available_models()
    options = (["persistence", "trend", "shape"] + [f"traj:{m}" for m in av] + [f"trajm:{m}" for m in av]
               + [f"dl:{k}" for k in te.DL_MODELS] + [f"dlm:{k}" for k in te.DL_MODELS] + [f"pinn:{k}" for k in te.PINN_MODELS])
    scope = st.radio("What to score", ACC_SCOPES, key="acc_scope", horizontal=True,
                     help="Best 2 per level: for every held-out battery the models of each level are ranked by their CV "
                          "error on the other (training) batteries and the 2 lowest are scored, so the choice never "
                          "sees the battery it is scored on. Picking the best 2 from a scoreboard afterwards would be "
                          "optimistic.")
    cv_mode = scope == ACC_SCOPES[0]
    if cv_mode:
        level_keys = {"Level 2 · ML forecast": [f"traj:{m}" for m in av], "Level 2 · ML measured": [f"trajm:{m}" for m in av],
                      "Level 3 · DL forecast": [f"dl:{k}" for k in te.DL_MODELS],
                      "Level 3 · DL measured": [f"dlm:{k}" for k in te.DL_MODELS],
                      "Level 4 · PINN (original inputs) + ablation": ["pinn:PINN", "pinn:PINN-ablation"],
                      "Level 4 · PINN twin core (best PINN before the GA)": ["pinn:PINN-twin"],
                      "Level 4 · GA-PINN (each battery's GA winner)": [GA_ASSESS_KEY]}
        levels = st.multiselect("Levels (baselines are always included: the verdict needs them)", list(level_keys),
                                default=list(level_keys), key="acc_levels")
        keys = ["persistence", "trend", "shape"] + [k for lv in levels for k in level_keys[lv]]
        st.caption("Every model of each chosen level is fitted for every held-out battery (its CV error is needed to "
                   "choose), but the scoreboard shows only the 2 with the lowest CV error per level and mode, plus the "
                   "baselines and Level 4 (only the PINN and its no-physics control, so there is nothing to choose). "
                   "The models picked on each battery are listed below the verdict.")
    else:
        keys = st.multiselect("Models (level 1 baselines, 2 machine learning, 3 deep learning, 4 physics-informed)", options,
                              default=["trend", "shape"] + [f"traj:{m}" for m in ("Bayesian Ridge", "Random Forest", "SVM")
                                                            if m in av] + ["dl:GRU"],
                              key="acc_models", format_func=lambda k: model_label(k))
    view = (lambda r: te.select_by_cv(r, 2, keys)[0]) if cv_mode else (lambda r: r)
    cells_g = tuple(te.protocol_cells(CT_RAW, grp))
    tasks = [(c, o, k) for c in cells_g for o in sorted(origins) for k in keys]
    secs = sum(ASSESS_SECONDS.get(k.split(":", 1)[-1], 3.0) for _, _, k in tasks)
    if len(cells_g) < 3:
        st.warning(f"The {grp} group has fewer than three usable batteries: nothing to learn from when one is held out.")
    st.caption(f"{len(tasks)} forecasts, about {max(1, round(secs / 60))} min on a fresh run (the Gaussian Process is the "
               "slowest model). Forecasts already computed with the same settings are reused instantly.")
    run_key = (dkey, grp, float(eol_ah), tuple(sorted(origins)), tuple(keys))
    if st.button(f"Run the assessment ({len(tasks)} forecasts)", key="acc_go", type="primary", icon=":material/fact_check:",
                 disabled=len(cells_g) < 3 or not keys or not origins):
        prog = st.progress(0.0)
        live = st.empty()
        parts: List[pd.DataFrame] = []
        t_start = time.time()
        for i, (c, o, k) in enumerate(tasks):
            prog.progress(i / max(len(tasks), 1),
                          text=f"{i}/{len(tasks)} · {model_label(k)} on {c} from {100 * o:.0f}% · "
                               f"{time.time() - t_start:.0f}s")
            if k == GA_ASSESS_KEY:
                r_ga = assess_ga_task(store, ct_run, dkey, cells_g, float(o), c)
                if not len(r_ga):
                    st.warning(f"GA-PINN on {c}: the search did not finish.")
                else:
                    parts.append(r_ga)
            else:
                parts.append(assess_task_cached(store, ct_run, imp, dkey, cells_g, k, float(o), c, float(eol_ah)))
            if i % 3 == 2 or i == len(tasks) - 1:
                part = pd.concat(parts, ignore_index=True)
                sbp = te.protocol_scoreboard(view(part))
                if len(sbp):
                    live.dataframe(sbp[["Batteries", "RMSE (SOH pts)", "95% CI ±", "EOL |error| (cycles)"]].round(2),
                                   use_container_width=True)
                st.session_state["acc_res"] = part              # partial results survive an interruption
                st.session_state["acc_key"] = run_key
        prog.empty()
        live.empty()
    res = st.session_state.get("acc_res")
    if res is None or not len(res) or st.session_state.get("acc_key", (None,))[:3] != run_key[:3]:
        placeholder("Press 'Run the assessment': the scoreboard, the verdict, the winner on each battery and the error by "
                    "forecast origin appear here.")
        return
    if st.session_state.get("acc_key") != run_key:
        st.info("The results below were computed with different models or origins than currently selected; press Run to "
                "update them (forecasts already computed are reused).")
    res_all = res
    picks = pd.DataFrame()
    if cv_mode:
        res, picks = te.select_by_cv(res_all, 2, keys)
        n_inc = int(res["status"].astype(str).str.startswith("incomplete").sum())
        if n_inc:
            st.warning(f"The run stopped before every model of a level had finished on some batteries ({n_inc // 2} "
                       "battery-level combinations): no model is picked there, because choosing among the models that "
                       "happened to finish is not the method being assessed. Press 'Run the assessment' again without "
                       "touching the page; finished forecasts are reused.")
    sb = te.protocol_scoreboard(res)
    if sb.empty:
        st.warning("No forecast succeeded: " + "; ".join(sorted(set(res["status"].astype(str).str[:70]))[:3]))
        return
    best, bb = sb.index[0], sb.attrs.get("best_baseline")
    better = sb[sb["vs best baseline"].astype(str).str.startswith("better")]
    n_done = int(res.loc[res["status"] == "ok", "Cell_ID"].nunique())
    if 0 < n_done < 3:
        st.caption(f"With {n_done} batteries no 95% interval is shown: it would be meaningless (too few batteries).")
    if n_done < len(cells_g):
        st.warning(f"These results cover only {n_done} of the {len(cells_g)} batteries of the {grp} group: the run was "
                   "interrupted (any click on the page during a run stops it). Press 'Run the assessment' again without "
                   "touching the page: finished forecasts are reused, so it continues where it stopped. A 95% interval "
                   "and a comparison need at least 2 batteries.")
    if bb is None:
        st.info("No baseline was selected, so no model can be compared with one. Add 'Fleet shape' (and 'Linear trend') "
                "to the models to get the verdict 'better / worse than the best baseline'.")
    k = st.columns(4)
    k[0].metric("Lowest error", best, help="Mean RMSE over the held-out batteries.")
    k[1].metric("Its error", f"{sb.loc[best, 'RMSE (SOH pts)']:.2f} SOH pts", delta=f"± {sb.loc[best, '95% CI ±']:.2f} (95% CI)",
                delta_color="off")
    eol_col = sb["EOL |error| (cycles)"].dropna()
    if len(eol_col):
        k[2].metric("Best end-of-life timing", eol_col.idxmin(), delta=f"{eol_col.min():.0f} cycles off", delta_color="off")
    k[3].metric("Beat the best baseline", f"{len(better)} model(s)", delta=f"baseline: {bb}", delta_color="off")
    show(fig_scoreboard(sb, P), key="acc_board", data=sb.reset_index())
    show_table(sb.style.format({"RMSE (SOH pts)": "{:.2f}", "95% CI ±": "{:.2f}", "EOL |error| (cycles)": "{:.1f}",
                                "Coverage": "{:.0%}", "p (Wilcoxon)": "{:.3f}"}, na_rep="—"),
               note="RMSE in SOH points, averaged per battery first and then across batteries (95% CI). 'vs best "
                    "baseline' is paired on the same batteries. Coverage should be near 90%.")
    n_b = int(sb["Batteries"].max())
    txt = (f"In the **{grp}** group ({n_b} batteries), the lowest error is **{best}** "
           f"({sb.loc[best, 'RMSE (SOH pts)']:.2f} ± {sb.loc[best, '95% CI ±']:.2f} SOH points). ")
    txt += ("Better than the best baseline (" + str(bb) + "): " + ", ".join(better.index) + "."
            if len(better) else f"No model is better than the best baseline ({bb}) on all batteries.")
    if n_b < 6:
        txt += f" With {n_b} batteries no test can reach 5% significance: read the per-battery table below."
    card("Verdict", [txt])
    piv = _per_battery_table(res)
    if len(piv):
        order = [m for m in sb.index if m in piv.index]
        piv = piv.loc[order]
        wins = piv.idxmin().value_counts()
        piv["Battery wins"] = [int(wins.get(m, 0)) for m in piv.index]
        show_table(piv.style.format("{:.2f}", na_rep="—", subset=[c for c in piv.columns if c != "Battery wins"])
                   .highlight_min(axis=0, subset=[c for c in piv.columns if c != "Battery wins"],
                                  props="background-color: rgba(23,126,120,0.35); font-weight: 600;"),
                   note="Error of every model on every held-out battery (mean over origins); the best model on each "
                        "battery is highlighted. 'Battery wins' counts on how many batteries a model was the best.")
    by_o = res[res["status"] == "ok"].groupby(["model", "origin"])["RMSE (SOH pts)"].mean().unstack()
    by_o.columns = [f"from {100 * o:.0f}%" for o in by_o.columns]
    if by_o.shape[1] > 1:
        show_table(by_o.loc[[m for m in sb.index if m in by_o.index]].style.format("{:.2f}", na_rep="—"),
                   note="Error by forecast origin: later origins should give smaller errors.")
    bad = res[res["status"] != "ok"]
    if len(bad):
        st.caption(f"{len(bad)} forecasts failed and are counted as failures: "
                   + "; ".join(sorted(set(bad["status"].str[:60]))[:3]))
    if cv_mode:
        _cv_picks_block(picks, res_all, cells_g)
    stamp = {"app version": te.ENGINE_VERSION, "group": grp,
             "EOL setting": f"{eol_ah:.2f} Ah" if eol_ah > 1 else f"{100 * eol_ah:.0f} % of initial capacity",
             "data": "reference condition" if grp == "Mixed conditions" else "as recorded",
             "group batteries": " ".join(cells_g)}
    tag = f"{grp.replace(' ', '-')}__v{te.ENGINE_VERSION}"
    download("Download the assessment, every model (CSV)", res_all.assign(**stamp).to_csv(index=False).encode(),
             f"assessment_{tag}.csv", "text/csv", key="acc_csv")
    if cv_mode and len(picks):
        download("Download the models picked by CV (CSV)", picks.assign(**stamp).to_csv(index=False).encode(),
                 f"assessment_{tag}_cv_picks.csv", "text/csv", key="acc_picks_csv")


def _cv_picks_block(picks: pd.DataFrame, res_all: pd.DataFrame, cells_g: Sequence[str]) -> None:
    """Which models CV picked on each held-out battery, whether the star agreed with the test, and (for reference
    only) every model's score."""
    if picks is None or not len(picks):
        st.info("No model of the chosen levels has a CV error yet, so nothing could be picked.")
        return
    agree = te.cv_pick_agreement(picks)
    if len(agree):
        show_table(agree.style.format({"Star also best on test": "{:.0%}", "Mean test rank of the star": "{:.1f}"}),
                   note="Does the star agree with the test? For each level, how often the model with the lowest CV error "
                        "(chosen from the training batteries) was also the best on the held-out battery, and its mean "
                        "rank there (1 = best). Rank near 1: the star can be trusted; near the number of models: it "
                        "cannot.")
    short = picks.copy()
    short["Battery · origin"] = short["Cell_ID"] + " · " + (100 * short["origin"]).round().astype(int).astype(str) + "%"
    short["Pick"] = (short["Picked model"].str.replace(" (measured)", "", regex=False).str.split(" · ").str[-1]
                     + " (test rank " + short["Test rank"].astype(str) + "/" + short["Candidates"].astype(str) + ")")
    piv = short.pivot_table(index="Battery · origin", columns=["Level", "Choice"], values="Pick", aggfunc="first")
    piv.columns = [f"{a} · {b}" for a, b in piv.columns]
    show_table(piv, note="The models chosen by CV on each held-out battery: the choice can differ from battery to "
                         "battery, which is part of what the scoreboard above measures.")
    expected = {lv: int(res_all.loc[res_all["family"] == fam, "key"].nunique())
                for fam, lv in te.CV_SELECT_FAMILIES.items() if (res_all["family"] == fam).any()}
    short_c = picks.groupby("Level")["Candidates"].min()
    low = [lv for lv, n in short_c.items() if n < expected.get(lv, n)]
    if low:
        st.warning("On some batteries, fewer models had a CV error than were run (" + ", ".join(low) + "): the choice "
                   "was made among the models available there.")
    with st.expander("Every model of the chosen levels (for reference, not for choosing)"):
        st.caption("Picking the best 2 from this table and reporting their scores would be optimistic: with a few "
                   "batteries some models win by luck, and the same batteries would confirm them. Report the scoreboard "
                   "above.")
        sb_all = te.protocol_scoreboard(res_all)
        if len(sb_all):
            show_table(sb_all.style.format({"RMSE (SOH pts)": "{:.2f}", "95% CI ±": "{:.2f}", "EOL |error| (cycles)": "{:.1f}",
                                            "Coverage": "{:.0%}", "p (Wilcoxon)": "{:.3f}"}, na_rep="—"))


def _level_results(res: Sequence[Any], show_t: str, level: int, key: str, ctx: Optional[Dict[str, Any]] = None) -> None:
    """Chart, accuracy table, importance and inputs of one level's runs for one battery and one mode."""
    if res[0].origin_note:
        st.warning(res[0].origin_note)
    entries = [{"name": f.model, "level": level, "n_grid": f.n_grid, "soh": f.soh, "lo": f.lo, "hi": f.hi, "m": f.metrics,
                "target": show_t} for f in res]
    show(fig_ladder(entries, res[0].n0, P, f"Level {level} · {LEVEL_INFO[level][0]}", show_t), key=f"{key}_fig",
         export=False)
    singles = [f for f in res if np.isfinite(f.cv_rmse) and not f.model.startswith("Level ensemble")]
    rec = min(singles, key=lambda f: f.cv_rmse).model if singles else None
    rows = [{"Model": f.model + ("  ★ recommended" if f.model == rec else ""), "Accuracy (%)": f.metrics.accuracy,
             "RMSE": f.metrics.rmse, "Coverage": f.metrics.coverage, "CV RMSE (held-out batteries)": f.cv_rmse,
             "Training batteries": len(f.train_cells), "Time (s)": f.fit_seconds} for f in res]
    show_table(pd.DataFrame(rows).set_index("Model").sort_values("RMSE").style.format(
        {"Accuracy (%)": "{:.2f}", "RMSE": "{:.4f}", "Coverage": "{:.0%}", "CV RMSE (held-out batteries)": "{:.4f}",
         "Time (s)": "{:.1f}"}),
        note="RMSE on this battery's test cycles (SOH units). CV RMSE: error on batteries held out during training, "
             "the better guide to how the model generalises: ★ marks the model with the lowest CV RMSE, chosen without "
             "looking at this battery's future. The self-weighted ensemble combines the models by their CV RMSE.")
    base_of = lambda f: f.model.split(" · ", 1)[-1].replace(" (measured)", "")
    supported = [f for f in res if level in (2, 3) and not f.model.startswith("Level ensemble")
                 and base_of(f) not in ("GRU", "Transformer")]
    if supported:
        names = [f.model for f in supported]
        default = names.index(rec) if rec in names else 0
        pick = st.selectbox("Show the indicators of", names, index=default, key=f"{key}_imp_model",
                            help="Permutation importance on batteries held out during training, computed for the chosen "
                                 "model when you select it (then remembered). GRU and Transformer read sequences, so "
                                 "they have no comparable indicator chart.")
        f_sel = supported[names.index(pick)]
        imp_tab = f_sel.importance
        if (imp_tab is None or not len(imp_tab)) and ctx is not None:
            with st.spinner(f"Computing the indicators of {pick}…"):
                try:
                    bm = base_of(f_sel)
                    fi = level_fit_cached(ct, imp, DATA_KEY, level, show_t, f_sel.n0, bm, f_sel.mode, ctx.get("train"),
                                          json.dumps(ctx.get("params", {}).get(bm) or {}, sort_keys=True, default=float),
                                          float(eol_ah), True)
                    imp_tab = fi.importance
                except Exception as exc:
                    st.caption(f"Indicators could not be computed: {exc}")
        if imp_tab is not None and len(imp_tab):
            show(fig_importance(imp_tab, P, pick), key=f"{key}_imp", export=False)
        else:
            st.caption("Train this level again to compute the indicators of its models.")
    best = min(res, key=lambda f: f.metrics.rmse)
    with st.expander(f"Inputs for {show_t} (origin at cycle {res[0].n0})", icon=":material/fact_check:"):
        st.markdown("**Inputs used:** " + " · ".join(best.features))
        if best.inputs_at_origin:
            show_table(pd.DataFrame({"Input": list(best.inputs_at_origin), "Value": list(best.inputs_at_origin.values())})
                       .set_index("Input").style.format({"Value": "{:.4g}"}),
                       note="Values at the origin for the best model of this run.")


# Defining equations of every model, shown behind a small ∑ icon next to the model choices.
MODEL_EQUATIONS: Dict[str, List[Tuple[str, str]]] = {
    "persistence": [("SOH stays at its value at the origin n₀:", r"\widehat{SOH}(n) = SOH(n_0)")],
    "trend": [("Robust (Theil–Sen) straight line through the last 20 cycles before the origin:",
               r"\widehat{SOH}(n) = SOH(n_0) + b\,(n - n_0),\quad b = \operatorname{median}_{i<j}\frac{SOH_j - SOH_i}{n_j - n_i}")],
    "shape": [("Typical fade curve of the training batteries against charge throughput A (median):", r"F(A) = \operatorname{median}_b\, SOH_b(A)"),
              ("Equivalent age and relative speed of this battery:", r"F(a^*) = SOH(n_0),\qquad k = \Big(\frac{dSOH/dA\,|_{battery}}{dF/dA\,|_{a^*}}\Big)^{0.7}"),
              ("Forecast:", r"\widehat{SOH}(n) = SOH(n_0) + F\big(a^* + k\,\Delta A(n)\big) - F(a^*)")],
    "_traj": [("Forecast mode, smooth models (SOH change, anchored at today's value):",
               r"\widehat{SOH}(m) = SOH(n_0) + f(\mathbf{x}_{m}) - f(\mathbf{x}_{m}\,|\,m{=}n_0)"),
              ("Forecast mode, tree models (average fade rate per Ah since the origin):",
               r"\widehat{SOH}(m) = SOH(n_0) + \min\big(f(\mathbf{x}_m),0\big)\,\Delta A_m"),
              ("Measured mode (each cycle from its own measurements, 5-cycle trailing median):", r"\widehat{SOH}(m) = f(\mathbf{x}_m)")],
    "Bayesian Ridge": [("Linear model with Gaussian priors (weights and noise precisions estimated from the data):",
                        r"y = \mathbf{w}^\top \mathbf{x} + \varepsilon,\quad \mathbf{w}\sim\mathcal N(0,\lambda^{-1}I),\quad \varepsilon\sim\mathcal N(0,\alpha^{-1})")],
    "Decision Tree": [("Piecewise-constant prediction: the mean of the training targets in the leaf R(x):", r"f(\mathbf{x}) = \bar y_{R(\mathbf{x})}")],
    "Random Forest": [("Average of B trees, each grown on a bootstrap sample with random feature subsets:",
                       r"f(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})")],
    "SVM": [("Support-vector regression with an RBF kernel and an ε-insensitive loss:",
             r"f(\mathbf{x}) = \sum_i (\alpha_i - \alpha_i^*)\,K(\mathbf{x}_i, \mathbf{x}) + b,\quad K(\mathbf{x},\mathbf{x}') = e^{-\gamma\lVert\mathbf{x}-\mathbf{x}'\rVert^2}")],
    "Gaussian Process": [("Gaussian process with one length scale per input plus a linear trend term and noise:",
                          r"k(\mathbf{x},\mathbf{x}') = \sigma^2\exp\!\Big(-\tfrac12\sum_d \tfrac{(x_d-x_d')^2}{\ell_d^2}\Big) + \sigma_{lin}^2\,\mathbf{x}^\top\mathbf{x}' + \sigma_n^2\,\delta_{\mathbf{x}\mathbf{x}'}")],
    "XGBoost": [("Gradient boosting: each new tree fits the gradient of the loss, added with a learning rate η:",
                 r"F_M(\mathbf{x}) = \sum_{m=1}^{M} \eta\, f_m(\mathbf{x}),\quad \mathcal L = \sum_i \ell(y_i, F(\mathbf{x}_i)) + \sum_m\big(\gamma T_m + \tfrac{\lambda}{2}\lVert w_m\rVert^2\big)")],
    "LightGBM": [("Gradient boosting with leaf-wise trees and histogram splits:", r"F_M(\mathbf{x}) = \sum_{m=1}^{M} \eta\, f_m(\mathbf{x})")],
    "MLP": [("Feed-forward network (ReLU layers, L2 penalty α, early stopping):",
             r"\mathbf{h}_{l} = \max(0,\, W_l\,\mathbf{h}_{l-1} + \mathbf{b}_l),\quad f(\mathbf{x}) = \mathbf{w}^\top \mathbf{h}_L + b")],
    "GRU": [("Gated recurrent unit over the last cycles, then a dense output layer:",
             r"z_t=\sigma(W_z x_t+U_z h_{t-1}+b_z),\ r_t=\sigma(W_r x_t+U_r h_{t-1}+b_r)"),
            ("", r"\tilde h_t=\tanh\big(W_h x_t+U_h(r_t\odot h_{t-1})+b_h\big),\quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t")],
    "Transformer": [("Self-attention over the last cycles, then a dense output layer:",
                     r"\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)V")],
    "PINN": [("Observation (electrochemistry):", r"SOH = (1 - LLI_{SEI} - LLI_{pl})\,(1 - LAM),\qquad R/R_0 = 1 + g\,LLI_{SEI}"),
             ("SEI growth (diffusion-limited, Arrhenius):", r"\frac{dLLI_{SEI}}{dA} = \frac{k_{SEI}\, e^{\frac{E_a}{R}\left(\frac{1}{T_{ref}}-\frac{1}{T}\right)}}{LLI_{SEI}+\delta_0}"),
             ("Lithium plating (cold, high current, promoted by LAM: knees):",
              r"\frac{dLLI_{pl}}{dA} = k_{pl}\, e^{b_C\,x_C}\Big(\frac{I}{2\,A}\Big)^{a_I}(1 + c\,LAM)"),
             ("Loss of active material:", r"\frac{dLAM}{dA} = k_{LAM}\Big(\frac{I}{2\,A}\Big)^{a_L}"),
             ("Training loss (data + physics residuals at collocation points):",
              r"\mathcal L = \tfrac{1}{\sigma^2}\lVert \widehat{SOH}-SOH\rVert^2 + w_R\lVert \widehat R - R\rVert^2 + \lambda_{phys}\sum_{j}\lVert \tfrac{d s_j}{dA} - g_j(s)\rVert^2")],
    "PINN-ablation": [("The same network and data term without the physics residuals (λ_phys = 0):",
                       r"\mathcal L = \tfrac{1}{\sigma^2}\lVert \widehat{SOH}-SOH\rVert^2 + w_R\lVert \widehat R - R\rVert^2")],
}


def equation_icons(models: Sequence[str], common: Optional[str] = None) -> None:
    """A small ∑ button per model with equations; clicking it shows them."""
    items = [m for m in models if m in MODEL_EQUATIONS]
    if not items and common is None:
        return
    pop = getattr(st, "popover", None)
    cols = st.columns(min(len(items) + (1 if common else 0), 6) or 1)
    k = 0
    for name in ([common] if common else []) + items:
        label = "How Level 2–3 predict" if name == "_traj" else name
        with (cols[k % len(cols)].popover(label, icon=":material/functions:", use_container_width=True)
              if pop else cols[k % len(cols)].expander(label, icon=":material/functions:")):
            for text, tex in MODEL_EQUATIONS[name]:
                if text:
                    st.caption(text)
                st.latex(tex)
        k += 1


def influence_panel(level: int, key: str, mdls: Sequence[str], mode: str, train: Optional[List[str]], show_t: str,
                    n0: int, params: Dict[str, Dict[str, Any]]) -> None:
    family = {2: "ml", 3: "dl", 4: "pinn"}[level]
    with st.expander("Which training batteries perturb the models?", icon=":material/troubleshoot:"):
        st.caption("Retrains the chosen model once without each training battery and measures how the forecast error on "
                   f"{show_t} changes. A battery is said to **perturb** the model when the error falls without it. This "
                   "compares with the battery's actual later cycles, so it is a diagnostic of the training data, not part "
                   "of the forecast. It takes one training per training battery.")
        if not mdls:
            st.caption("Select a model above first.")
            return
        c1, c2 = st.columns([2, 1])
        m = c1.selectbox("Model to analyse", list(mdls), key=f"{key}_inf_model")
        thr = c2.select_slider("Neutral below (SOH points)", [0.05, 0.1, 0.2, 0.5], value=0.1, key=f"{key}_inf_thr")
        n_tr = len(train) if train is not None else len(te.v2_training_pool(ct, show_t))
        if st.button(f"Analyse {n_tr} training batteries for {m} ({mode} mode)", key=f"{key}_inf_go",
                     icon=":material/troubleshoot:"):
            prog = st.progress(0.0)
            try:
                st.session_state[f"{key}_inf"] = (m, mode, show_t, te.training_influence(
                    ct, imp, show_t, n0, m, family, mode, train, float(eol_ah), params.get(m), float(thr),
                    progress=lambda f_, msg: prog.progress(min(max(float(f_), 0.0), 1.0), text=msg)))
            except Exception as exc:
                st.warning(f"Analysis failed: {exc}")
            prog.empty()
        saved = st.session_state.get(f"{key}_inf")
        if saved:
            m_, md_, t_, tab = saved
            pert = tab[tab["Effect"] == "perturbs"]
            helps = tab[tab["Effect"] == "helps"]
            known = pert[pert["Group"].isin(te.GROUP_ISSUES)]
            lines = [f"**{m_}** on **{t_}** ({md_} mode): error with all training batteries "
                     f"{tab.attrs.get('base_rmse', float('nan')):.2f} SOH points."]
            if len(pert):
                lines.append(f"**{len(pert)} batteries perturb the model**: " + ", ".join(
                    f"{b} ({-d:.2f})" for b, d in pert["Change (SOH pts)"].items()) + " (error decrease without them).")
            else:
                lines.append("No training battery perturbs the model beyond the threshold.")
            if len(known):
                lines.append(f"Of these, {len(known)} have a known data issue: " + ", ".join(
                    f"{b} ({g})" for b, g in known["Group"].items()) + ".")
            if len(helps):
                lines.append(f"{len(helps)} batteries help the most: " + ", ".join(helps.index[-3:][::-1]) + ".")
            card("Training-data diagnosis", lines)
            show_table(tab.style.format({"Error without it (SOH pts)": "{:.2f}", "Change (SOH pts)": "{:+.2f}"}, na_rep="—")
                       .apply(lambda col: ["color: #E07B39; font-weight: 600" if v == "perturbs" else
                                           ("color: #2FA7A0" if v == "helps" else "") for v in col], subset=["Effect"]),
                       note="Change = error without the battery minus error with all of them: negative means the "
                            "battery perturbs the model. The comment names the likely cause when its group has a "
                            "known data issue.")


def _level_fit(ct_: pd.DataFrame, imp_: Optional[pd.DataFrame], level: int, t: str, n0: int, m: str, mode: str,
               train: Optional[List[str]], prm: Optional[Dict[str, Any]], eol: float, importance: bool) -> Any:
    if level == 4:
        return te.train_pinn_model(ct_, imp_, t, n0, m, mode, train, eol, prm)
    if level == 3 and m in ("GRU", "Transformer"):
        return te.train_deep_trajectory(ct_, imp_, t, n0, m, mode, train, eol, model_params=prm, importance=importance)
    return te.train_trajectory_model(ct_, imp_, t, n0, m, mode, train, eol, model_params=prm, importance=importance)


@st.cache_data(show_spinner=False, max_entries=500)
def level_fit_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], data_key: str, level: int, t: str, n0: int, m: str,
                     mode: str, train: Optional[Tuple[str, ...]], prm_json: str, eol: float, importance: bool) -> Any:
    """Same model, battery, settings and data: the trained result is reused instantly."""
    prm = json.loads(prm_json) or None
    return _level_fit(_ct, _imp, level, t, n0, m, mode, list(train) if train is not None else None, prm, eol, importance)


@st.cache_data(show_spinner=False, max_entries=500)
def self_tune_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], data_key: str, t: str, n0: int, m: str, mode: str,
                     train: Optional[Tuple[str, ...]], eol: float, n_iter: int) -> Dict[str, Any]:
    """Self-optimisation memory: the best settings found for this model, battery set and mode are remembered."""
    r = te.tune_level_model(_ct, _imp, t, n0, m, mode, list(train) if train is not None else None, eol,
                            n_iter=n_iter, method="bayesian")
    return {"params": r.best_params, "default": r.default_score, "best": r.best_score, "gain": r.improvement_pct}


def model_level_section(level: int, title: str, intro: str, models: Dict[str, str], default: Sequence[str],
                        trainer: Callable[..., Any], key: str, extra: Optional[Callable[[], Dict[str, Any]]] = None,
                        modes: Sequence[str] = ("forecast", "measured")) -> None:
    """One independent level with the two modes. Results of each mode are kept separately (tabs), so both stay
    visible after switching mode and training again."""
    section(title)
    st.markdown(intro)
    sc = scheme()
    c1, c2 = st.columns([2, 1])
    mdls = c1.multiselect("Models", list(models), default=[m for m in default if m in models], key=f"{key}_models",
                          format_func=lambda m: models[m])
    mdls_hp = [m for m in mdls if m != "GA-PINN"]          # GA-PINN: settings come from the GA, not the editor
    equation_icons(mdls_hp, "_traj" if level in (2, 3) else None)
    mode = c2.radio("Mode to train", list(modes), key=f"{key}_mode", format_func=te.TRAJ_MODES.get,
                    help="Forecast: only what is known at the training | test line (a real forecast). Measured: each "
                         "test cycle's own measured parameters (estimation). Results of both modes are kept below.")
    kw = extra() if extra else {}
    train = None if sc["mode"] == "within" else list(sc["train"])
    if train:
        grp_ = te.condition_groups(ct)["Group"]
        odd = sorted(c for c in train if grp_.get(c) in te.V2_EXCLUDED_GROUPS)
        if odd:
            st.caption(f"The training set keeps the data as they are, including {len(odd)} batteries from groups with "
                       "known data issues (mixed loads, corrupted logging, pulsed load). After training, 'Which training "
                       "batteries perturb the models?' below shows which ones actually hurt the prediction.")
    hp_prefix = f"{key}_hp"
    pending = st.session_state.pop(f"{key}_pending", None)      # tuned values from the last automatic selection
    if pending:
        for m_, prm in pending.items():
            for k_, v in prm.items():
                st.session_state[f"{hp_prefix}_{m_}_{k_}"] = v
    with st.expander("Model parameters (optimise each model by hand)", icon=":material/tune:"):
        st.caption("Each model's own settings. The values from an automatic selection appear here, and you can still "
                   "adjust them. 'Reset' restores the defaults.")
        params = hyperparam_editor(mdls_hp, hp_prefix) if mdls_hp else {}
        if st.button("Reset to defaults", key=f"{key}_hp_reset", icon=":material/restart_alt:"):
            for k_ in [k_ for k_ in st.session_state if str(k_).startswith(hp_prefix + "_")]:
                del st.session_state[k_]
            st.rerun()
    with st.expander("Automatic parameter selection", icon=":material/auto_fix_high:"):
        st.caption("Searches each selected model's parameters and keeps the settings with the lowest error on training "
                   "batteries held out during cross-validation. The test battery's cycles after the training | test "
                   "line are never used, so tuning cannot see the answer. The defaults are always tried first.")
        t1, t2 = st.columns([2, 1])
        n_iter = t1.select_slider("Candidates per model", [6, 10, 16, 24], value=10, key=f"{key}_tn_iter",
                                  help="More candidates search better but take longer; deep models take about 10–30 s each.")
        method = t2.radio("Search", list(te.TUNING_METHODS), index=1, horizontal=True, key=f"{key}_tn_method",
                          format_func={"random": "Random", "bayesian": "Bayesian optimisation"}.get)
        show_t0 = sc.get("show", sc["targets"][0])
        if st.button(f"Tune Level {level}: {len(mdls)} model(s) on {show_t0}, {mode} mode", key=f"{key}_tn_go",
                     icon=":material/auto_fix_high:", disabled=not mdls_hp):
            prog = st.progress(0.0)
            best, rows = {}, []
            for i, m in enumerate(mdls_hp):
                r = te.tune_level_model(ct, imp, show_t0, sc["n0"][show_t0], m, mode, train, float(eol_ah), n_iter=int(n_iter),
                                        method=method, progress=lambda f_, msg: prog.progress(
                                            min((i + max(min(float(f_), 1.0), 0.0)) / max(len(mdls_hp), 1), 1.0),
                                            text=f"{m}: {msg}"))
                best[m] = r.best_params
                rows.append({"Model": m, "Default CV RMSE": r.default_score, "Tuned CV RMSE": r.best_score,
                             "Improvement (%)": r.improvement_pct, "Time (s)": r.seconds,
                             "Best parameters": ", ".join(f"{k_}={v:.3g}" if isinstance(v, float) else f"{k_}={v}"
                                                          for k_, v in r.best_params.items())})
            prog.empty()
            st.session_state[f"{key}_pending"] = best
            st.session_state[f"{key}_tuning"] = pd.DataFrame(rows).set_index("Model")
            st.rerun()
        tun = st.session_state.get(f"{key}_tuning")
        if tun is not None and len(tun):
            show_table(tun.style.format({"Default CV RMSE": "{:.4f}", "Tuned CV RMSE": "{:.4f}", "Improvement (%)": "{:+.0f}",
                                         "Time (s)": "{:.0f}"}),
                       note="Errors on held-out training batteries (SOH units). The best parameters are now set in "
                            "'Model parameters' and used by the next Train.")
    if sc["mode"] == "within":
        pool = te.v2_training_pool(ct, cell)
        st.caption(f"Within scheme: trained on {len(pool)} batteries ({', '.join(pool[:6])}{'…' if len(pool) > 6 else ''}): "
                   "the same condition group when it has at least 3 other batteries, otherwise all comparable batteries, "
                   "plus this battery's own cycles up to the origin.")
    so1, so2 = st.columns([1, 2])
    self_opt = so1.toggle("Self-optimise", value=False, key=f"{key}_selfopt",
                          help="Before training, each model tunes its own parameters (Bayesian search on held-out training "
                               "batteries) and remembers the best settings for this battery set, so the next run reuses "
                               "them instantly. The test battery's future is never used.")
    so2.caption("Self-optimise: each model tunes itself once per battery set and remembers it (the first run takes "
                "longer). After training, the models are also combined into a self-weighted ensemble. With many "
                f"training batteries, training rows are sampled evenly across batteries (at most {te.TRAJ_MAX_ROWS:,}, "
                f"SVM {te.TRAJ_MAX_ROWS_SVM:,}) to keep training fast; the predicted battery keeps all its rows.")
    if st.button(f"Train in {mode} mode ({len(mdls) * len(sc['targets'])} runs)", key=f"{key}_go", type="primary",
                 icon=":material/timeline:", disabled=not mdls):
        prog = st.progress(0.0)
        out: Dict[str, List[Any]] = {}
        k_ = 0
        tuned: Dict[str, Any] = {}
        used_prm: Dict[str, Any] = {}
        tr_key = tuple(train) if train is not None else None
        for t in sc["targets"]:
            for m in mdls:
                prog.progress(k_ / max(len(mdls) * len(sc["targets"]), 1),
                              text=f"{'Self-optimising and training' if self_opt else 'Training'} {m} on {t}")
                k_ += 1
                try:
                    prm = params.get(m)
                    m_fit = m
                    if level == 4 and m == "GA-PINN":       # GA search for this battery, start and training set
                        tr_ga = list(tr_key) if tr_key else [x for x in te.v2_training_pool(ct, t) if x != t]
                        gk = ga_search_key(ct, DATA_KEY, t, int(sc["n0"][t]), tr_ga, st.empty())
                        if not gk:
                            st.warning(f"GA-PINN on {t}: the search did not finish.")
                            continue
                        m_fit, prm = gk.split(":", 1)[1], None
                    if self_opt and m_fit == m:
                        so = self_tune_cached(ct, imp, DATA_KEY, t, sc["n0"][t], m, mode, tr_key, float(eol_ah), 8)
                        prm, tuned[m] = so["params"], so
                    used_prm[m] = prm
                    f = level_fit_cached(ct, imp, DATA_KEY, level, t, sc["n0"][t], m_fit, mode, tr_key,
                                         json.dumps(prm or {}, sort_keys=True, default=float), float(eol_ah), False)
                    out.setdefault(t, []).append(f)
                    ladder_add(f.model, level, sc["n0"][t], f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
                except Exception as exc:
                    st.warning(f"{m} on {t}: {exc}")
            res_t = out.get(t, [])
            # importance only for the recommended model (lowest error on held-out training batteries)
            cand = [f for f in res_t if np.isfinite(f.cv_rmse)]
            if cand and level in (2, 3):
                best = min(cand, key=lambda f: f.cv_rmse)
                bm = best.model.split(" · ", 1)[1].replace(" (measured)", "")
                if bm not in ("GRU", "Transformer"):
                    try:
                        prm_b = tuned[bm]["params"] if bm in tuned else params.get(bm)
                        fb = level_fit_cached(ct, imp, DATA_KEY, level, t, sc["n0"][t], bm, mode, tr_key,
                                              json.dumps(prm_b or {}, sort_keys=True, default=float), float(eol_ah), True)
                        res_t[res_t.index(best)] = fb
                    except Exception:
                        pass
            if len(cand) >= 2:
                try:
                    e = te.ensemble_trajectories(res_t, ct, float(eol_ah))
                    res_t.append(e)
                    ladder_add(e.model, level, sc["n0"][t], e.n_grid, e.soh, e.lo, e.hi, e.metrics, t)
                except Exception as exc:
                    st.warning(f"Ensemble on {t}: {exc}")
        prog.empty()
        st.session_state.setdefault(key, {})[mode] = out
        st.session_state.setdefault(f"{key}_ctx", {})[mode] = {"train": tr_key, "params": used_prm}
        if tuned:
            st.session_state[f"{key}_pending"] = {m_: v["params"] for m_, v in tuned.items()}
            st.session_state[f"{key}_selfopt_log"] = pd.DataFrame(
                [{"Model": m_, "Default CV RMSE": v["default"], "Self-optimised CV RMSE": v["best"],
                  "Improvement (%)": v["gain"]} for m_, v in tuned.items()]).set_index("Model")
    so_log = st.session_state.get(f"{key}_selfopt_log")
    if so_log is not None and len(so_log):
        show_table(so_log.style.format({"Default CV RMSE": "{:.4f}", "Self-optimised CV RMSE": "{:.4f}",
                                        "Improvement (%)": "{:+.0f}"}),
                   note="Self-optimisation: the error of each model on held-out training batteries with its default and "
                        "with its self-chosen settings. The chosen settings are now shown in 'Model parameters'.")
    store_ = st.session_state.get(key, {})
    show_t = sc.get("show", sc["targets"][0])
    tabs = st.tabs([f"{'Forecast' if md == 'forecast' else 'Measured'} mode" + (" ✓" if store_.get(md, {}).get(show_t) else "")
                    for md in modes])
    for tab, md in zip(tabs, modes):
        with tab:
            res = store_.get(md, {}).get(show_t, [])
            if res:
                _level_results(res, show_t, level, f"{key}_{md}", st.session_state.get(f"{key}_ctx", {}).get(md))
            else:
                placeholder(f"Select '{te.TRAJ_MODES[md]}' above and press Train: the results of this mode appear here "
                            "and stay available when you train the other mode.")
    influence_panel(level, key, mdls, mode, train, show_t, sc["n0"][show_t], params)


def trajectory_section() -> None:
    model_level_section(
        2, "Level 2 · Machine learning: can a battery's SOH be predicted from the data of other batteries?",
        "Mission 2 asks whether future degradation can be predicted accurately and which variables are most "
        "informative. The models learn, from **every cycle of every training battery**, how SOH depends on the cycle "
        "number, the operating conditions and the measured health parameters, then predict the selected battery **cycle "
        "by cycle**. This level is independent: it uses no baseline.",
        {m: m for m in te.available_models()}, ("Bayesian Ridge", "Random Forest", "SVM", "Gaussian Process", "XGBoost", "LightGBM"),
        lambda t, n0, m, mode, train, model_params=None: te.train_trajectory_model(ct, imp, t, n0, m, mode, train,
                                                                                  float(eol_ah), model_params=model_params),
        "tr")


def deep_section() -> None:
    model_level_section(
        3, "Level 3 · Deep learning: do neural networks predict better than Level 2?",
        "The **same question, data, training batteries and test** as Level 2, answered with neural networks. The "
        "**MLP** receives exactly the Level-2 inputs (a direct comparison of the learner). The **GRU** and the "
        "**Transformer** read the battery's **last 10 cycles** as a sequence: in forecast mode its SOH, resistance, "
        "temperature-rise and CV-time history up to the origin; in measured mode its mean voltage, resistance, charge "
        "times, temperatures and efficiency. This level is independent of Levels 1 and 2.",
        te.DL_MODELS, ("MLP", "GRU", "Transformer"),
        lambda t, n0, m, mode, train, model_params=None, **_: te.train_deep_trajectory(ct, imp, t, n0, m, mode, train,
                                                                                       float(eol_ah), model_params=model_params),
        "dl")


def fig_mechanisms(f: Any, P: Palette, battery: str) -> go.Figure:
    mech = f.mechanisms
    fig = go.Figure()
    cols = {"LLI by SEI growth": "#0072B2", "LLI by lithium plating": "#AA4499", "Loss of active material": "#56B4E9"}
    for k, col in cols.items():
        fig.add_trace(go.Scatter(x=mech["n"], y=100 * np.asarray(mech[k]), mode="lines", name=k, stackgroup="m",
                                 line=dict(color=col, width=1.5), hovertemplate="cycle %{x}: %{y:.1f}%<extra>" + k + "</extra>"))
    fig.update_xaxes(title_text="Discharge cycle n (from the origin)")
    fig.update_yaxes(title_text="Capacity lost (% of initial)")
    return style_fig(fig, P, 400, f"Which mechanisms does the PINN attribute {battery}'s ageing to?")



# ---- mechanism colours and electrochemical expectations (shared by the physics and hybrid sections) ----
PV2_EXPECT = {
    "Hot": "At 43 °C, SEI growth is commonly reported as a major contributor (it accelerates with temperature), "
           "together with loss of active material.",
    "Cold": "At 4 °C, lithium plating during charging is commonly reported to matter; a deep discharge "
            "(2.0–2.2 V cut-off) can add loss of active material.",
    "Reference": "At room temperature and moderate current, gradual SEI growth and loss of active material are "
                 "commonly reported.",
    "High current": "High current is commonly reported to raise loss of active material (mechanical stress) and heating.",
    "Pulsed load": "Pulsed discharge with rests: no single expected mechanism; compare with the other groups.",
}
MECH_KEYS = {"LLI by SEI growth": "SEI", "LLI by lithium plating": "plating", "Loss of active material": "LAM"}


# ---- Shared physics on the real NASA batteries ----
@st.cache_data(show_spinner=False, max_entries=8)
def nasa_physics_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cells: Tuple[str, ...],
                        restarts: int, version: int, fit_with: str) -> Dict[str, Any]:
    return backed("physics", lambda: te.nasa_physics_fit(_ct, _imp, list(cells), restarts=restarts, version=version,
                                                         fit_with=fit_with),
                  _ct, _imp, cells, restarts, version, fit_with)


NP_VERSIONS = {"Version 3: learned SEI exponent + plating triggered by lost graphite (recommended)": 3,
               "Version 2: plating only below 15 °C, knee term, no charge-current term": 2,
               "Version 1: first fit (plating at any temperature)": 1}
NP_EVIDENCE = {"Impedance Re (SEI film); check with the load-step resistance": "Re",
               "Load-step resistance; check with Re": "R_dc"}


def fig_nasa_shares(tab: pd.DataFrame, P: Palette) -> go.Figure:
    order = {"Hot": 0, "High current": 1, "Reference": 2, "Pulsed load": 3, "Cold": 4}
    t = tab.assign(_o=tab["group"].map(order).fillna(9)).sort_values(["_o", "battery"])
    fig = go.Figure()
    for m in te.MECHANISMS:
        fig.add_trace(go.Bar(x=[t["group"], t["battery"]], y=t[f"{m} (%)"], name=m,
                             marker_color=MECH_COLORS[MECH_KEYS[m]],
                             hovertemplate="%{y:.0f} % of the capacity loss<extra>" + m + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Share of the capacity loss (%)", range=[0, 100])
    return style_fig(fig, P, 460, "Shared physics: mechanism split of every NASA battery (last cycle)", hovermode="closest")


def fig_nasa_traj(tr: pd.DataFrame, P: Palette, cell: str) -> go.Figure:
    fig = go.Figure()
    for m in te.MECHANISMS:
        fig.add_trace(go.Scatter(x=tr["n"], y=100 * tr[m].clip(lower=0), name=m, stackgroup="mech", mode="lines",
                                 line=dict(width=0.6, color=MECH_COLORS[MECH_KEYS[m]])))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * (1 - tr["SOH observed"]), name="Measured capacity loss", mode="markers",
                             marker=dict(color=P.text, size=4)))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * (1 - tr["SOH (model)"]), name="Model capacity loss",
                             line=dict(color=P.text, width=1.4, dash="dot")))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Share of initial capacity (%)", rangemode="tozero")
    return style_fig(fig, P, 420, f"Shared physics fit of {cell}: capacity loss split over life")


def nasa_physics_section() -> None:
    section("Level 4 · Shared physics on the NASA batteries (no neural network) · Which mechanisms explain the real "
            "capacity loss?")
    st.markdown(
        "A purely physics-based model, **without any neural network**: three degradation equations with 11 physical "
        "parameters estimated from the data. It is the model that **passed the recovery test** (simulated batteries with "
        "a known split: 1.1 share points of error, against 42–45 for the PINN): one set of physics parameters for "
        "**all** batteries, with SEI "
        "growth, lithium plating and loss of active material computed from their rate laws, so differences between "
        "batteries can only come from their **conditions** (temperature, charge and discharge current, depth of "
        "discharge). It is fitted to every battery of the constant-condition groups at once, from many random starts. "
        "Check before reading the split: the **fit error** (a battery the model cannot reproduce gets an unreliable "
        "split), the **agreement between starts** (is the answer unique?), and, if impedance is held out of the fit, "
        "whether the SEI share follows the **measured Re** it never saw.")
    groups = te.condition_groups(CT_RAW)
    cells = tuple(sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(CT_RAW, g)}))
    c1, c2, c3 = st.columns([1.6, 1.4, 0.7])
    ver = NP_VERSIONS[c1.radio("Model", list(NP_VERSIONS), key="np_ver",
                               help="Version 1 let plating run at any temperature and used it as a catch-all for steady "
                                    "fade at 30-40 °C. Version 2 follows electrochemistry: no plating above 15 °C cell "
                                    "temperature; no charge-current term, because NASA charged every battery at 1.5 A; "
                                    "a shared knee term for accelerating fade. Version 3 adds what version 2's forecasts "
                                    "missed (they were too optimistic): SEI growth with a learned exponent (steady growth "
                                    "on cracking particles is allowed) and plating triggered by lost graphite at any "
                                    "temperature, the classic cause of late-life knees. Version 3 reads any knee as "
                                    "triggered plating: the most common cause, not the only one.")]
    fit_with = NP_EVIDENCE[c2.radio("Fit the SEI to", list(NP_EVIDENCE), key="np_ev",
                                    help="The other resistance measurement is held out and only used as a check.")]
    restarts = c3.select_slider("Random starts", [10, 30], value=30, key="np_starts")
    st.caption(f"{len(cells)} batteries: " + ", ".join(cells))
    cfg = (DATA_KEY_RAW, cells, int(restarts), ver, fit_with)
    if st.button("Fit the shared physics to the NASA batteries", key="np_go", type="primary", icon=":material/science:",
                 disabled=len(cells) < 3):
        try:
            with st.spinner("Fitting one physics to all batteries from many starts…"):
                res = nasa_physics_cached(CT_RAW, imp, DATA_KEY_RAW, cells, int(restarts), ver, fit_with)
            st.session_state["np"] = {"cfg": cfg, "res": res}
        except Exception as exc:
            report_error("Shared physics fit failed", exc, debug)
    saved = st.session_state.get("np")
    if not saved:
        placeholder("Press 'Fit the shared physics to the NASA batteries': the split of every battery, the fit quality and "
                    "the checks appear here.")
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res = saved["res"]
    tab, s = te.nasa_physics_table(res, groups)
    L = s["learned"]
    lines = [f"{s['near-best starts']} of {s['starts']} starts reach the best fit; their splits differ by at most "
             f"{s['max spread']:.1f} share points" + (" (unique answer)." if s["max spread"] < 2 else " (NOT unique)."),
             f"Mean fit error {s['mean fit error']:.2f} SOH points over {len(tab)} batteries.",
             f"Learned physics (version {s['version']}): SEI activation energy {L['Ea']:.1f} kJ/mol; plating cold "
             f"sensitivity {L['b_C']:.2f}; LAM current sensitivity {L['a_I']:.2f}, depth sensitivity {L['a_V']:.2f}"
             + (f"; knee {L['beta0']:.2f} (depth {L['beta_V']:.2f})." if s["version"] == 2 else
                f"; SEI growth exponent {L['z']:.2f} (0.5 slows down, 1 steady, >1 accelerates); triggered plating "
                f"rate {L['k_tr']:.3g} above lost active material {100 * L['m_c']:.1f} %." if s["version"] == 3 else
                f"; charge-current sensitivity {L['a_c']:.2f}."),
             ("Parameters at a limit (NOT determined by the data): " + ", ".join(s["at_bound"]) + ".") if s["at_bound"]
             else "No parameter at a limit: all determined by the data."]
    for col, held, label in (("Re check: rank correlation SEI vs measured Re", not s["use_re"], "measured Re"),
                             ("R check: rank correlation SEI vs load-step resistance", not s["fit_r"], "load-step resistance")):
        if col in tab and held:
            rc = tab[col].dropna()
            if len(rc):
                lines.append(f"Held-out check: the model's SEI rises with the {label} on {int((rc > 0.5).sum())} of {len(rc)} "
                             "batteries (rank correlation > 0.5). Weak alone: any steadily rising quantity correlates; look "
                             "also at batteries whose resistance does NOT rise.")
    card("Shared physics on NASA", lines)
    show(fig_nasa_shares(tab, P), key="np_shares", data=tab)
    gm = tab.groupby("group")[[f"{m} (%)" for m in te.MECHANISMS] + ["capacity lost (%)", "fit error (SOH points)"]].mean()
    show_table(gm.style.format("{:.1f}"), note="Mean per group. Electrochemistry expects more SEI growth in Hot, more "
                                               "plating in Cold (charging in the cold), more loss of active material with "
                                               "deep discharge and high current.")
    show_table(tab.set_index("battery").style.format(precision=2, na_rep="—"),
               note="Fit error above ~2 SOH points: the model does not reproduce that battery well; read its split with "
                    "caution.")
    pick = st.selectbox("Battery to inspect", list(res["data"]), key="np_cell",
                        index=list(res["data"]).index(cell) if cell in res["data"] else 0)
    show(fig_nasa_traj(te.nasa_physics_trajectory(res, pick), P, pick), key="np_traj")
    download("Download the shared-physics results (CSV)",
             tab.assign(**{"app version": te.ENGINE_VERSION, "model version": s["version"], "Re in fit": s["use_re"],
                           "R_dc in fit": s["fit_r"], "parameters at a limit": " ".join(s["at_bound"]),
                           **{f"learned {k}": v for k, v in L.items()}}).to_csv(index=False).encode(),
             f"shared_physics_NASA__v{te.ENGINE_VERSION}.csv", "text/csv", key="np_csv")



# ---- Hybrid: shared physics (the explanation) + a small bounded network (what the physics misses) ----
HYB_UNEXPL_COLOR = "#9AA3AD"


@st.cache_data(show_spinner=False, max_entries=8)
def hybrid_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cells: Tuple[str, ...], restarts: int,
                  fit_with: str, lam: float, version: int = 3) -> Dict[str, Any]:
    return backed("hybrid", lambda: te.hybrid_portable(te.nasa_hybrid_fit(_ct, _imp, list(cells), fit_with=fit_with,
                                                                          restarts=restarts, lam=lam, version=version)),
                  _ct, _imp, cells, restarts, fit_with, lam, version)


@st.cache_data(show_spinner=False)
def hybrid_validation_cached(lam: float) -> pd.DataFrame:
    return te.hybrid_validation(restarts=10, lam=lam)


def fig_hybrid_shares(tab: pd.DataFrame, P: Palette) -> go.Figure:
    order = {"Hot": 0, "High current": 1, "Reference": 2, "Pulsed load": 3, "Cold": 4}
    t = tab.assign(_o=tab["group"].map(order).fillna(9)).sort_values(["_o", "battery"])
    fig = go.Figure()
    for m in te.MECHANISMS:
        fig.add_trace(go.Bar(x=[t["group"], t["battery"]], y=t[f"{m} (%)"], name=m, marker_color=MECH_COLORS[MECH_KEYS[m]],
                             hovertemplate="%{y:.0f} % of the capacity loss<extra>" + m + "</extra>"))
    col = f"{te.HYBRID_UNEXPLAINED} (%)"
    fig.add_trace(go.Bar(x=[t["group"], t["battery"]], y=t[col], name="Not explained by the physics (network)",
                         marker_color=HYB_UNEXPL_COLOR, marker_pattern_shape="/",
                         hovertemplate="%{y:.0f} % of the capacity loss<extra>not explained by the physics</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Share of the capacity loss (%)", range=[0, 100])
    return style_fig(fig, P, 460, "Hybrid: mechanism split (physics) and the part the physics does not explain (network)",
                     hovermode="closest")


def fig_hybrid_traj(tr: pd.DataFrame, P: Palette, cell: str) -> go.Figure:
    fig = go.Figure()
    for m in te.MECHANISMS:
        fig.add_trace(go.Scatter(x=tr["n"], y=100 * tr[m].clip(lower=0), name=m, stackgroup="mech", mode="lines",
                                 line=dict(width=0.6, color=MECH_COLORS[MECH_KEYS[m]])))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * tr[te.HYBRID_UNEXPLAINED].clip(lower=0), name="Not explained by the physics",
                             stackgroup="mech", mode="lines", line=dict(width=0.6, color=HYB_UNEXPL_COLOR)))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * (1 - tr["SOH observed"]), name="Measured capacity loss", mode="markers",
                             marker=dict(color=P.text, size=4)))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * (1 - tr["SOH physics"]), name="Physics alone",
                             line=dict(color=P.text, width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * (1 - tr["SOH (model)"]), name="Hybrid (physics + network)",
                             line=dict(color="#D55E00", width=2)))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Share of initial capacity (%)", rangemode="tozero")
    return style_fig(fig, P, 430, f"Hybrid fit of {cell}: physics mechanisms, network correction and measurement")


def _lobo_r2(df: pd.DataFrame, cols: List[str]) -> Tuple[float, np.ndarray]:
    """Leave-one-battery-out R² of an OLS of 'gap' on cols (+ intercept), and the last column's coefficient in every
    leave-one-out fit (its stability across batteries)."""
    pred, last = np.full(len(df), np.nan), []
    for b in df["battery"].unique():
        tr, te_ = df["battery"] != b, df["battery"] == b
        X = np.column_stack([np.ones(tr.sum())] + [df.loc[tr, c].to_numpy(float) for c in cols])
        beta = np.linalg.lstsq(X, df.loc[tr, "gap"].to_numpy(float), rcond=None)[0]
        Xt = np.column_stack([np.ones(te_.sum())] + [df.loc[te_, c].to_numpy(float) for c in cols])
        pred[te_.to_numpy()] = Xt @ beta
        last.append(beta[-1])
    y = df["gap"].to_numpy(float)
    return float(1 - np.nansum((y - pred) ** 2) / np.nansum((y - y.mean()) ** 2)), np.array(last)


def _bol_discharge_curve(c: str, depth: Optional[pd.DataFrame]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Beginning-of-life discharge curve of battery c: Ah delivered from full charge down to each voltage (median of
    its first 10 valid cycles; voltage as a running minimum). 3.95-3.10 V from the cycle table's Q(V) features,
    2.8-1.8 V from the within-battery depth table; 3.10-2.8 V interpolated. Returns (V ascending, Q)."""
    g = CT_RAW[(CT_RAW["Cell_ID"] == c) & ~CT_RAW["outlier"]].sort_values("n").head(10)
    if not len(g) or not all(col in g for col in te.QV_COLS):
        return None
    pts = [(float(v), float(q)) for v, q in zip(te.QV_GRID, g[te.QV_COLS].median()) if np.isfinite(q)]
    if depth is not None and len(depth):
        pts += [(float(r.V), float(r.Ah)) for r in depth[depth["battery"] == c].itertuples()]
    if len(pts) < 5:
        return None
    pts.sort()                                             # V ascending -> Q should decrease
    V = np.array([p_[0] for p_ in pts])
    Q = np.maximum.accumulate(np.array([p_[1] for p_ in pts])[::-1])[::-1]   # monotone: more Ah at lower voltage
    return V, Q


def physics_gap_section() -> None:
    """Step 0 of the PINN improvement plan (STATUS 5bj): is the capacity loss the physics misses driven by resistance
    growth? NASA measures capacity under load to a fixed cut-off: a growing resistance makes the voltage reach the
    cut-off earlier, an ACCELERATING loss of measured capacity that no material-degradation equation contains."""
    section("Diagnostic · is the loss the physics misses driven by resistance?")
    st.markdown("Every physics model so far is **optimistic late in life**: its equations slow down, real batteries "
                "speed up. Hypothesis: capacity is measured **under load to a fixed cut-off voltage**, so as resistance "
                "grows the extra voltage drop **x = current × (R − R at start)** reaches the cut-off earlier. Test: does "
                "x explain the **physics gap** (SOH predicted by the shared physics − SOH measured) **beyond "
                "throughput alone**, and do batteries with more resistance growth have a larger gap?")
    fleet = st.session_state.get("itw_fleet") or {}
    h = fleet.get("main")
    cells_an = tuple(sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(CT_RAW, g)}))
    if h is None:
        if not st.button("Run the resistance diagnostic", key="gap_go", icon=":material/troubleshoot:",
                         help="Uses the shared physics (version 2, as the Integrated twin; saved in the backup)."):
            placeholder("Press the button (or build the twin's fleet-wide parts on the Integrated twin page first).")
            return
        with st.spinner("Fitting the shared physics (saved, so only once)…"):
            try:
                h = hybrid_cached(CT_RAW, imp, DATA_KEY_RAW, cells_an, 30, "Re", 0.3, 2)
            except Exception as exc:
                report_error("Shared physics unavailable", exc, debug)
                return
    groups = te.condition_groups(CT_RAW)
    depth = None
    if store is not None:
        try:
            depth = m3_depth_cached(CT_RAW, DATA_KEY_RAW, tuple(sorted(h["data"])))
        except Exception:
            depth = None
    rows, ends = [], []
    for c, cd in h["data"].items():
        tr = te.hybrid_trajectory(h, c)
        r = CT_RAW[(CT_RAW["Cell_ID"] == c) & ~CT_RAW["outlier"]][["n", "R_dc_ohm"]].dropna().sort_values("n")
        if len(r) < 8 or not len(tr):
            continue
        rs = r.set_index("n")["R_dc_ohm"].rolling(7, center=True, min_periods=1).median()
        r0 = float(rs.head(5).median())
        d = tr[["n", "SOH observed", "SOH physics"]].copy()
        d["R"] = np.interp(d["n"], rs.index.to_numpy(float), rs.to_numpy(float))
        d["x (V)"] = float(cd["I_dis"]) * (d["R"] - r0)
        d["gap"] = d["SOH physics"] - d["SOH observed"]
        d["throughput (Ah)"] = np.interp(d["n"], np.asarray(cd["n"], float), np.asarray(cd["a"], float)) * te.PINN_A_SCALE
        g = str(groups.loc[c, "Group"]) if c in groups.index else "?"
        d["battery"], d["group"] = c, g
        rows.append(d)
        tail = d.tail(5)
        gap_end, x_end, i_dis = float(tail["gap"].mean()), float(tail["x (V)"].mean()), float(cd["I_dis"])
        # magnitude check (literature caveat: resistance knees may need very large resistance growth): the extra
        # voltage drop that would remove the unexplained capacity, read on this battery's own early discharge curve
        need_mv, need_r = np.nan, np.nan
        cur = _bol_discharge_curve(c, depth)
        c_bol = float(CT_RAW.loc[CT_RAW["Cell_ID"] == c, "C_bol_Ah"].iloc[0]) if "C_bol_Ah" in CT_RAW else np.nan
        v_cut = float(CT_RAW[(CT_RAW["Cell_ID"] == c) & ~CT_RAW["outlier"]].sort_values("n").head(10)["V_min_V"].median())
        if cur is not None and gap_end > 0 and np.isfinite(c_bol) and np.isfinite(v_cut):
            V, Q = cur
            q_cut = float(np.interp(v_cut, V, Q))
            q_star = q_cut - gap_end * c_bol
            if q_star > Q.min():
                v_star = float(np.interp(q_star, Q[::-1], V[::-1]))
                need_mv = 1000 * max(v_star - v_cut, 0.0)
                need_r = 100 * (need_mv / 1000) / (i_dis * r0) if r0 > 0 and i_dis > 0 else np.nan
        ends.append({"group": g, "battery": c, "gap at end (SOH pts)": 100 * gap_end,
                     "R growth (%)": 100 * (tail["R"].mean() / r0 - 1), "x at end (mV)": 1000 * x_end,
                     "x needed (mV)": need_mv, "measured / needed": (1000 * x_end / need_mv) if need_mv > 0 else np.nan,
                     "R growth needed (%)": need_r,
                     "within-battery rank corr.": d["gap"].corr(d["x (V)"], method="spearman")})
    if not rows:
        st.info("No battery has both a physics trajectory and DC resistance data.")
        return
    D, E = pd.concat(rows, ignore_index=True), pd.DataFrame(ends)
    res = []
    for g, dg in list(D.groupby("group")) + [("all", D)]:
        eg = E if g == "all" else E[E["group"] == g]
        if dg["battery"].nunique() < 3:
            continue
        dg = dg.assign(a2=dg["throughput (Ah)"] ** 2)
        r_time, _ = _lobo_r2(dg, ["throughput (Ah)", "a2"])
        r_both, bx = _lobo_r2(dg, ["throughput (Ah)", "a2", "x (V)"])
        reduction = 1 - (1 - r_both) / (1 - r_time) if r_time < 1 else 0.0
        X = np.column_stack([np.ones(len(dg)), dg["throughput (Ah)"], dg["a2"], dg["x (V)"]])
        beta = np.linalg.lstsq(X, dg["gap"].to_numpy(float), rcond=None)[0]
        rho = eg["gap at end (SOH pts)"].corr(eg["x at end (mV)"], method="spearman") if len(eg) >= 4 else np.nan
        res.append({"group": g, "batteries": dg["battery"].nunique(),
                    "measured / needed (median)": float(eg["measured / needed"].median())
                    if eg["measured / needed"].notna().any() else np.nan,
                    "across batteries: rank corr. (gap vs x)": rho,
                    "R² throughput only (leave-one-out)": r_time, "R² + resistance term": r_both,
                    "unexplained error removed by resistance": reduction,
                    "effect positive in every leave-one-out fit": bool((bx > 0).all()),
                    "gap per 10 mV of x (SOH pts)": 100 * beta[3] * 0.01})
    R = pd.DataFrame(res).set_index("group")
    grp = R.drop(index="all", errors="ignore")
    # criteria calibrated by simulation (3-5 batteries): detect a real effect 76-98 %, false alarm 0-4 % (STATUS 5bj)
    ok = grp[(grp["unexplained error removed by resistance"] >= 0.15)
             & grp["effect positive in every leave-one-out fit"]
             & (grp["measured / needed (median)"].fillna(0) >= 0.5)]  # and the measured drop is large enough
    verdict = ("supported" if len(ok) >= 2 else "partly supported" if len(ok) == 1 else "not supported")
    col = {"supported": "#2E9E6E", "partly supported": "#E69F00", "not supported": "#9AA3AD"}[verdict]
    st.markdown(f"<div style='padding:10px 14px;border-radius:8px;background:{col}22;border-left:6px solid {col}'>"
                f"<b>Hypothesis {verdict}</b> · groups meeting both tests: "
                f"{html.escape(', '.join(ok.index) or 'none')} (the resistance term removes ≥ 15 % of the error "
                f"left by throughput alone, leave-one-battery-out, its effect is positive in every fit, and the "
                f"measured voltage drop covers ≥ 50 % of the drop needed)</div>", unsafe_allow_html=True)
    st.caption("Magnitude check: the voltage drop needed is read on each battery's own early discharge curve "
               "(3.95-3.10 V from the Q(V) features, 2.8-1.8 V from the raw traces, 3.10-2.8 V interpolated) as the "
               "rise of the effective cut-off that removes the capacity the physics misses. Measured = I x (R - R at "
               "start) from the load-step resistance, which misses part of the charge-transfer loss: hence the factor 2.")
    show_table(R.style.format("{:.2f}", na_rep="—", subset=[c for c in R.columns if c not in
                                                             ("batteries", "effect positive in every leave-one-out fit")])
               .format("{:.0f}", subset=["batteries"]),
               note="Leave-one-battery-out: each battery's gap is predicted from the others, so a term only helps if it "
                    "generalises across batteries. Throughput and its square stand for 'time' (both grow over life). "
                    "The rank correlation across batteries is descriptive only: with 3-5 batteries it is unreliable.")
    fig = go.Figure()
    for g, dg in D.groupby("group"):
        fig.add_trace(go.Scatter(x=1000 * dg["x (V)"], y=100 * dg["gap"], mode="markers", name=g, opacity=0.55,
                                 marker=dict(size=5), text=dg["battery"],
                                 hovertemplate="%{text} · x %{x:.0f} mV · gap %{y:.1f} pts<extra></extra>"))
    fig.update_xaxes(title_text="Extra voltage drop x = I × (R − R at start) (mV)")
    fig.update_yaxes(title_text="Physics gap: SOH physics − SOH measured (points)")
    show(style_fig(fig, P, 460, "Does the physics gap grow with the resistance-driven voltage drop?"), key="gap_fig", data=D)
    show_table(E.set_index(["group", "battery"]).style.format("{:.2f}", na_rep="—"))
    st.session_state["gap_diag"] = {"summary": R, "per_battery": E, "verdict": verdict}
    download("Download the diagnostic (CSV)", D.to_csv(index=False).encode(),
             f"physics_gap_resistance__v{te.ENGINE_VERSION}.csv", "text/csv", key="gap_csv")


def hybrid_section() -> None:
    section("Explanation · Hybrid: shared physics + a small neural network (what the physics misses)")
    st.markdown(
        "The **shared physics** (section above) computes SEI growth, lithium plating and loss of active material and is "
        "**frozen**; a **small neural network** then learns only what the physics leaves unexplained (measurement minus "
        "physics). Its correction is zero at the start of life, bounded (±25 SOH points), penalised towards zero, and shown openly as a fourth "
        "category, **not explained by the physics**. So the mechanism split stays the physics' split, the network adds "
        "accuracy, and a battery the network must correct strongly is a battery for which the physics is missing "
        "something (e.g. a knee). Why frozen: on simulated batteries with a known split, every variant in which the "
        "network also shaped the physics made the split worse; the network improves the fit, not the explanation.")
    cells = tuple(sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(CT_RAW, g)}))
    hv = {"Physics version 3 (recommended)": 3, "Physics version 2": 2}[st.radio(
        "Physics inside the hybrid", ["Physics version 3 (recommended)", "Physics version 2"], key="hy_ver",
        horizontal=True)]
    c1, c2, c3 = st.columns([1.6, 0.8, 0.9])
    fit_with = NP_EVIDENCE[c1.radio("Fit the SEI to", list(NP_EVIDENCE), key="hy_ev")]
    restarts = c2.select_slider("Random starts", [10, 30], value=30, key="hy_starts")
    lam = c3.select_slider("Network penalty", [0.1, 0.3, 1.0], value=0.3, key="hy_lam",
                           help="Higher: the network corrects less. The mechanism split does not depend on it.")
    with st.expander("Check the hybrid on simulated batteries with a known split (about 10 seconds)"):
        st.caption("11 batteries follow the shared physics exactly; three of them (R2, C2, H3) also get a knee the "
                   "physics cannot describe. A good hybrid keeps the physics' split and fits the knees better.")
        if st.button("Run the check", key="hy_val"):
            with st.spinner("Fitting physics and network on the simulated batteries…"):
                st.session_state["hy_val_res"] = hybrid_validation_cached(float(lam))
        v = st.session_state.get("hy_val_res")
        if v is not None:
            k = v["battery"].isin(te.HYBRID_KNEE_CELLS)
            card("Check on simulated batteries", [
                f"Split error (share points, mean): physics alone {v['split error, physics alone (points)'].mean():.2f}, "
                f"hybrid {v['split error, hybrid (points)'].mean():.2f} (identical by design).",
                f"Fit error on the knee batteries: physics alone {v.loc[k, 'fit error, physics alone (SOH points)'].mean():.2f}, "
                f"hybrid {v.loc[k, 'fit error, hybrid (SOH points)'].mean():.2f} SOH points; on the others "
                f"{v.loc[~k, 'fit error, physics alone (SOH points)'].mean():.2f} and "
                f"{v.loc[~k, 'fit error, hybrid (SOH points)'].mean():.2f}.",
                "The knees bend the shared physics a little for every battery (split error not zero): a limit of "
                "any shared model when some batteries behave differently."])
            show_table(v.set_index("battery").style.format("{:.2f}"))
    cfg = (DATA_KEY_RAW, cells, int(restarts), fit_with, float(lam), hv)
    if st.button("Fit the hybrid to the NASA batteries", key="hy_go", type="primary", icon=":material/hub:",
                 disabled=len(cells) < 3):
        try:
            with st.spinner("Shared physics first, then the network on what it leaves…"):
                h = hybrid_cached(CT_RAW, imp, DATA_KEY_RAW, cells, int(restarts), fit_with, float(lam), hv)
            st.session_state["hy"] = {"cfg": cfg, "h": h}
        except Exception as exc:
            report_error("Hybrid fit failed", exc, debug)
    saved = st.session_state.get("hy")
    if not saved:
        placeholder("Press 'Fit the hybrid to the NASA batteries': the split with the unexplained part, the fit "
                    "improvement and the batteries the physics does not describe appear here.")
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    h = saved["h"]
    groups = te.condition_groups(CT_RAW)
    tab, s = te.hybrid_table(h, groups)
    need = s["batteries needing the network"]
    card("Hybrid on NASA", [
        f"Mean fit error: physics alone {s['mean fit physics']:.2f}, hybrid {s['mean fit hybrid']:.2f} SOH points.",
        ("Batteries the physics does not describe well (network correction above 2 SOH points RMS): " + ", ".join(need) + "."
         ) if need else "No battery needs a large network correction: the physics describes all of them.",
        "The SEI / plating / LAM split is the shared physics' split (frozen); the grey part is what the network adds."])
    show(fig_hybrid_shares(tab, P), key="hy_shares", data=tab)
    show_table(tab.set_index("battery").style.format(precision=2, na_rep="—"),
               note="Physics over-predicts the loss: the network removes loss the physics predicts but the battery did "
                    "not show (shown as a number, not in the bars).")
    pick = st.selectbox("Battery to inspect", list(h["data"]), key="hy_cell",
                        index=list(h["data"]).index(cell) if cell in h["data"] else 0)
    show(fig_hybrid_traj(te.hybrid_trajectory(h, pick), P, pick), key="hy_traj")
    download("Download the hybrid results (CSV)",
             tab.assign(**{"app version": te.ENGINE_VERSION, "physics version": h.get("version", 2),
                           "SEI fitted to": fit_with, "network penalty": lam,
                           **{f"learned {k_}": v_ for k_, v_ in s["learned"].items()}}).to_csv(index=False).encode(),
             f"hybrid_NASA__v{te.ENGINE_VERSION}.csv", "text/csv", key="hy_csv")


def pinn_section() -> None:
    model_level_section(
        4, "Level 4 · Physics-informed neural network: does knowing the physics improve the prediction?",
        "The network learns three hidden degradation states (lithium lost to **SEI growth**, lithium lost to **plating**, "
        "**loss of active material**) from all training batteries and the selected battery's own cycles. Electrochemistry "
        "links them to what is measured (SOH = (1 − LLI) × (1 − LAM); resistance grows with the SEI film), and the "
        "**degradation equations** are enforced, also over the battery's future: SEI growth that slows as the film "
        "thickens and speeds up with temperature (Arrhenius); plating that grows in the cold, at high current and as "
        "active material is lost (the cause of knees); loss of active material with cycling. The **same network "
        "without physics** is included as an ablation, so you can see what the physics itself contributes. Forecast "
        "mode only: the PINN models how the battery evolves. **PINN twin core** is the best PINN before the GA: the "
        "cut-off voltage as an input with a depth term, a start that ignores NASA's low first cycle, and 3 averaged "
        "trainings. **GA-PINN** is the same physics-informed network with the "
        "settings a genetic algorithm chose for this battery: network size, learning rate, physics weight, the form "
        "of the SEI law and an optional acceleration term, judged only on the other batteries (nested leave-one-out). "
        "The search runs here for the selected battery, forecast start and training batteries, like any other model "
        "(a few minutes per battery the first time on NASA; every evaluation is saved, so a re-run is instant).",
        {"PINN": te.PINN_MODELS["PINN"],
         "PINN-twin": "PINN twin core · best PINN before the GA (cut-off input, burn-in start, 3 seeds)",
         "GA-PINN": "GA-PINN · this battery's genetic-algorithm winner (settings and physics form chosen by the GA)"},
        ("PINN", "PINN-twin", "GA-PINN"),
        lambda t, n0, m, mode, train, model_params=None, **_: te.train_pinn_model(ct, imp, t, n0, m, mode, train,
                                                                                  float(eol_ah), model_params),
        "pinn", modes=("forecast",))
    sc = scheme()
    show_t = sc.get("show", sc["targets"][0])
    res = (st.session_state.get("pinn", {}).get("forecast", {}) or {}).get(show_t, [])
    phys = [f for f in res if getattr(f, "mechanisms", None) is not None and f.model.endswith("physics-informed")]
    if phys:
        f = phys[0]
        show(fig_mechanisms(f, P, show_t), key="pinn_mech", export=False)
        learned = {k: v for k, v in f.inputs_at_origin.items() if not k.startswith("At the last cycle")}
        show_table(pd.DataFrame({"Learned physics": list(learned), "Value": list(learned.values())})
                   .set_index("Learned physics").style.format({"Value": "{:.4g}"}),
                   note="Fitted across the training batteries. Effects of conditions that do not vary among the "
                        "training batteries (for example cold or current within one group) cannot be learned: they stay "
                        "at their starting values.")


# ------------------------------------------------------------------------------------------------------- SOH estimation
EST_GROUPS = ("Reference", "Hot", "Cold", "Pulsed load", "High current")
EST_CHOICES = {"GPR": "GPR · Gaussian process on honest indicators",
               "PINN-est": "Estimation PINN · physics-informed, default settings",
               "GA-PINN-est": "Estimation GA-PINN · settings chosen by the GA on the training batteries",
               "GA-PINN-GPR": "GA-PINN + GPR · weighted by their validation errors on the training batteries"}
EST_FINAL_SEEDS = 3          # the final estimate averages 3 trainings (the GA uses 1 per candidate, as in Colab)


@st.cache_data(show_spinner=False, max_entries=600)
def est_cached(_ct: pd.DataFrame, key: str, cell: str, train: Tuple[str, ...], kind: str, params_json: str) -> Dict[str, Any]:
    params = json.loads(params_json)
    ver = getattr(te, "ESTIMATION_VERSION", "1")           # a new estimator version never reuses old saved results
    return backed(f"soh_est_v{ver}", lambda: te.estimate_soh(_ct, cell, list(train), kind, params or None),
                  key, cell, train, kind, params_json)


@st.cache_data(show_spinner=False, max_entries=100)
def est_ga_cached(_ct: pd.DataFrame, key: str, train: Tuple[str, ...], pop: int, gens: int) -> Dict[str, Any]:
    return backed("soh_est_ga", lambda: te.ga_estimation(_ct, list(train), pop, gens), key, train, pop, gens)


def ga_est_resumable(train_t: Tuple[str, ...], deadline: Optional[float], note: Optional[Any] = None) -> Dict[str, Any]:
    """GA for the estimation PINN, resumable: every PINN training (one held-out training battery, one setting) is saved
    the moment it finishes, so a press stopped at the time budget continues on the next press. Fitness = leave each
    TRAINING battery out: RMSE + 0.5 |late bias| (the battery being estimated is not among them)."""
    count = {"n": 0}
    total = 16 * len(train_t)
    rmse_of: Dict[str, float] = {}

    def evaluate(g: Dict[str, Any]) -> float:
        gj = json.dumps({k: (int(v) if k == "accel" else v) for k, v in g.items()}, sort_keys=True, default=float)
        err, late = [], []
        for v in train_t:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError
            r = est_cached(CT_RAW, DATA_KEY, v, tuple(sorted(x for x in train_t if x != v)), "PINN-est", gj)
            count["n"] += 1
            if note is not None:
                note.caption(f"GA training {count['n']} of about {total} (each saved; a stopped press continues)")
            e = r["pred"] - r["soh"]; err.append(e); late.append(float(np.mean(e[-max(1, len(e) // 3):])))
        rmse, lb = 100 * float(np.sqrt(np.mean(np.concatenate(err) ** 2))), 100 * float(np.mean(late))
        rmse_of[gj] = rmse
        return rmse + 0.5 * abs(lb)
    try:
        run = te.ga_pinn_run(evaluate, 6, 4, seed=0, genes=getattr(te, "EST_GENES", None), start=getattr(te, "EST_START", None))
    except TimeoutError:
        return {"done": False}
    best = dict(run["best"]) if run.get("best") else dict(getattr(te, "EST_START", te.GA_PINN_START))
    best["accel"] = int(bool(best.get("accel", 0)))
    bj = json.dumps({k: (int(v) if k == "accel" else v) for k, v in best.items()}, sort_keys=True, default=float)
    return {"done": True, "best": best, "val_rmse": rmse_of.get(bj, float("nan"))}


def gpr_val_rmse(train_t: Tuple[str, ...]) -> float:
    """The GPR's validation RMSE (SOH points) on the training batteries, each left out in turn."""
    err = []
    for v in train_t:
        r = est_cached(CT_RAW, DATA_KEY, v, tuple(sorted(x for x in train_t if x != v)), "GPR", "null")
        err.append(r["pred"] - r["soh"])
    return 100 * float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def run_estimator(cell: str, train: Sequence[str], choice: str, deadline: Optional[float] = None,
                  note: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """One estimator for one battery; the battery's capacity / SOH is never used (engine test). Returns None when
    the GA-PINN search stopped at the time budget (press again to continue)."""
    train_t = tuple(sorted(train))
    if choice in ("GA-PINN-est", "GA-PINN-GPR"):
        ga = ga_est_resumable(train_t, deadline, note)
        if not ga["done"]:
            return None
        g = ga["best"]
        r = dict(est_cached(CT_RAW, DATA_KEY, cell, train_t, "PINN-est",
                            json.dumps({**g, "seeds": EST_FINAL_SEEDS}, sort_keys=True, default=float)))
        r["label"] = "Estimation GA-PINN · " + (te.est_key(g) if hasattr(te, "est_key") else te.ga_pinn_key(g).split("|", 1)[1])
        if choice == "GA-PINN-GPR":
            rg = est_cached(CT_RAW, DATA_KEY, cell, train_t, "GPR", "null")
            rm_p, rm_g = ga["val_rmse"], gpr_val_rmse(train_t)
            w = (1 / rm_p ** 2) / (1 / rm_p ** 2 + 1 / rm_g ** 2) if np.isfinite(rm_p) and np.isfinite(rm_g) else 0.5
            pa = w * np.asarray(r["pred_all"]) + (1 - w) * np.asarray(rg["pred_all"])
            ok = np.isin(r["n_all"], r["n"])
            y = np.asarray(r["soh"]); pr = pa[ok]
            far, _ = te._est_hampel(pd.Series(y), window=7, k=3.0, min_abs=0.015)
            keep = ~far.to_numpy(bool)
            ss, ssk = float(np.sum((y - y.mean()) ** 2)), float(np.sum((y[keep] - y[keep].mean()) ** 2))
            r.update({"pred_all": pa, "pred": pr, "std": None, "rmse": 100 * float(np.sqrt(np.mean((pr - y) ** 2))),
                      "r2": 1 - float(np.sum((y - pr) ** 2)) / ss if ss > 0 else float("nan"),
                      "rmse_clean": 100 * float(np.sqrt(np.mean((pr[keep] - y[keep]) ** 2))),
                      "r2_clean": 1 - float(np.sum((y[keep] - pr[keep]) ** 2)) / ssk if ssk > 0 else float("nan"),
                      "label": f"GA-PINN + GPR · weights {w:.2f} / {1 - w:.2f} (validation {rm_p:.2f} / {rm_g:.2f})"})
    else:
        params = json.dumps({"seeds": EST_FINAL_SEEDS}) if choice == "PINN-est" else "null"
        r = dict(est_cached(CT_RAW, DATA_KEY, cell, train_t, choice, params))
        r["label"] = EST_CHOICES[choice]
    return r


def soh_estimation_section() -> None:
    """Mission 2 · estimation: what is the SOH now, from this cycle's measurements, without a capacity test."""
    section("Mission 2 · SOH estimation · today's SOH from today's measurements")
    st.markdown(
        "Each cycle's **measured indicators** (charge between 3.8 and 3.5 V on discharge, CC and CV charge times; "
        "cleaned, smoothed, relative to the battery's own first cycles) give its SOH **now**, without a capacity "
        "test. Models are trained on the **other batteries of the group**; the battery being estimated never "
        "contributes its capacity (an engine test changes it and checks that the estimate is identical). "
        "Discharge time and energy are not used: at constant current they are the capacity itself. "
        "Forecasting the future is the next section; this estimate is what the twin can use in field mode and for "
        "fault detection.")
    avail = [g for g in EST_GROUPS if len(te.protocol_cells(CT_RAW, g)) >= 3]
    if not avail:
        placeholder("No condition group has three or more batteries.")
        return
    c1, c2, c3 = st.columns([1, 1, 2.2])
    grp = c1.selectbox("Group", avail, key="est_group")
    cells = te.protocol_cells(CT_RAW, grp)
    cell = c2.selectbox("Battery to estimate", cells, key="est_cell")
    models = c3.multiselect("Estimators", list(EST_CHOICES), default=["GPR", "PINN-est"], format_func=EST_CHOICES.get,
                            key="est_models",
                            help="The GA-PINN option runs a GA on the training batteries first (several minutes the "
                                 "first time; saved afterwards).")
    train = [x for x in cells if x != cell]
    budget = 15
    if "GA-PINN-est" in models or "GA-PINN-GPR" in models:
        budget = st.number_input("Time budget per press for the GA-PINN search (minutes)", 2, 60, 15, key="est_budget",
                                 help="The GA trains about 16 settings x each training battery; every training is "
                                      "saved, so a press that reaches the budget continues on the next press.")
    b1, b2 = st.columns(2)
    if b1.button("Estimate this battery", key="est_go", type="primary", icon=":material/monitor_heart:"):
        res, note, deadline = {}, st.empty(), time.time() + 60 * float(budget)
        for m in models:
            with st.spinner(f"{EST_CHOICES[m]}…"):
                try:
                    r = run_estimator(cell, train, m, deadline, note)
                    if r is None:
                        st.info("The GA-PINN search reached the time budget: press **Estimate this battery** again "
                                "to continue (finished trainings are saved).")
                    else:
                        res[m] = r
                except Exception as exc:
                    report_error(f"{m} on {cell}", exc, False)
        note.empty()
        st.session_state["est_one"] = {"key": (DATA_KEY, grp, cell, tuple(models)), "res": res}
    if b2.button("Every battery of the group (each held out)", key="est_all", icon=":material/groups:"):
        rows, bar = [], st.progress(0.0)
        for i, c in enumerate(cells):
            for m in models:
                bar.progress((i + 1) / len(cells), text=f"{c} · {EST_CHOICES[m]}")
                try:
                    r = run_estimator(c, [x for x in cells if x != c], m, time.time() + 60 * float(budget))
                    if r is None:
                        st.info(f"{c}: the GA-PINN search reached the time budget; press again to continue.")
                        continue
                    rows.append({"battery": c, "estimator": EST_CHOICES[m], "RMSE (SOH pts)": r["rmse"], "R²": r["r2"],
                                 "RMSE (far points excluded)": r.get("rmse_clean", np.nan)})
                except Exception as exc:
                    report_error(f"{m} on {c}", exc, False)
        bar.empty()
        st.session_state["est_all_res"] = {"key": (DATA_KEY, grp, tuple(models)), "rows": rows}
    one = st.session_state.get("est_one")
    if one and one["key"] == (DATA_KEY, grp, cell, tuple(models)) and one["res"]:
        fig = go.Figure()
        first = next(iter(one["res"].values()))
        fig.add_trace(go.Scatter(x=first["n"], y=100 * first["soh"], mode="markers", name=f"measured SOH · {cell}",
                                 marker=dict(size=5, color="#9AA3AD")))
        colors = {"GPR": "#CC79A7", "PINN-est": "#56B4E9", "GA-PINN-est": "#009E73"}
        rows = []
        for m, r in one["res"].items():
            col = colors.get(m, "#E69F00")
            if r.get("std") is not None:
                fig.add_trace(go.Scatter(x=np.r_[r["n"], r["n"][::-1]], y=100 * np.r_[r["pred"] + 1.96 * r["std"],
                                         (r["pred"] - 1.96 * r["std"])[::-1]], fill="toself", opacity=0.15,
                                         line=dict(width=0), fillcolor=col, name=f"{m} · 95 % band", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=r["n"], y=100 * r["pred"], mode="lines", line=dict(color=col, width=2.6),
                                     name=r["label"]))
            rows.append({"estimator": r["label"], "R²": r["r2"], "RMSE (SOH pts)": r["rmse"],
                         "R² (far points excluded)": r.get("r2_clean", np.nan),
                         "RMSE (far points excluded)": r.get("rmse_clean", np.nan),
                         "training batteries": ", ".join(r["train"])})
        fig.update_xaxes(title_text="Cycle"); fig.update_yaxes(title_text="SOH (%)")
        show(style_fig(fig, P, 460, f"{cell}: SOH estimated from its measurements (none of its capacity used)"),
             key="est_fig")
        show_table(pd.DataFrame(rows).set_index("estimator").style.format(
                       {"R²": "{:.3f}", "RMSE (SOH pts)": "{:.2f}", "R² (far points excluded)": "{:.3f}",
                        "RMSE (far points excluded)": "{:.2f}"}),
                   note="Scored on the battery's valid cycles. 'Far points excluded': cycles more than 3 robust std (and "
                        "1.5 SOH points) from their neighbours are left out of the SCORE only (many are real regeneration "
                        "bumps after a rest; the model never sees any capacity). RMSE is the better measure for batteries "
                        "that barely fade.")
    allr = st.session_state.get("est_all_res")
    if allr and allr["key"] == (DATA_KEY, grp, tuple(models)) and allr["rows"]:
        A = pd.DataFrame(allr["rows"])
        piv = A.pivot_table(index="battery", columns="estimator", values="RMSE (SOH pts)")
        piv.loc["MEAN"] = piv.mean()
        show_table(piv.style.format("{:.2f}"), note=f"{grp}: RMSE (SOH points), each battery held out in turn.")
        download("Download (CSV)", A.to_csv(index=False).encode(), f"soh_estimation_{grp}__v{te.ENGINE_VERSION}.csv",
                 "text/csv", key="est_csv")
    st.selectbox("Estimator the twin will use for its field mode and fault detection (next release)", list(EST_CHOICES),
                 format_func=EST_CHOICES.get, key="twin_estimator",
                 help="Kept for the next step: the twin re-anchored to this estimate when no capacity test exists.")


def view_models() -> None:
    recommendations("models")
    soh_estimation_section()            # Mission 2 · estimation (today's SOH from today's measurements)
    section("Forecasting models: from baselines to physics-informed networks (levels 1–4)")
    sc = scheme_controls()
    if not sc["targets"]:
        return
    ladder_intro()
    ladder_baselines()          # Level 1 · baselines
    trajectory_section()        # Level 2 · machine learning
    deep_section()              # Level 3 · deep learning
    pinn_section()              # Level 4 · PINN and GA-PINN
    ladder_leaderboard()        # every level on this battery's test cycles
    assessment_section()        # every battery of the group (the evidence)
    hybrid_section()            # explanation: which mechanisms caused the loss (not a forecaster)
    # removed from the page at the user's request (2026-11-04): nasa_physics_section (shared physics alone; the twin
    # and Mission 3 still use it through the fleet parts) and physics_gap_section (resistance diagnostic, ruled out:
    # STATUS 5bl); both functions are kept


# =============================================================================
# View 3: operations & optimal control
# =============================================================================

# ---- Mission 3 · Anomaly detection (two indicators, robust to capacity regeneration) ----
@st.cache_data(show_spinner=False)
def anomaly_validation2_cached(kappa_thr: float, z_thr: float) -> pd.DataFrame:
    return te.anomaly_validation2(restarts=6, kappa_thr=kappa_thr, z_thr=z_thr)


@st.cache_data(show_spinner=False, max_entries=8)
def anomaly_scan2_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cells: Tuple[str, ...],
                         kappa_thr: float, z_thr: float) -> Dict[str, pd.DataFrame]:
    return backed("anomaly", lambda: te.nasa_anomaly_scan2(_ct, _imp, list(cells), restarts=6, kappa_thr=kappa_thr,
                                                           z_thr=z_thr),
                  _ct, _imp, cells, kappa_thr, z_thr)


def fig_anomaly2(tr: pd.DataFrame, P: Palette, cell: str, kappa_thr: float, z_thr: float,
                 height: int = 620, n_max: Optional[float] = None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.44, 0.28, 0.28],
                        subplot_titles=("Measured SOH",
                                        "FLEET indicator: ageing speed κ vs the fleet's physics",
                                        "OWN indicator: is the fade slope steeper than in the previous window? (z)"))
    fig.add_trace(go.Scatter(x=tr["n"], y=100 * tr["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.text, size=4)), 1, 1)
    fig.add_trace(go.Scatter(x=tr["n"], y=tr["kappa"], name="κ", line=dict(color="#AA4499", width=2)), 2, 1)
    fig.add_hline(y=kappa_thr, line=dict(color="#AA4499", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=1.0, line=dict(color=P.text, dash="dot", width=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=tr["n"], y=tr["slope z"], name="slope z", line=dict(color="#D55E00", width=2)), 3, 1)
    fig.add_hline(y=z_thr, line=dict(color="#D55E00", dash="dash", width=1), row=3, col=1)
    for key, col, lab in (("first_fleet_alarm", "#AA4499", "FLEET alarm"), ("first_own_alarm", "#D55E00", "OWN alarm")):
        x = tr.attrs.get(key)
        if x is not None:
            for r in (1, 2, 3):
                fig.add_vline(x=x, line=dict(color=col, width=1.5), row=r, col=1)
            fig.add_annotation(x=x, y=1, yref="y domain", text=f"{lab} at cycle {x:.0f}", showarrow=False,
                               xanchor="left", font=dict(color=col), row=1, col=1)
    fig.update_yaxes(title_text="SOH (%)", row=1, col=1)
    fig.update_yaxes(title_text="κ", row=2, col=1)
    fig.update_yaxes(title_text="z", row=3, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=3, col=1)
    if n_max is not None:
        fig.update_xaxes(range=[0, n_max])
    return style_fig(fig, P, height, f"Anomaly detection replay · {cell}")



DETECTION_COLORS = {"Normal": "#9AA3AD", "Regeneration (normal)": "#56B4E9",
                    "FLEET alarm: ages faster than the fleet": "#AA4499",
                    "OWN alarm: fade steeper than its own past": "#D55E00", "Common test event": "#E69F00",
                    "FLEET + OWN alarm": "#CC0000", "Past end of life": "#6B1E1E"}


def fig_detection_live(tr: pd.DataFrame, status: List[str], n: int, P: Palette, cell: str, kappa_thr: float,
                       z_thr: float, removed: Optional[pd.DataFrame] = None, soh_eol: Optional[float] = None,
                       n_eol: Optional[float] = None) -> go.Figure:
    n_last = float(tr["n"].max())
    now = tr[tr["n"] <= n]
    st_now = status[:len(now)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.56, 0.22, 0.22],
                        subplot_titles=(f"{cell} · state of health, coloured by what the detector concluded at each cycle",
                                        "FLEET · ageing speed κ vs the fleet", "OWN · fade slope vs its own past (z)"))
    for lab, col in DETECTION_COLORS.items():
        m = [s_ == lab for s_ in st_now]
        if any(m):
            sub = now[m]
            fig.add_trace(go.Scatter(x=sub["n"], y=100 * sub["SOH"], mode="markers", name=lab,
                                     marker=dict(color=col, size=9 if lab != "Normal" else 6,
                                                 line=dict(width=0.5, color=P.plot_bg) if lab != "Normal" else None),
                                     hovertemplate="cycle %{x}: %{y:.1f} %<extra>" + lab + "</extra>"), 1, 1)
    if removed is not None and len(removed):
        rm = removed[removed["n"] <= n]
        if len(rm):
            fig.add_trace(go.Scatter(x=rm["n"], y=100 * rm["SOH"], mode="markers", name="Removed by data cleaning",
                                     marker=dict(color="rgba(0,0,0,0)", size=7, line=dict(color="#9AA3AD", width=1.2)),
                                     hovertemplate="cycle %{x}: %{y:.1f} % (removed: recovery after a rest or invalid "
                                                   "measurement; not used by the detector)<extra></extra>"), 1, 1)
    if soh_eol is not None and np.isfinite(soh_eol):
        fig.add_hline(y=100 * soh_eol, line=dict(color="#6B1E1E", dash="dash", width=1.2), row=1, col=1)
        fig.add_annotation(x=0, y=100 * soh_eol, xref="x", yref="y", text=f"end of life {100 * soh_eol:.1f} %",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(color="#B04040", size=11),
                           row=1, col=1)
        if n_eol is not None and n_eol <= n:
            fig.add_vline(x=n_eol, line=dict(color="#B04040", width=1.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=now["n"], y=now["kappa"], name="κ", line=dict(color="#AA4499", width=2), showlegend=False), 2, 1)
    fig.add_hline(y=kappa_thr, line=dict(color="#AA4499", dash="dash", width=1), row=2, col=1)
    fig.add_trace(go.Scatter(x=now["n"], y=now["slope z"], name="z", line=dict(color="#D55E00", width=2), showlegend=False), 3, 1)
    fig.add_hline(y=z_thr, line=dict(color="#D55E00", dash="dash", width=1), row=3, col=1)
    for r in (1, 2, 3):
        fig.add_vline(x=n, line=dict(color=P.text, width=1, dash="dot"), row=r, col=1)
    fig.update_xaxes(range=[0, n_last * 1.02])
    y_all = 100 * pd.concat([tr["SOH"], removed["SOH"] if removed is not None and len(removed) else tr["SOH"]]).dropna()
    if len(y_all):
        lo_y = min(float(y_all.min()), 100 * soh_eol if soh_eol is not None and np.isfinite(soh_eol) else 999)
        fig.update_yaxes(range=[lo_y - 2, float(y_all.max()) + 2], row=1, col=1)
    fig.update_yaxes(title_text="SOH (%)", row=1, col=1)
    fig.update_yaxes(title_text="κ", row=2, col=1)
    fig.update_yaxes(title_text="z", row=3, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=3, col=1)
    return style_fig(fig, P, 640, f"Live detection replay · {cell} · cycle {n}")


def detection_live_replay(tr: pd.DataFrame, common: Optional[Dict[float, List[str]]], cell_id: str,
                          kappa_thr: float, z_thr: float) -> None:
    """Replay one battery cycle by cycle with the detector's verdict colouring each point and the event log below."""
    st.markdown(f"##### :material/play_circle: Live detection replay · {cell_id}")
    raw = CT_RAW[CT_RAW["Cell_ID"] == cell_id]
    c_bol_c = float(raw["C_bol_Ah"].iloc[0]) if len(raw) and "C_bol_Ah" in raw else np.nan
    soh_eol_c = te.soh_eol_for(c_bol_c, eol_ah) if np.isfinite(c_bol_c) else None
    removed = raw.loc[~raw["n"].isin(tr["n"]) & raw["SOH"].notna(), ["n", "SOH"]].sort_values("n")
    status = te.anomaly_cycle_status(tr, common, soh_eol=soh_eol_c)
    ev = te.anomaly_events(tr, common, kappa_thr, z_thr, soh_eol=soh_eol_c, eol_label=(f"{float(eol_ah):.2f} Ah" if float(eol_ah) > 1.0
                                                                                  else f"{100 * float(eol_ah):.0f} % of initial capacity"))
    n_eol = te.eol_cycle_from_replay(tr, soh_eol_c)
    if len(removed):
        st.caption(f"{len(removed)} of {len(raw)} cycles are shown as hollow circles: removed by the data cleaning "
                   "(the cycle where capacity recovers after a rest, plus the two transient cycles after it, and invalid "
                   "measurements), so they never trigger or hide an alarm.")
    ns = tr["n"].astype(int).tolist()
    key_n, key_p = f"an2_n_{cell_id}", "an2_play"
    st.session_state.setdefault(key_n, ns[min(3, len(ns) - 1)])
    st.session_state.setdefault(key_p, False)
    c = st.columns([1.0, 0.8, 0.8, 0.8, 1.3, 1.3])
    playing = bool(st.session_state[key_p])
    if c[0].button("Pause" if playing else "Play", key="an2_toggle", type="primary",
                   icon=":material/pause:" if playing else ":material/play_arrow:"):
        st.session_state[key_p] = not playing
        if not playing and st.session_state[key_n] >= ns[-1]:
            st.session_state[key_n] = ns[0]
    if c[1].button("Step", key="an2_step", icon=":material/skip_next:"):
        st.session_state[key_n] = min(ns[-1], st.session_state[key_n] + 1)
    if c[2].button("Back", key="an2_back", icon=":material/skip_previous:"):
        st.session_state[key_n] = max(ns[0], st.session_state[key_n] - 1)
    if c[3].button("Reset", key="an2_reset", icon=":material/replay:"):
        st.session_state[key_n], st.session_state[key_p] = ns[0], False
    speed = c[4].select_slider("Cycles per frame", [1, 2, 3, 5], value=2, key="an2_speed")
    interval = c[5].select_slider("Frame interval (s)", [0.4, 0.7, 1.0], value=0.7, key="an2_int")

    def frame() -> None:
        if st.session_state.get(key_p):
            nxt = st.session_state[key_n] + int(speed)
            if nxt >= ns[-1]:
                nxt, st.session_state[key_p] = ns[-1], False
            st.session_state[key_n] = nxt
        n = min(ns, key=lambda v: abs(v - int(st.session_state[key_n])))
        i = ns.index(n)
        verdict = status[i]
        col = DETECTION_COLORS.get(verdict, "#9AA3AD")
        k1, k2, k3 = st.columns([1, 1, 2])
        k1.metric("Cycle", f"{n} / {ns[-1]}")
        k2.metric("SOH now", f"{100 * float(tr['SOH'].iloc[i]):.1f} %")
        past = n_eol is not None and n >= n_eol
        k3.markdown(f"<div style='padding:10px 14px;border-radius:8px;background:{col}22;border-left:6px solid {col}'>"
                    f"<b>Detector now:</b> {verdict}"
                    + (f" · <b>past end of life</b> (since cycle {n_eol:.0f})" if past and verdict != "Past end of life" else "")
                    + "</div>", unsafe_allow_html=True)
        st.progress(min(1.0, (n - ns[0]) / max(ns[-1] - ns[0], 1)),
                    text=f"{'▶ streaming' if st.session_state.get(key_p) else '⏸ paused'} · nothing after the cursor is "
                         "visible to the detector")
        show(fig_detection_live(tr, status, n, P, cell_id, kappa_thr, z_thr, removed, soh_eol_c, n_eol), key="an2_live",
             export=False)
        seen = ev[ev["cycle"] <= n].sort_values("cycle", ascending=False)
        st.markdown(f"**What happened to {cell_id}** (up to cycle {n})")
        if seen.empty:
            st.caption("No event yet: the battery behaves as expected so far.")
        else:
            with st.container(height=min(300, 50 + 44 * len(seen)), border=True):
                st.markdown("  \n".join(
                    f"{EVENT_STYLE.get(r.kind, ('gray', ':material/info:'))[1]} :{EVENT_STYLE.get(r.kind, ('gray', ''))[0]}"
                    f"[**cycle {r.cycle:.0f} · {r.indicator}**] — {r[3]}" for r in seen.itertuples(index=False)))

    frag = getattr(st, "fragment", None)
    if frag is not None:
        try:
            frag(run_every=f"{interval}s" if st.session_state.get(key_p) else None)(frame)()
            return
        except TypeError:
            pass
    frame()


def anomaly_section() -> None:
    section("Mission 3 · Anomaly detection · Is a battery ageing abnormally?")
    st.markdown(
        "Each battery is replayed cycle by cycle, using only the past, with **two indicators**, each with its own "
        "reference. **FLEET**: does the battery age faster than the fleet's physics predicts for its conditions? Its "
        "ageing-speed factor κ is refitted every cycle; alarm when κ stays above the threshold (works from the first "
        "cycles: a battery abnormal from the start). **OWN**: does the battery age faster than in its own past? The "
        "robust (Theil–Sen) fade slope of the last 30 cycles is compared with the 30 before; alarm when it is "
        "significantly steeper (a knee). A 30-cycle window spans a full rest-and-recovery period of the NASA tests, so "
        "capacity regeneration shifts the level but hardly the slope. An alarm is a prompt to inspect, not a diagnosis.")
    c1, c2 = st.columns(2)
    kappa_thr = c1.select_slider("FLEET threshold (κ)", [1.5, 1.8, 2.2], value=1.8, key="an2_k",
                                 help="1.8: validated (fast batteries found at cycle 12, healthy ones 0.44-1.30).")
    z_thr = c2.select_slider("OWN threshold (z)", [2.5, 3.0, 4.0], value=3.0, key="an2_z",
                             help="3.0: validated (all knees found, no false alarm despite regeneration).")
    with st.expander("Check the detector on simulated batteries with known anomalies and regeneration (about 10 s)"):
        st.caption("11 batteries, all with NASA-like capacity regeneration every 18-25 cycles: three get a knee from "
                   "cycle 53 (R2, C2, H3), two age 2.5 times faster from the start (R3, C3), six are healthy with normal "
                   "manufacturing spread. FLEET should flag the fast ones, OWN the knees, nobody else. (The first "
                   "detector raised false alarms on all 8 non-knee batteries of this cohort.)")
        if st.button("Run the check", key="an2_val"):
            with st.spinner("Replaying the simulated batteries…"):
                st.session_state["an2_val_res"] = anomaly_validation2_cached(float(kappa_thr), float(z_thr))
        v = st.session_state.get("an2_val_res")
        if v is not None:
            ok = v["result"].astype(str)
            good = ok.str.startswith("OWN detected") | ok.str.startswith("FLEET detected") | (ok == "quiet (correct)")
            good &= ~ok.str.contains("false alarm", case=False)
            card("Check on simulated batteries", [f"{int(good.sum())} of {len(v)} batteries handled correctly.",
                                                   "Knees: " + ", ".join(v.loc[v["truth"] == "knee", "result"]) + ".",
                                                   "Fast from the start: " + ", ".join(v.loc[v["truth"] == "fast", "result"]) + "."])
            show_table(v.set_index("battery").style.format(precision=2, na_rep="—"))
    cells = tuple(sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(CT_RAW, g)}))
    st.caption(f"{len(cells)} NASA batteries of the constant-condition groups.")
    cfg = (DATA_KEY_RAW, cells, float(kappa_thr), float(z_thr))
    if st.button("Scan the NASA batteries", key="an2_go", type="primary", icon=":material/notifications_active:",
                 disabled=len(cells) < 3):
        try:
            with st.spinner("Replaying every battery cycle by cycle…"):
                scan = anomaly_scan2_cached(CT_RAW, imp, DATA_KEY_RAW, cells, float(kappa_thr), float(z_thr))
            st.session_state["an2"] = {"cfg": cfg, "scan": scan}
        except Exception as exc:
            report_error("Anomaly scan failed", exc, debug)
    saved = st.session_state.get("an2")
    if not saved:
        placeholder("Press 'Scan the NASA batteries': both indicators per battery and a replay of any battery appear here.")
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last scan; the results below use the previous settings.")
    scan = saved["scan"]
    groups = te.condition_groups(CT_RAW)
    summ = te.anomaly_summary2(scan)
    summ.insert(1, "group", summ["battery"].map(lambda c: str(groups.loc[c, "Group"]) if c in groups.index else ""))
    _cc = te.anomaly_common_cause(scan, {c: str(groups.loc[c, "Group"]) if c in groups.index else c for c in scan})
    summ["OWN alarm type"] = summ["battery"].map(lambda c: ("common event with " + ", ".join(sorted({p for v in _cc[c].values() for p in v})))
                                                 if any(_cc[c].values()) else ("individual" if _cc[c] else ""))
    ff = summ.loc[summ["first FLEET alarm (cycle)"].notna(), ["battery", "first FLEET alarm (cycle)"]]
    fo = summ.loc[summ["first OWN alarm (cycle)"].notna(), ["battery", "first OWN alarm (cycle)"]]
    card("Anomaly scan on NASA", [
        f"FLEET (ages faster than the fleet): {len(ff)} of {len(summ)}" + (": " + ", ".join(
            f"{b} (cycle {x:.0f})" for b, x in ff.itertuples(index=False)) if len(ff) else "."),
        f"OWN (ages faster than its own past): {len(fo)} of {len(summ)}" + (": " + ", ".join(
            f"{b} (cycle {x:.0f})" for b, x in fo.itertuples(index=False)) if len(fo) else "."),
        "Compare with the batteries the hybrid could not explain (B0005, B0006, B0018 late in life; B0041, B0045 "
        "early)."])
    show_table(summ.set_index("battery").style.format(precision=2, na_rep="—"))
    flagged = list(dict.fromkeys(ff["battery"].tolist() + fo["battery"].tolist()))
    pick = st.selectbox("Battery to replay", list(scan), key="an2_cell",
                        index=list(scan).index(flagged[0]) if flagged else 0)
    campaign = {c: str(groups.loc[c, "Group"]) if c in groups.index else c for c in scan}
    cc = te.anomaly_common_cause(scan, campaign)
    detection_live_replay(scan[pick], cc.get(pick), pick, float(kappa_thr), float(z_thr))
    download("Download the anomaly scan (CSV)",
             pd.concat([tr.assign(battery=c) for c, tr in scan.items()], ignore_index=True)
             .assign(**{"app version": te.ENGINE_VERSION, "kappa threshold": kappa_thr, "z threshold": z_thr})
             .to_csv(index=False).encode(), f"anomaly_scan_NASA__v{te.ENGINE_VERSION}.csv", "text/csv", key="an2_csv")









# =============================================================================
# View: integrated twin (Mission 3, option B: one twin, each task from the model proven best at it)
# =============================================================================
TWIN_STATUS_COLORS = {"Normal": "#009E73", "Watch": "#E69F00", "Alarm": "#CC0000", "End of life reached": "#6B1E1E"}
TWIN_HYB = dict(restarts=30, fit_with="Re", lam=0.3)        # the Models view's defaults: both views share one result
# The conservative end-of-life option (twin's own track record) was tested and not adopted (STATUS 5au): removed from
# the page (STATUS 5cj); Watch always uses the median forecast. The engine keeps the computation.
TWIN_PARTS = {"main": "Explanation (physics version 2)", "check": "Explanation check (physics version 3)",
              "scan": "Anomaly detection", "fleet": "Fleet-wide parts"}


def _twin_eol_relative() -> None:
    """Switch the control bar's end of life to 80 % of initial capacity (runs before the widgets are drawn)."""
    st.session_state["eol_kind"] = "% of initial capacity"     # its number input defaults to 80 %


def _twin_eol_label() -> str:
    return f"{float(eol_ah):.2f} Ah" if float(eol_ah) > 1.0 else f"{100 * float(eol_ah):.0f} % of initial capacity"


def twin_fleet_parts(build: bool) -> Optional[Dict[str, Any]]:
    """Fleet-wide parts of the integrated twin, built once per session and shared by every battery: the explanation
    (hybrid with physics version 2 = main split, version 3 = model-form check) and the anomaly scan. Both are saved
    in the result backup, so a warm start reloads them in seconds."""
    cells_an = tuple(sorted({c for g in te.NASA_PHYSICS_GROUPS for c in te.protocol_cells(CT_RAW, g)}))
    key = (DATA_KEY_RAW, cells_an)
    saved = st.session_state.get("itw_fleet")
    if saved and saved.get("key") == key:
        return saved
    if not build:
        return None
    out: Dict[str, Any] = {"key": key, "cells": cells_an, "errors": {}, "main": None, "check": None, "scan": None,
                           "common": {}}
    if len(cells_an) < 3:
        out["errors"]["fleet"] = (f"Explanation and anomaly detection need at least 3 batteries of the constant-condition "
                                  f"groups ({', '.join(te.NASA_PHYSICS_GROUPS)}); this dataset has {len(cells_an)}.")
        st.session_state["itw_fleet"] = out
        return out
    status_box = getattr(st, "status", None)
    box = status_box("Building the twin's fleet-wide parts (once per session)…", expanded=True) if status_box \
        else st.container()
    with box:
        for ver, name in ((2, "main"), (3, "check")):
            st.write(f"Explanation · hybrid with physics version {ver} ({len(cells_an)} batteries, "
                     f"{TWIN_HYB['restarts']} starts)…")
            try:
                out[name] = hybrid_cached(CT_RAW, imp, DATA_KEY_RAW, cells_an, TWIN_HYB["restarts"],
                                          TWIN_HYB["fit_with"], TWIN_HYB["lam"], ver)
            except Exception as exc:
                out["errors"][name] = str(exc)
        st.write("Anomaly detection · replaying every battery with the FLEET and OWN indicators…")
        try:
            scan = anomaly_scan2_cached(CT_RAW, imp, DATA_KEY_RAW, cells_an, 1.8, 3.0)
            groups = te.condition_groups(CT_RAW)
            campaign = {c: str(groups.loc[c, "Group"]) if c in groups.index else c for c in scan}
            out["scan"], out["common"] = scan, te.anomaly_common_cause(scan, campaign)
        except Exception as exc:
            out["errors"]["scan"] = str(exc)
        if status_box:
            try:
                box.update(label="Fleet-wide parts ready", state="complete", expanded=False)
            except Exception:
                pass
    st.session_state["itw_fleet"] = out
    return out


def twin_battery_parts(c: str) -> Dict[str, Any]:
    """This battery's forecast stream: the group's frozen model, starting at 30 % of the TRAINING batteries' median
    life (known at deployment; v5.36.0) and re-anchored every cycle to the measured capacity (the Mission 2 update-study choice; same settings as the Live twin's
    default, so both views share the saved result)."""
    grp_c, fz, train_c = frozen_context(c)
    out: Dict[str, Any] = {"group": grp_c, "fz": fz, "train": train_c, "frames": None, "track": None,
                           "why_not": frozen_unavailable_reason(c, grp_c, fz, train_c), "error": None}
    if out["why_not"]:
        return out
    try:
        with st.spinner(f"Streaming {c} through the twin core ({twin_core()}, {grp_c}; saved, so it runs only once)…"):
            ns, nm = te.frozen_live_window(ct, c, train_c)
            out["frames"], out["track"] = frozen_cached(ct, imp, DATA_KEY, c, tuple(train_c), fz[0], fz[1],
                                                        float(soh_eol), core_refit(), "capacity", float(eol_ah), ns, nm)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def fig_twin(n: int, meas: pd.DataFrame, removed: pd.DataFrame, future: Optional[pd.DataFrame],
             status: Optional[List[str]], an_n: Optional[np.ndarray], frame: Optional[Tuple[np.ndarray, ...]],
             snap: Dict[str, Any], traj: Optional[pd.DataFrame], P: Palette, c: str, soh_eol_c: float,
             x_max: float, cons_b: float = float("nan"), watch_q: Optional[float] = None) -> go.Figure:
    """Top: measured SOH up to today (coloured by the detector), today's forecast with its band and end-of-life
    estimate. Bottom: where the capacity went (explanation), up to today."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38],
                        subplot_titles=(f"{c} · state of health: measured up to today, forecast from today",
                                        "Where the capacity went (hybrid explanation, % of initial capacity)"))
    now = meas[meas["n"] <= n]
    if status is not None and an_n is not None and len(now):
        lab = dict(zip(an_n, status))
        st_now = [lab.get(float(v), "Normal") for v in now["n"]]
        for name, col in DETECTION_COLORS.items():
            sel = [s_ == name for s_ in st_now]
            if any(sel):
                sub = now[sel]
                fig.add_trace(go.Scatter(x=sub["n"], y=100 * sub["SOH"], mode="markers", name=name,
                                         marker=dict(color=col, size=8 if name != "Normal" else 6),
                                         hovertemplate="cycle %{x}: %{y:.1f} %<extra>" + name + "</extra>"), 1, 1)
    elif len(now):
        fig.add_trace(go.Scatter(x=now["n"], y=100 * now["SOH"], mode="markers", name="Measured SOH",
                                 marker=dict(color=P.measured, size=6)), 1, 1)
    rm = removed[removed["n"] <= n] if removed is not None and len(removed) else None
    if rm is not None and len(rm):
        fig.add_trace(go.Scatter(x=rm["n"], y=100 * rm["SOH"], mode="markers", name="Removed by data cleaning",
                                 marker=dict(color="rgba(0,0,0,0)", size=7, line=dict(color="#9AA3AD", width=1.2)),
                                 hovertemplate="cycle %{x}: %{y:.1f} % (recovery after a rest or invalid; not used)"
                                               "<extra></extra>"), 1, 1)
    if future is not None and len(future):
        fig.add_trace(go.Scatter(x=future["n"], y=100 * future["SOH"], mode="markers",
                                 name="Measured later (not seen by the twin)",
                                 marker=dict(color="rgba(0,0,0,0)", size=6, line=dict(color=P.muted, width=1))), 1, 1)
    if frame is not None:
        fut, med, lo, hi = frame
        keep = fut <= x_max
        fig.add_trace(go.Scatter(x=np.r_[fut[keep], fut[keep][::-1]], y=100 * np.r_[hi[keep], lo[keep][::-1]],
                                 fill="toself", fillcolor="rgba(213,94,0,0.15)", line=dict(width=0), mode="lines",
                                 name="Forecast band", hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=fut[keep], y=100 * med[keep], name="Forecast (frozen model, from today)", mode="lines",
                                 line=dict(color="#D55E00", width=2.4, dash="dash")), 1, 1)
        if np.isfinite(cons_b) and watch_q is not None:
            cons = med - cons_b / 100 * (fut - snap["forecast from"])
            fig.add_trace(go.Scatter(x=fut[keep], y=100 * cons[keep], mode="lines",
                                     name=f"Conservative {int(100 * watch_q)} % (own track record)",
                                     line=dict(color="#CC79A7", width=2, dash="dot")), 1, 1)
            ec = snap.get(f"EOL conservative q{int(100 * watch_q)}", float("nan"))
            if np.isfinite(ec) and not np.isfinite(snap["EOL reached at"]):
                fig.add_trace(go.Scatter(x=[ec], y=[100 * soh_eol_c], mode="markers", name="Conservative end of life",
                                         marker=dict(symbol="diamond-open", size=12, color="#CC79A7",
                                                     line=dict(width=2)),
                                         hovertemplate="conservative end of life: cycle %{x:.0f}<extra></extra>"),
                              1, 1)
        e, e_lo, e_hi = snap["EOL forecast"], snap["EOL early"], snap["EOL late"]
        if np.isfinite(e) and not np.isfinite(snap["EOL reached at"]):
            fig.add_trace(go.Scatter(
                x=[e], y=[100 * soh_eol_c], mode="markers", name="Forecast end of life",
                marker=dict(symbol="diamond", size=12, color="#D55E00", line=dict(color=P.plot_bg, width=1)),
                error_x=dict(type="data", symmetric=False, color="#D55E00", thickness=2.5, width=7,
                             array=[max((e_hi if np.isfinite(e_hi) else x_max) - e, 0)],
                             arrayminus=[max(e - (e_lo if np.isfinite(e_lo) else e), 0)]),
                hovertemplate="end of life forecast: cycle %{x:.0f}<extra></extra>"), 1, 1)
    if np.isfinite(soh_eol_c):
        fig.add_hline(y=100 * soh_eol_c, line=dict(color="#6B1E1E", dash="dash", width=1.2), row=1, col=1)
        fig.add_annotation(x=0, y=100 * soh_eol_c, xref="x", yref="y", text=f"end of life {100 * soh_eol_c:.1f} %",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(color="#B04040", size=11))
    y_ok = 100 * meas["SOH"].to_numpy(float)                 # axis from the valid cycles: invalid ones (0 %, 120 %)
    if len(y_ok):                                           # would squash the curve; they stay hollow, off-axis
        lo_y, hi_y = float(np.nanmin(y_ok)), float(np.nanmax(y_ok))
        if np.isfinite(soh_eol_c):
            lo_y = min(lo_y, 100 * soh_eol_c)
        if frame is not None:
            fut, med, lo, hi = frame
            k_ = fut <= x_max
            if k_.any():
                lo_y = min(lo_y, max(float(np.nanmin(100 * lo[k_])), lo_y - 15))
                hi_y = max(hi_y, min(float(np.nanmax(100 * hi[k_])), hi_y + 5))
        fig.update_yaxes(range=[lo_y - 2, hi_y + 2], row=1, col=1)
    if traj is not None and len(traj):
        tr = traj[traj["n"] <= n]
        for m in te.MECHANISMS:
            fig.add_trace(go.Scatter(x=tr["n"], y=100 * tr[m].clip(lower=0), name=m, stackgroup="mech", mode="lines",
                                     line=dict(width=0.6, color=MECH_COLORS[MECH_KEYS[m]])), 2, 1)
        fig.add_trace(go.Scatter(x=tr["n"], y=100 * tr[te.HYBRID_UNEXPLAINED].clip(lower=0),
                                 name="Not explained by the physics", stackgroup="mech", mode="lines",
                                 line=dict(width=0.6, color=HYB_UNEXPL_COLOR)), 2, 1)
        fig.add_trace(go.Scatter(x=tr["n"], y=100 * (1 - tr["SOH observed"]), name="Measured capacity loss",
                                 mode="markers", marker=dict(color=P.text, size=3), showlegend=False), 2, 1)
    for r in (1, 2):
        fig.add_vline(x=n, line=dict(color=P.text, width=1, dash="dot"), row=r, col=1)
    fig.add_annotation(x=n, y=0.98, xref="x", yref="y domain", text=" today", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(color=P.muted, size=11))
    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(title_text="SOH (%)", row=1, col=1)
    fig.update_yaxes(title_text="Lost (%)", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=2, col=1)
    return style_fig(fig, P, 720, f"Integrated twin · {c} · cycle {n}", hovermode="closest")


def fig_twin_whatif(d: pd.DataFrame, summ: Dict[str, Any], P: Palette, c: str, soh_eol_c: float) -> go.Figure:
    ends = [summ["cycle"]] + [v for v in (summ["physics EOL own"], summ["physics EOL changed"]) if np.isfinite(v)]
    d = d[d["n"] <= max(ends) * 1.12]                  # stop shortly after end of life: below it the physics means little
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["n"], y=100 * d["measured SOH"], mode="markers", name="Measured (own conditions)",
                             marker=dict(color=P.measured, size=4)))
    fig.add_trace(go.Scatter(x=d["n"], y=100 * d["measured, with the condition effect"], mode="lines",
                             name="Measured, with the physics' condition effect", line=dict(color="#0072B2", width=2)))
    fig.add_trace(go.Scatter(x=d["n"], y=100 * d["physics SOH, own conditions"], name="Physics, own conditions",
                             line=dict(color=P.text, width=1.4, dash="dash")))
    fig.add_trace(go.Scatter(x=d["n"], y=100 * d["physics SOH, changed conditions"], name="Physics, changed conditions",
                             line=dict(color="#0072B2", width=1.6, dash="dash")))
    fig.add_hline(y=100 * soh_eol_c, line=dict(color="#6B1E1E", dash="dash", width=1.1))
    fig.add_vline(x=summ["cycle"], line=dict(color=P.muted, width=1, dash="dot"))
    fig.update_xaxes(title_text="Discharge cycle n (this battery's charge throughput per cycle)")
    fig.update_yaxes(title_text="SOH (%)")
    return style_fig(fig, P, 460, f"What-if for {c}: the same battery under the changed conditions")


def twin_model_card(grp_c: Optional[str], fz: Optional[Tuple[str, str]], fleet: Optional[Dict[str, Any]]) -> None:
    if not fz:
        fc = "none for this group"
    elif twin_core() == TWIN_CORES[0]:
        fc = (f"{model_label(fz[0])} (frozen {grp_c} model; test RMSE "
              f"{te.FROZEN_SCORES.get(grp_c, float('nan')):.2f} SOH pts at 30 % of life)")
    else:
        fc = f"{model_label(fz[0])} ({twin_core()}, {grp_c}; compared in the twin scorecard below)"
    _rf_ = core_refit()
    fdesc = ("Starts at 30 % of the training batteries' median life (known at deployment), re-anchored every cycle to "
             "the measured capacity" + (f"; retrained every {_rf_} cycles (sidebar)." if _rf_ else
                                        " (Mission 2 update study: anchoring matters, retraining barely does)."))
    rows = [("Forecast SOH and end of life", fc, fdesc),
            ("Anomalies", "FLEET (ageing speed vs the fleet) + OWN (fade slope vs its own past), common-cause check, "
             "end-of-life event", "Validated 11/11 on simulated knees and regeneration; causal."),
            ("Explanation", "Hybrid: shared physics (frozen) + bounded network; physics version 2 = split, version 3 = "
             "model-form check", "Fitted on every battery's whole record: it explains, it does not predict."),
            ("What-if", "Shared-physics sensitivities", "Only conditions NASA varied, only at tested values; a "
             "sensitivity at its range limit is flagged as limited.")]
    show_table(pd.DataFrame(rows, columns=["Task", "Model", "How it is used"]).set_index("Task"))


def twin_whatif(fleet: Dict[str, Any], c: str, soh_eol_c: float) -> None:
    section("What-if · only the conditions NASA varied, only at the values NASA tested")
    h = fleet.get("main")
    if h is None or c not in h["data"]:
        placeholder(f"The what-if needs the hybrid explanation of {c}; it covers the constant-condition groups "
                    f"({', '.join(te.NASA_PHYSICS_GROUPS)}).")
        return
    groups = te.condition_groups(CT_RAW)
    ident = te.whatif_identifiability(h, groups)
    tested = te.whatif_tested(h, groups)
    own = tested["by_battery"].loc[c]
    st.markdown("The shared physics of the explanation (version 2) holds each mechanism's sensitivity to the operating "
                "conditions. Here the battery's record is replayed under other conditions. Only conditions that vary "
                "among the NASA batteries can be changed, and only to values NASA tested, so the twin never "
                "extrapolates outside its evidence.")
    show_table(ident.drop(columns=["key"]).set_index("condition"),
               note="limited: the fit put the sensitivity at the edge of its allowed range. At 0 this means the data "
                    "show no extra effect (e.g. discharge current), not that the effect is known to be zero. "
                    "Edge of range: the data could not "
                    "determine it; the effect shown is that edge, not a measured effect.")
    cols = st.columns(len(te.WHATIF_CONDITIONS))
    pick: Dict[str, Optional[float]] = {}
    status_of = dict(zip(ident["key"], ident["status"]))
    for col, (key, spec) in zip(cols, te.WHATIF_CONDITIONS.items()):
        vals = tested["values"][key]
        own_v = float(own[f"{key} tested"])
        ok = status_of.get(key) != "not identifiable" and len(vals) >= 2
        v = col.selectbox(spec["label"], vals, index=vals.index(own_v) if own_v in vals else 0, key=f"itw_wi_{key}",
                          disabled=not ok, format_func=lambda x, o=own_v: f"{x:g}" + ("  (own)" if x == o else ""),
                          help=None if ok else "Not identifiable from these data: kept at the battery's own value.")
        pick[key] = None if (not ok or v == own_v) else float(v)
    if not any(v is not None for v in pick.values()):
        placeholder("Choose a different tested value for at least one condition to see its effect on this battery.")
        return
    try:
        d, s = te.whatif_projection(h, c, soh_eol_c, groups=groups, **pick)
    except ValueError as exc:
        st.warning(str(exc))
        return
    k = st.columns(4)
    k[0].metric("Capacity lost by the last recorded cycle", f"{s['loss changed (%)']:.1f} %",
                delta=f"{s['loss changed (%)'] - s['loss own (%)']:+.1f} points vs own conditions", delta_color="inverse")
    k[1].metric("Loss ratio (changed / own)", fmt(s["loss ratio"], ".2f", "×"))
    k[2].metric("Physics end of life, changed", fmt(s["physics EOL changed"], ".0f", "cycles"),
                delta=(f"own {s['physics EOL own']:.0f}" if np.isfinite(s["physics EOL own"]) else "own: beyond range"),
                delta_color="off")
    k[3].metric("Life ratio (changed / own)", fmt(s["life ratio"], ".2f", "×"))
    notes = []
    if s["limited"]:
        notes.append("Limited condition(s) changed (" + ", ".join(te.WHATIF_CONDITIONS[k_]["label"] for k_ in s["limited"])
                     + "): the effect comes from a sensitivity at its range limit, not from evidence.")
    if not s["combination tested"]:
        notes.append("NASA never tested this combination of conditions together: each effect is from tested values, "
                     "their combination assumes the effects simply multiply.")
    notes.append("Read the difference, not the level: the physics alone is optimistic on NASA (it ages batteries too "
                 "slowly late in life), so its end-of-life cycles are later than the measured ones.")
    for t_ in notes:
        st.caption(t_)
    show(fig_twin_whatif(d, s, P, c, soh_eol_c), key="itw_wi_fig", data=d)


def view_twin() -> None:
    section(f"Integrated twin · {cell}")
    st.markdown("One view of the battery as the twin sees it **today**: its health and end-of-life forecast, whether "
                "it behaves abnormally, why it has lost capacity, and how other operating conditions would change "
                "that. Each task comes from the model that proved best at it (Mission 3, option B). Move *today* "
                "to replay the battery's life: the forecast and the detector see nothing after it.")
    _rf = core_refit()
    st.caption(f"Twin core model: **{twin_core()}**" + (f", retrained every {_rf} cycles" if _rf else "")
               + " (change it in the sidebar). The forecast, the watch status and the downloaded timeline all come "
                 "from this core.")
    grp_c, fz, _ = frozen_context(cell)
    fleet = twin_fleet_parts(build=False)
    with st.expander("Which model does what", icon=":material/account_tree:"):
        twin_model_card(grp_c, fz, fleet)
    if fleet is None:
        if st.button("Build the integrated twin", key="itw_build", type="primary", icon=":material/hub:",
                     help="Fits the explanation (two physics versions) and replays the anomaly detector for the whole "
                          "fleet once per session; saved in the result backup. Later batteries open in seconds."):
            fleet = twin_fleet_parts(build=True)
        else:
            placeholder("Press 'Build the integrated twin'. The first build fits the explanation for the whole fleet "
                        "(a few minutes on NASA, seconds with the result backup); every battery then opens quickly.")
            return
    for part, msg in (fleet.get("errors") or {}).items():
        st.warning(f"{TWIN_PARTS.get(part, part)} unavailable: {msg}")
    bp = twin_battery_parts(cell)
    if bp["why_not"]:
        st.info(f"Forecast unavailable: {bp['why_not']}", icon=":material/info:")
    if bp["error"]:
        st.warning(f"Forecast failed: {bp['error']}")

    raw = CT_RAW[CT_RAW["Cell_ID"] == cell].sort_values("n")
    c_bol_c = float(raw["C_bol_Ah"].iloc[0]) if len(raw) and "C_bol_Ah" in raw else float("nan")
    soh_eol_c = te.soh_eol_for(c_bol_c, eol_ah) if np.isfinite(c_bol_c) else float(soh_eol)
    if np.isfinite(soh_eol_c) and soh_eol_c >= 0.98:
        st.warning(f"The end of life ({_twin_eol_label()}) is at or above {cell}'s initial capacity "
                   f"({c_bol_c:.2f} Ah), so with this setting the battery would count as dead from its first cycles. "
                   "End of life is left undefined here (no threshold, no end-of-life forecast). Cold batteries hold "
                   "much less capacity: use a percentage of the initial capacity.", icon=":material/battery_alert:")
        st.button("Use 80 % of initial capacity", key="itw_eol_rel", on_click=_twin_eol_relative,
                  icon=":material/percent:")
        soh_eol_c = float("nan")
    meas = ct_cell.loc[~ct_cell["outlier"], ["n", "SOH"]].dropna().sort_values("n")
    if len(meas) < 6:
        st.info("This battery has too few valid cycles for the integrated twin.")
        return
    an = fleet["scan"].get(cell) if fleet.get("scan") else None
    common = (fleet.get("common") or {}).get(cell)
    used_n = set(an["n"].astype(float)) if an is not None else set(meas["n"].astype(float))
    removed = raw.loc[~raw["n"].astype(float).isin(used_n) & raw["SOH"].notna(), ["n", "SOH"]]
    t_main = te.hybrid_trajectory(fleet["main"], cell) if fleet.get("main") and cell in fleet["main"]["data"] else None
    t_check = te.hybrid_trajectory(fleet["check"], cell) if fleet.get("check") and cell in fleet["check"]["data"] else None
    frames = bp["frames"]
    s_cal = ((st.session_state.get("twin_scorecard") or {}).get("scales") or {}).get((twin_core(), cell))
    if frames and s_cal is not None and abs(s_cal - 1.0) > 1e-9 and st.toggle(
            f"Calibrated band (width × {s_cal:g}, learned on the other batteries of the group)", key="itw_cal",
            help="From the twin scorecard (4b): the band width that gives 80 % coverage on the OTHER batteries of "
                 "this group, applied to this battery. Changes the band, the early/late end-of-life and the status."):
        frames = {k_: (fr[0], np.asarray(fr[1], float),
                       np.asarray(fr[1], float) - s_cal * (np.asarray(fr[1], float) - np.asarray(fr[2], float)),
                       np.asarray(fr[1], float) + s_cal * (np.asarray(fr[3], float) - np.asarray(fr[1], float)))
                  + tuple(fr[4:]) for k_, fr in frames.items()}

    ns = meas["n"].astype(int).tolist()
    c1, c2, c3 = st.columns([3.0, 1.1, 1.1], vertical_alignment="bottom")
    n = int(c1.select_slider("Today (discharge cycle)", options=ns, value=ns[-1], key=f"itw_n_{cell}",
                             help="Everything up to this cycle is known to the twin; the forecast starts here."))
    reveal = c2.toggle("Show later measurements", value=False, key="itw_reveal",
                       help="Measurements after today, which the twin has not seen: to judge the forecast.")
    watch = int(c3.select_slider("Watch window (cycles)", [10, 20, 30, 50], value=20, key="itw_watch",
                                 help="Status 'Watch' when the early edge of the forecast band reaches end of life "
                                      "within this many cycles."))
    watch_q = None                                         # Watch uses the median forecast (5cj)
    snap = te.integrated_snapshot(n, soh_eol_c, meas, frames, an, common, t_main, t_check, watch, watch_q)

    k = st.columns(5)
    k[0].metric("Today", f"cycle {n}", delta=f"of {ns[-1]} recorded", delta_color="off")
    k[1].metric("Measured SOH", fmt(100 * snap["SOH measured"], ".1f", "%"))
    e, e_lo, e_hi = snap["EOL forecast"], snap["EOL early"], snap["EOL late"]
    if not np.isfinite(soh_eol_c):
        k[2].metric("End of life", "undefined", delta="see the warning above", delta_color="off")
        k[3].metric("Remaining", "—")
    elif np.isfinite(snap["EOL reached at"]):
        k[2].metric("End of life", f"reached at {snap['EOL reached at']:.0f}")
        k[3].metric("Remaining", "0 cycles")
    elif np.isfinite(e):
        band = (f"band {e_lo:.0f}–{e_hi:.0f}" if np.isfinite(e_lo) and np.isfinite(e_hi) else
                f"band from {e_lo:.0f}" if np.isfinite(e_lo) else None)
        if watch_q is not None:
            ec = snap.get(f"EOL conservative q{int(100 * watch_q)}", float("nan"))
            band = (f"conservative {int(100 * watch_q)} %: cycle {ec:.0f}" if np.isfinite(ec) and
                    np.isfinite(snap.get(f"optimism q{int(100 * watch_q)} (points/cycle)", float("nan")))
                    else "conservative: too little track record")
        k[2].metric("End of life (forecast)", f"cycle {e:.0f}", delta=band, delta_color="off")
        k[3].metric("Remaining (forecast)", f"{snap['remaining cycles']:.0f} cycles")
    else:
        hz = snap["forecast horizon"]
        k[2].metric("End of life (forecast)", "beyond horizon" if np.isfinite(hz) else "—",
                    delta=(f"after cycle {hz:.0f}" if np.isfinite(hz) else None), delta_color="off")
        k[3].metric("Remaining (forecast)", f"> {hz - n:.0f} cycles" if np.isfinite(hz) else "—")
    sh = snap["shares"]
    if sh:
        dom = snap["dominant"]
        k[4].metric("Main loss mechanism so far", MECH_KEYS.get(dom, dom) + f" {100 * sh[dom]:.0f} %",
                    delta=f"{100 * sh[te.HYBRID_UNEXPLAINED]:.0f} % not explained by the physics", delta_color="off")
    else:
        k[4].metric("Main loss mechanism so far", "—")
    col = TWIN_STATUS_COLORS.get(snap["status"], "#9AA3AD")
    st.markdown(f"<div style='padding:10px 14px;border-radius:8px;background:{col}22;border-left:6px solid {col}'>"
                f"<b>Status today: {html.escape(snap['status'])}</b> · {html.escape(snap['reason'])}"
                + (f" · detector: {html.escape(snap['detector'])}" if snap["detector"] else "") + "</div>",
                unsafe_allow_html=True)

    bw = snap.get("band half-width (points)", float("nan"))
    if np.isfinite(bw) and bw > 5:
        st.caption(f"This model's forecast band is wide (±{bw:.0f} SOH points already one cycle ahead), so the status "
                   "uses the median forecast; the band's early edge is reported, not acted on.")
    frame = None
    if frames and np.isfinite(snap["forecast from"]):
        frame = frames[int(snap["forecast from"])]
    x_last = float(raw["n"].max()) if len(raw) else float(ns[-1])
    x_max = x_last * 1.08
    if np.isfinite(snap["forecast horizon"]):              # at least ~35 % of the record ahead of today
        x_max = max(x_max, min(snap["forecast horizon"], n + max(15.0, 0.35 * x_last)))
    if np.isfinite(e_hi):
        x_max = max(x_max, min(e_hi * 1.05, x_last * 2.0))
    elif np.isfinite(e) and not np.isfinite(snap["EOL reached at"]):
        x_max = max(x_max, min(e * 1.1, x_last * 2.0))
    status_list = te.anomaly_cycle_status(an, common, soh_eol=soh_eol_c) if an is not None else None
    future = meas[meas["n"] > n] if reveal else None
    cons_b = (snap.get(f"optimism q{int(100 * watch_q)} (points/cycle)", float("nan")) if watch_q is not None
              else float("nan"))
    show(fig_twin(n, meas, removed, future, status_list, an["n"].to_numpy(float) if an is not None else None, frame,
                  snap, t_main, P, cell, soh_eol_c, x_max, cons_b, watch_q), key="itw_fig", export=False)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f"**What happened to {cell}** (up to cycle {n})")
        if an is None:
            st.caption("Anomaly detection covers the constant-condition groups (" + ", ".join(te.NASA_PHYSICS_GROUPS)
                       + ").")
        else:
            ev = te.anomaly_events(an, common, 1.8, 3.0, soh_eol=soh_eol_c, eol_label=_twin_eol_label())
            seen = ev[ev["cycle"] <= n].sort_values("cycle", ascending=False)
            if seen.empty:
                st.caption("No event yet: the battery behaves as expected so far.")
            else:
                with st.container(height=min(300, 50 + 44 * len(seen)), border=True):
                    st.markdown("  \n".join(
                        f"{EVENT_STYLE.get(r.kind, ('gray', ':material/info:'))[1]} "
                        f":{EVENT_STYLE.get(r.kind, ('gray', ''))[0]}[**cycle {r.cycle:.0f} · {r.indicator}**] — {r[3]}"
                        for r in seen.itertuples(index=False)))
    with right:
        st.markdown("**Why it lost capacity** (up to cycle {})".format(n))
        if not sh:
            st.caption("No explanation for this battery: the hybrid covers the constant-condition groups ("
                       + ", ".join(te.NASA_PHYSICS_GROUPS) + ").")
        else:
            sh2 = snap["shares check"]
            rows = [{"Part of the loss": MECH_KEYS.get(m, m) if m in MECH_KEYS else "not explained by the physics",
                     "Physics v2 (main)": 100 * sh[m], "Physics v3 (check)": 100 * sh2[m] if sh2 else np.nan}
                    for m in (*te.MECHANISMS, te.HYBRID_UNEXPLAINED)]
            show_table(pd.DataFrame(rows).set_index("Part of the loss").style.format("{:.0f} %", na_rep="—"))
            gap = snap["split gap (points)"]
            lines = []
            if np.isfinite(gap):
                lines.append(f"SEI and LAM shares differ by up to {gap:.0f} points between the two physics versions"
                             + (": the SEI vs LAM proportion is not determined by these data (model-form uncertainty)."
                                if gap > 10 else ": the split is robust to the physics version."))
            lines.append("Lithium plating appears only in the cold in both versions; there it dominates.")
            lines.append("The explanation is fitted on the whole record of every battery: it explains the past, it is "
                         "not part of the forecast.")
            for t_ in lines:
                st.caption(t_)

    tl = te.integrated_timeline(soh_eol_c, meas, frames, an, common, t_main, t_check, watch, watch_q)
    download("Download this battery's twin timeline (CSV)",
             tl.assign(battery=cell, group=grp_c, **{"app version": te.ENGINE_VERSION, "twin core": twin_core(),
                                                    "core retrain every": core_refit(),
                                                    "end of life": _twin_eol_label(),
                                                    "watch uses": "median forecast"}).to_csv(index=False).encode(),
             f"integrated_twin_{cell}__v{te.ENGINE_VERSION}.csv", "text/csv", key="itw_csv")
    twin_whatif(fleet, cell, soh_eol_c)
    ga_pinn_section(fleet)
    twin_scorecard_section(fleet)


# =============================================================================
# Mission 3 (rebuilt on the validated twin): operation (stage 2) and maintenance (stage 1), in Operations & control
# =============================================================================
M3_EOL_RULES = {"NASA convention: 1.4 Ah, but never above 80 % of initial capacity": "nasa",
                "Control-bar end of life": "bar"}
M3_GROUPS = ("Reference", "High current", "Hot", "Cold", "Pulsed load")   # constant conditions (Mixed excluded)


def m3_soh_eol_for(c: str, rule: str) -> float:
    """End of life of battery c as a fraction of its initial capacity. NASA convention: 1.4 Ah (30 % fade of the
    rated 2 Ah), but never above 80 % of the battery's initial capacity (cold batteries start at 0.9-1.7 Ah: 1.4 Ah
    would end their life within a few cycles; v5.34.0 bug: B0046-48 at 11-18 cycles). NaN when the control-bar
    threshold is undefined for this battery."""
    d = CT_RAW[CT_RAW["Cell_ID"] == c]
    c_bol = float(d["C_bol_Ah"].iloc[0]) if len(d) and "C_bol_Ah" in d else float("nan")
    if not np.isfinite(c_bol) or c_bol <= 0:
        return float("nan")
    if rule == "nasa":
        return float(min(te.DEFAULT_EOL_AH / c_bol, 0.80))
    s = te.soh_eol_for(c_bol, eol_ah)
    return float(s) if np.isfinite(s) and s < 0.98 else float("nan")


def m3_prices() -> Dict[str, Any]:
    with st.expander("Prices and assumptions (NASA has no prices: results are swept over them)", expanded=False,
                     icon=":material/payments:"):
        c1, c2, c3, c4, c5 = st.columns(5)
        p = dict(cost=float(c1.number_input("Battery cost (cycles of revenue)", 5.0, 500.0, 60.0, 5.0, key="m3_cost",
                                            help="Replacement cost expressed as beginning-of-life cycles of net "
                                                 "revenue of the battery's own operation.")),
                 buy=float(c2.number_input("Energy buy / sell price", 0.0, 0.95, 0.4, 0.05, key="m3_buy",
                                           help="Charging energy costs this fraction of the price the delivered "
                                                "energy sells for (J_op = revenue − energy cost).")),
                 k=float(c3.number_input("Unplanned replacement × cost", 1.0, 10.0, 3.0, 0.5, key="m3_k")),
                 lead=int(c4.number_input("Lead time (cycles)", 0, 60, 10, 1, key="m3_lead",
                                          help="Cycles between ordering a replacement and installing it.")),
                 rule=M3_EOL_RULES[c5.selectbox("End of life", list(M3_EOL_RULES), key="m3_eol")])
    return p


def fig_m3_options(r: pd.DataFrame, P: Palette, c: str) -> go.Figure:
    lab = [f"{i:g} A · {v:.1f} V" + (" (own)" if o else "") for i, v, o in zip(r["I_dis"], r["V_cut"], r["own"])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Profit rate (per hour)", "Cycles to end of life"))
    col = ["#0072B2" if o else "#9AA3AD" for o in r["own"]]
    fig.add_trace(go.Bar(x=lab, y=r["profit rate (per hour)"], marker_color=col, showlegend=False,
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=lab, y=r["cycles to EOL"], marker_color=col, showlegend=False,
                         hovertemplate="%{x}: %{y:.0f} cycles<extra></extra>"), 1, 2)
    return style_fig(fig, P, 420, f"Operation options for {c} (tested combinations at its temperature)")


@st.cache_data(show_spinner=False, max_entries=2)
def m3_depth_cached(_ct: pd.DataFrame, key: str, cells: Tuple[str, ...]) -> pd.DataFrame:
    """Within-battery depth table from the raw telemetry (saved in the result backup)."""
    return backed("m3depth", lambda: te.m3_depth_table(store, _ct, list(cells)), _ct, cells)


def m3_operation_section(fleet: Dict[str, Any], p: Dict[str, Any]) -> None:
    section("Mission 3 · 1. Operation: which current and cut-off?")
    st.markdown("Aggressive operation earns more per hour but ages the battery faster. For the selected battery, "
                "every discharge current and cut-off **NASA ran at its temperature** is compared. Energy and hours "
                "per cycle are **measured inside the batteries** from the raw discharge traces: each battery's Ah, Wh "
                "and hours down to every voltage it reached, normalised by its own Ah to 2.7 V, so cell-to-cell "
                "differences cannot pose as a depth effect. Ageing comes from the shared physics, scaled to this "
                "battery's measured fade. Cut-off = measured end-of-discharge voltage (about 0.1 V below NASA's "
                "nominal setting). Temperature is the battery's environment, not a control.")
    h = fleet.get("main")
    if h is None:
        st.warning("The shared physics is not available: " + (fleet.get("errors") or {}).get("main", "not built"))
        return
    ident = te.whatif_identifiability(h, te.condition_groups(CT_RAW))
    if store is None:
        depth = None
    else:
        with st.spinner("Measuring Ah, energy and time to each depth inside every battery (raw data, once)…"):
            try:
                depth = m3_depth_cached(CT_RAW, DATA_KEY_RAW, tuple(sorted(h["data"])))
            except Exception as exc:
                report_error("Depth measurement failed; economics fall back to between-battery values", exc, debug)
                depth = None
    with st.expander("What NASA lets the twin vary (identifiability)", icon=":material/rule:"):
        show_table(ident.drop(columns=["key"]).set_index("condition"),
                   note="limited = sensitivity at the edge of its allowed range; at 0 the data show no extra effect "
                        "(e.g. discharge current, which changes together with the group), so no claim is made.")
    eol_c = m3_soh_eol_for(cell, p["rule"])
    if cell not in h["data"]:
        placeholder(f"{cell} is not in the constant-condition groups ({', '.join(te.NASA_PHYSICS_GROUPS)}).")
    elif not np.isfinite(eol_c):
        st.info("End of life is undefined for this battery with the control-bar setting; choose the NASA convention "
                "in Prices and assumptions.")
    else:
        e = te.M3OperationEconomics(buy_price_ratio=p["buy"], battery_cost_cycles=p["cost"], soh_eol=eol_c)
        try:
            r = te.m3_operation_study(h, ct, cell, e, te.condition_groups(CT_RAW), depth)
        except Exception as exc:
            report_error("Operation study failed", exc, debug)
            r = pd.DataFrame()
        if len(r) < 2:
            st.info(f"NASA ran only one current and cut-off at {cell}'s temperature: nothing to choose here.")
        if len(r):
            best = r.iloc[0]
            k = st.columns(3)
            k[0].metric("Best option", f"{best['I_dis']:g} A · {best['V_cut']:.1f} V",
                        delta="own operation" if best["own"] else "differs from own", delta_color="off")
            k[1].metric("Profit rate vs own", f"{best['vs own (per hour)']:+.3f} per hour")
            k[2].metric("Fade scale k", f"{best['fade scale k']:.2f}",
                        help="Measured fade / physics fade for this battery (the physics alone is optimistic).")
            show(fig_m3_options(r, P, cell), key="m3_opt_fig", data=r)
            cols = ["I_dis", "V_cut", "own", "economics", "economics from", "Wh out per cycle", "hours per cycle",
                    "efficiency",
                    "cycles to EOL", "J_op per hour", "profit rate (per hour)", "vs own (per hour)", "flags"]
            show_table(r[cols].style.format({"Wh out per cycle": "{:.2f}", "hours per cycle": "{:.2f}",
                                             "efficiency": "{:.3f}", "J_op per hour": "{:.3f}",
                                             "profit rate (per hour)": "{:.3f}", "vs own (per hour)": "{:+.3f}"}))
            if (r["flags"] != "none").any():
                st.caption("Flagged options rest on a sensitivity at the edge of its range (see identifiability).")
    if st.button("Run for every battery and price setting", key="m3_ops_all", icon=":material/table_chart:",
                 help="All constant-condition batteries × battery cost 30/60/120 × buy price 0.2/0.4/0.6 × end of "
                      "life 70/80 % (NASA rule for the battery's own threshold is shown separately)."):
        rows, groups = [], te.condition_groups(CT_RAW)
        bar = st.progress(0.0, text="Simulating operation options…")
        cells = list(h["data"])
        for j, c in enumerate(cells):
            for cost, buy, eol in itertools.product((30, 60, 120), (0.2, 0.4, 0.6), (0.7, 0.8)):
                try:
                    r = te.m3_operation_study(h, ct, c, te.M3OperationEconomics(battery_cost_cycles=cost,
                                                                               buy_price_ratio=buy, soh_eol=eol), groups,
                                              depth)
                except Exception:
                    continue
                if len(r):
                    rows.append(r.assign(battery=c, group=str(groups.loc[c, "Group"]) if c in groups.index else "?",
                                         battery_cost_cycles=cost, buy_price_ratio=buy, soh_eol=eol))
            bar.progress((j + 1) / len(cells), text=f"{c} done")
        bar.empty()
        st.session_state["m3_ops"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    res = st.session_state.get("m3_ops")
    if res is not None and len(res):
        best = (res.sort_values("profit rate (per hour)", ascending=False)
                .groupby(["battery", "battery_cost_cycles", "buy_price_ratio", "soh_eol"]).head(1))
        tab = best.groupby(["group", "I_dis", "V_cut"]).size().rename("times best").reset_index()
        tot = best.groupby("group").size().rename("settings")
        tab = tab.merge(tot, on="group")
        tab["share"] = tab["times best"] / tab["settings"]
        st.markdown("**Which option is best, over all batteries and price settings**")
        show_table(tab.set_index(["group", "I_dis", "V_cut"]).style.format({"share": "{:.0%}"}))
        own_best = best.groupby("group")["own"].mean().rename("own operation best (share)")
        show_table(own_best.to_frame().style.format("{:.0%}"))
        download("Download the operation study (CSV)", res.drop(columns=[], errors="ignore").to_csv(index=False).encode(),
                 f"m3_operation_study__v{te.ENGINE_VERSION}.csv", "text/csv", key="m3_ops_csv")


def window_start_estimate(c: str, n_now: float) -> Optional[Tuple[float, str]]:
    """SOH at cycle n_now ESTIMATED from the battery's measurements, if the estimation was run for this battery on the
    Models page (best available: estimation GA-PINN, then estimation PINN, then GPR)."""
    one = st.session_state.get("est_one") or {}
    res = one.get("res") or {}
    if not res or one.get("key", (None, None, None))[2] != c:
        return None
    for m in ("GA-PINN-GPR", "GA-PINN-est", "PINN-est", "GPR"):
        r = res.get(m)
        if r is not None and len(r.get("n_all", [])):
            return float(np.interp(n_now, r["n_all"], r["pred_all"])), r["label"]
    return None


def m3_window_section(fleet: Dict[str, Any]) -> None:
    """Mission 3 · operating window: score a grid of temperature x current x cut-off for one battery from today."""
    section("Mission 3 · 1b. Operating window: the best conditions for one battery, from today")
    st.markdown(
        "Choose a battery and **today's cycle**. Its state today is either the **measured SOH** or the SOH **estimated "
        "from today's measurements** (run the estimation for this battery on the Models page first). Every candidate "
        "condition of the grid is then projected to end of life: ageing from the **shared physics** (identifiable "
        "mechanism laws), scaled to this battery's measured fade; charge and energy per cycle at each cut-off from the "
        "**measured discharge traces**. Today's damage is placed on each candidate's ageing curve and ageing continues "
        "from there. Candidates outside what NASA tested, or using a condition the data cannot identify, are flagged.")
    h = fleet.get("main")
    if h is None:
        placeholder("The operating window needs the shared physics (fleet-wide parts).")
        return
    groups = te.condition_groups(CT_RAW)
    tested = te.whatif_tested(h, groups)
    cells = sorted(h["data"])
    c1, c2, c3 = st.columns([1, 1.4, 1.6])
    c = c1.selectbox("Battery", cells, key="win_cell")
    d = CT_RAW[(CT_RAW["Cell_ID"] == c) & ~CT_RAW["outlier"]].sort_values("n")
    n_max = int(d["n"].max()) if len(d) else 10
    n_now = c2.slider("Today's cycle", 5, max(6, n_max), int(max(5, round(0.4 * n_max))), key="win_now")
    src = c3.radio("Today's state", ["Measured SOH (laboratory)", "Estimated from today's measurements"], key="win_src",
                   horizontal=True)
    meas = float(np.interp(n_now, d["n"], d["SOH"])) if len(d) else float("nan")
    soh_now, src_label = meas, "measured"
    if src.startswith("Estimated"):
        est = window_start_estimate(c, n_now)
        if est is None:
            st.info(f"No estimate for {c} yet: on the Models page, choose {c} in SOH estimation and press "
                    "**Estimate this battery**. Using the measured SOH meanwhile.")
        else:
            soh_now, src_label = est[0], est[1]
    st.caption(f"Today: cycle {n_now} · SOH {100 * soh_now:.1f} % ({src_label}); measured {100 * meas:.1f} %.")
    v_opts = [float(v) for v in te.M3_DEPTH_GRID if 1.95 <= v <= 2.75]
    g1, g2, g3 = st.columns(3)
    Ts = g1.multiselect("Temperatures (°C)", tested["values"]["T_C"], default=tested["values"]["T_C"], key="win_T")
    Is = g2.multiselect("Discharge currents (A)", tested["values"]["I_dis"], default=tested["values"]["I_dis"], key="win_I")
    Vs = g3.multiselect("Cut-off voltages (V)", v_opts, default=sorted(set(v_opts) & set(tested["values"]["V_cut"])) or [2.7],
                        key="win_V")
    k1, k2 = st.columns([1.2, 1])
    crit = k1.radio("Best means", ["Most energy delivered to end of life", "Longest life (cycles to end of life)"],
                    key="win_crit", horizontal=True)
    supported = k2.checkbox("Rank only candidates supported by the data", value=True, key="win_sup",
                            help="Excludes candidates with an untested value or a condition the data cannot identify.")
    if not (Ts and Is and Vs):
        placeholder("Choose at least one value of each condition.")
        return
    if st.button("Explore the operating window", key="win_go", type="primary", icon=":material/tune:"):
        try:
            depth = m3_depth_cached(CT_RAW, DATA_KEY, tuple(cells)) if store is not None else None
            W = te.m3_operating_window(h, c, n_now, soh_now, m3_soh_eol_for(c, "nasa"), Ts, Is, Vs, depth=depth,
                                       groups=groups)
            st.session_state["win_res"] = {"key": (DATA_KEY, c, n_now, round(soh_now, 5)), "W": W}
        except Exception as exc:
            report_error("operating window", exc, False)
    R = st.session_state.get("win_res")
    if not R or R["key"] != (DATA_KEY, c, n_now, round(soh_now, 5)):
        return
    W = R["W"]; T = W["table"].copy()
    col = "energy to EOL (Wh)" if crit.startswith("Most energy") else "remaining cycles"
    if col == "energy to EOL (Wh)" and T[col].isna().all():
        st.info("No measured energy for these cut-offs: ranking by life instead.")
        col = "remaining cycles"
    rank = T[~T["flags"].str.contains("not identifiable|value not tested", regex=True)] if supported else T
    rank = rank.sort_values(col, ascending=False)
    if not len(rank):
        placeholder("No candidate is supported by the data: untick the option above to see all candidates.")
        return
    best = rank.iloc[0]
    own = T[T["own"]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Best candidate", f"{best['T_C']:g} °C · {best['I_dis']:g} A · {best['V_cut']:g} V")
    if len(own):
        o = own.iloc[0]
        m2.metric("Remaining cycles", f"{int(best['remaining cycles'])}", f"{int(best['remaining cycles'] - o['remaining cycles']):+d} vs own")
        if np.isfinite(best["energy to EOL (Wh)"]) and np.isfinite(o["energy to EOL (Wh)"]):
            m3.metric("Energy to end of life", f"{best['energy to EOL (Wh)']:.0f} Wh",
                      f"{100 * (best['energy to EOL (Wh)'] / o['energy to EOL (Wh)'] - 1):+.0f} % vs own")
    top = rank.head(8)[["T_C", "I_dis", "V_cut", "own", "remaining cycles", "energy to EOL (Wh)", "Wh per cycle (new)", "flags"]]
    show_table(top.rename(columns={"T_C": "T (°C)", "I_dis": "I (A)", "V_cut": "cut-off (V)"}).reset_index(drop=True)
               .style.format({"energy to EOL (Wh)": "{:.0f}", "Wh per cycle (new)": "{:.2f}", "T (°C)": "{:g}",
                              "I (A)": "{:g}", "cut-off (V)": "{:g}"}, na_rep="—"),
               note=f"Ranked by {col}. Own operation = the battery's tested conditions. The physics is used for the "
                    "difference between conditions; flags mark extrapolation.")
    fig = go.Figure()
    cols_ = [P.accent, "#E69F00", "#56B4E9"]
    fig.add_trace(go.Scatter(x=d["n"], y=100 * d["SOH"], mode="markers", name=f"measured · {c}",
                             marker=dict(size=4, color="#9AA3AD")))
    shown = [tuple(r) for r in top[["T_C", "I_dis", "V_cut"]].head(3).to_numpy(float)]
    if len(own):
        shown = shown + [W["own"]] if W["own"] not in shown else shown
    for i, combo in enumerate(shown):
        n_, s_ = W["traj"][combo]
        is_own = combo == W["own"]
        fig.add_trace(go.Scatter(x=n_, y=100 * s_, mode="lines", name=("own: " if is_own else f"#{i + 1}: ") +
                                 f"{combo[0]:g} °C · {combo[1]:g} A · {combo[2]:g} V",
                                 line=dict(width=3 if i == 0 else 2, dash="dash" if is_own else "solid",
                                           color="#9AA3AD" if is_own else cols_[i % 3])))
    fig.add_hline(y=100 * m3_soh_eol_for(c, "nasa"), line_dash="dot", annotation_text="end of life")
    fig.add_vline(x=n_now, line_dash="dot", annotation_text="today")
    fig.update_xaxes(title_text="Cycle"); fig.update_yaxes(title_text="SOH (%)")
    show(style_fig(fig, P, 460, f"{c}: best candidates from today vs its own operation"), key="win_fig")
    piv_T = st.selectbox("Map at temperature (°C)", sorted(T["T_C"].unique()), key="win_mapT",
                         index=sorted(T["T_C"].unique()).index(best["T_C"]))
    M = T[np.isclose(T["T_C"], piv_T)].pivot_table(index="V_cut", columns="I_dis", values=col)
    if M.size:
        hm = go.Figure(go.Heatmap(z=M.to_numpy(), x=[f"{v:g} A" for v in M.columns], y=[f"{v:g} V" for v in M.index],
                                  colorscale="Viridis", colorbar=dict(title="Wh" if col.startswith("energy") else "cycles"),
                                  text=np.round(M.to_numpy(), 0), texttemplate="%{text}"))
        show(style_fig(hm, P, 360, f"{col} at {piv_T:g} °C: discharge current × cut-off"), key="win_map")
    download("Download all candidates (CSV)", T.to_csv(index=False).encode(),
             f"operating_window_{c}_cycle{n_now}__v{te.ENGINE_VERSION}.csv", "text/csv", key="win_csv")


def m3_lives(fleet: Dict[str, Any], rule: str, core: Optional[str] = None
             ) -> Tuple[Dict[str, Tuple[pd.DataFrame, str, float]], List[str]]:
    """Integrated timelines of every constant-condition battery that reached end of life (frozen forecast stream,
    detector), as the twin saw them cycle by cycle."""
    lives, notes = {}, []
    groups = te.condition_groups(CT_RAW)
    cands = [c for g in M3_GROUPS for c in te.protocol_cells(CT_RAW, g)]
    bar = st.progress(0.0, text="Replaying the twin on every battery…")
    for j, c in enumerate(cands):
        bar.progress((j + 1) / max(len(cands), 1), text=f"{c}…")
        eol_c = m3_soh_eol_for(c, rule)
        meas = ct[(ct["Cell_ID"] == c) & ~ct["outlier"]][["n", "SOH"]].dropna().sort_values("n")
        if not np.isfinite(eol_c) or len(meas) < 10:
            notes.append(f"{c}: end of life undefined or too few cycles")
            continue
        e_meas = te.eol_cycle_from_replay(meas.reset_index(drop=True), eol_c)
        if e_meas is None:
            notes.append(f"{c}: did not reach end of life in its record")
            continue
        grp_c, fz, train_c = frozen_context(c, core)
        why = frozen_unavailable_reason(c, grp_c, fz, train_c)
        if why:
            notes.append(f"{c}: {why}")
            continue
        try:
            ns, nm = te.frozen_live_window(ct, c, train_c)
            frames, _ = frozen_cached(ct, imp, DATA_KEY, c, tuple(train_c), fz[0], fz[1], float(eol_c),
                                      core_refit(core), "capacity",
                                      float(eol_ah), ns, nm)
        except Exception as exc:
            notes.append(f"{c}: forecast failed ({exc})")
            continue
        an = (fleet.get("scan") or {}).get(c)
        common = (fleet.get("common") or {}).get(c)
        tl = te.integrated_timeline(eol_c, meas, frames, an, common)
        lives[c] = (tl, str(groups.loc[c, "Group"]) if c in groups.index else "?", float(e_meas))
    bar.empty()
    return lives, notes


def _scorecard_battery(tl: pd.DataFrame, e: float, frames: Dict[int, Any]) -> Optional[Dict[str, float]]:
    """Criteria 1, 2, 4, 5 of the twin scorecard for one battery (STATUS 5bq): only cycles before its end of life."""
    d = tl[np.isfinite(tl["EOL forecast"]) & (tl["cycle"] < e)]
    if not len(d):
        return None
    err = d["EOL forecast"] - e
    start = float(d["cycle"].min())
    late = d[d["cycle"] >= start + (2 / 3) * (e - start)]
    a = d["EOL early"].fillna(-np.inf).to_numpy(float)
    b = d["EOL late"].fillna(np.inf).to_numpy(float)
    cov = float(((np.minimum(a, b) <= e) & (e <= np.maximum(a, b))).mean())
    viol, n_fr = 0, 0
    for n, fr in (frames or {}).items():
        if n >= e:
            continue
        med = np.asarray(fr[1], float)
        if not len(med):
            continue
        n_fr += 1
        if np.nanmax(med - np.minimum.accumulate(med)) > 0.005 or np.nanmax(med) > 1.02:
            viol += 1                                  # capacity growing back, or above a new battery
    return {"1 · end-of-life error (cycles)": float(err.abs().mean()),
            "2 · late-life bias (cycles, + = late)": float((late["EOL forecast"] - e).mean()) if len(late) else np.nan,
            "4 · band coverage": cov, "5 · impossible forecasts (share)": viol / max(n_fr, 1)}


def _group_share(per: pd.DataFrame, policy: str, cells: List[str]) -> float:
    base = per[per["policy"] == "run to end of life"].set_index("battery")["profit rate"].reindex(cells)
    best = per[per["policy"] == "perfect knowledge"].set_index("battery")["profit rate"].reindex(cells)
    pol = per[per["policy"] == policy].set_index("battery")["profit rate"].reindex(cells)
    room = float((best - base).sum())
    return float((pol - base).sum() / room) if room > 0 else np.nan


def _band_coverage(frames: Dict[int, Any], e: float, soh_eol: float, s: float = 1.0) -> float:
    """Share of forecast cycles (before end of life) whose end-of-life band contains the real end of life, with the
    band's half-widths scaled by s (calibration, STATUS 5bu)."""
    hits = []
    for n, fr in (frames or {}).items():
        if n >= e:
            continue
        fut, med, lo, hi = (np.asarray(x, float) for x in fr[:4])
        if not len(fut):
            continue
        lo_s, hi_s = med - s * (med - lo), med + s * (hi - med)
        below_lo, below_hi = np.nonzero(lo_s <= soh_eol)[0], np.nonzero(hi_s <= soh_eol)[0]
        early = fut[below_lo[0]] if len(below_lo) else np.inf
        late = fut[below_hi[0]] if len(below_hi) else np.inf
        hits.append(early <= e <= late)
    return float(np.mean(hits)) if hits else np.nan


BAND_SCALES = tuple(np.round(np.arange(0.1, 1.51, 0.05), 2))


GA_FILE = RESULTS_DIR / "ga_pinn_evaluations.csv"      # commit it to the repository to keep GA-PINN across restarts
# The GA-PINN search on NASA (8 batteries, 131 evaluations, STATUS 5bx), embedded so GA-PINN works with app.py alone;
# results/ga_pinn_evaluations.csv (a newer run or a restore) takes precedence when present.
GA_EMBEDDED_CSV = """battery,genome,fitness,RMSE_val (pts),late bias (pts),physics residual,seconds
B0005,h32|lr0.005|w1|p1|a0|it900,11.722848040552424,7.577409861442018,-8.211577693299308,0.3324082016542556,16.708733797073364
B0005,h48|lr0.005|w1|p1|a0|it600,10.597745605113333,8.268974998788647,-4.524485950853683,0.3804650545074836,29.763180017471313
B0005,h16|lr0.002|w0.3|p0|a1|it900,5.614709074621517,5.023449639428055,-1.1079086811300025,0.5748368317754599,13.72165822982788
B0005,h32|lr0.005|w3|p0|a1|it900,12.087473882801614,9.709958606903575,-4.682345318802924,0.12108927151882887,16.93030333518982
B0005,h32|lr0.01|w0.3|p0|a1|it600,4.933240328505106,4.443227521214045,0.9202890263832232,0.41572280740882156,11.159691333770752
B0005,h32|lr0.01|w1|p1|a1|it900,8.947586151517765,7.919844182056426,-1.9693671160219566,0.26141831859848663,19.00772738456726
B0005,h32|lr0.01|w0.3|p0|a1|it900,5.341997083661199,4.823580220097389,0.9478805962283966,0.44031126001067916,18.27687382698059
B0005,h48|lr0.01|w0.3|p0|a1|it600,5.5974809095582465,4.622221805719025,1.8116073615137032,0.4766045631893168,30.496106386184692
B0005,h32|lr0.01|w1|p0|a1|it600,4.978379739270875,4.826699701849384,0.2509665713144346,0.23145306877149488,10.837929248809814
B0005,h48|lr0.01|w0.3|p0|a1|it900,5.542596359747946,4.683241761322194,1.5124169093708764,0.46595281458177823,47.420072078704834
B0005,h16|lr0.01|w0.3|p0|a1|it900,6.246752650519332,5.589979879950106,-1.2366650720354457,0.4093543876764687,15.494611978530884
B0005,h32|lr0.002|w0.3|p0|a1|it600,6.19284395064283,5.224104987621368,1.8677854493785537,0.7385017679455782,11.41874074935913
B0005,h16|lr0.01|w0.3|p0.5|a1|it900,6.498227327113293,5.710829733220061,-1.5002133960449031,0.4163792648329166,14.86595892906189
B0005,h32|lr0.01|w3|p0|a1|it900,9.596949405328841,8.101222642021893,-2.9107486354864225,0.1041645248926456,19.100378036499023
B0005,h48|lr0.01|w0.3|p0|a0|it900,4.690730659254041,4.3602210007822615,0.4422139168821011,0.47523280443800936,50.479835987091064
B0006,h32|lr0.005|w1|p1|a0|it900,8.940190312140617,7.596096654135465,-2.6023534254337743,0.3623989515523213,18.100865125656128
B0006,h48|lr0.005|w1|p1|a0|it600,6.785017262103386,6.372489083611803,-0.6839704951157914,0.45525048488044956,31.19802165031433
B0006,h16|lr0.002|w0.3|p0|a1|it900,9.151616575176009,8.024700837364982,-2.1720986277900756,0.5878895829338715,15.412704467773438
B0006,h32|lr0.005|w3|p0|a1|it900,11.108927729394726,8.421467670532397,5.295579694551196,0.12087668163891976,18.59615993499756
B0006,h32|lr0.01|w0.3|p0|a1|it600,8.070816442517883,7.341194225495923,1.3929218723934462,0.537845048413513,11.907825946807861
B0006,h32|lr0.01|w1|p1|a1|it900,8.275296788144193,7.354526668942264,1.7569647585249555,0.36988276285498156,17.7267849445343
B0006,h32|lr0.01|w0.3|p0|a1|it900,9.106644942267593,7.590852551383426,2.9412123656769826,0.518002688787891,18.061290502548218
B0006,h48|lr0.01|w0.3|p0|a1|it600,7.086880120026765,6.959065678704588,-0.10737638000773554,0.6220171396719437,31.811739683151245
B0006,h32|lr0.01|w1|p0|a1|it600,10.040399661964665,7.496213490341784,5.0281542061926645,0.32134875156186427,12.02909779548645
B0006,h48|lr0.005|w1|p0|a0|it600,9.041896990523368,7.79740750747789,-2.350466743098054,0.3197329980650726,31.615903854370117
B0006,h48|lr0.005|w0.3|p1|a0|it900,9.782752057494958,7.414275388449097,-4.521165583905052,0.7783848521215053,47.6962947845459
B0006,h16|lr0.005|w0.3|p1|a1|it900,8.307863982363456,7.472827614552169,-1.5885635578129338,0.6536139237297967,14.916441440582275
B0006,h32|lr0.002|w1|p1|a0|it900,11.654504906474838,7.60347808224828,-8.018104423200736,0.3834301596037499,17.463375329971313
B0006,h48|lr0.005|w0.3|p0|a0|it900,8.042266609573987,7.379324632898832,-1.108347758692773,0.6912398475976111,48.6789186000824
B0006,h16|lr0.01|w0.3|p0.5|a1|it600,7.9092132050029065,7.759126967867615,-0.24203089571104547,0.4912542738779201,10.19660210609436
B0006,h48|lr0.005|w3|p1|a1|it600,8.328659721787577,7.37857401972973,-1.7573211543010268,0.18383754146058096,33.88025975227356
B0018,h32|lr0.005|w1|p1|a0|it900,10.177350257656661,7.310596790568365,-5.645217729703973,0.4016772798325043,18.406100273132324
B0018,h48|lr0.005|w1|p1|a0|it600,11.820561842697607,8.213652682555168,-7.021844515314098,0.5412931605608067,43.29631996154785
B0018,h16|lr0.002|w0.3|p0|a1|it900,8.766803643647508,5.244761317179779,-6.9259227942578505,0.6221279842569647,24.288336753845215
B0018,h32|lr0.005|w3|p0|a1|it900,11.090754741795335,8.742660579595888,-4.593575069660886,0.16180884848672378,24.024946689605713
B0018,h32|lr0.01|w0.3|p0|a1|it600,5.040295006275901,3.5446455265781016,-2.9143328911430304,0.556174334546624,14.44006085395813
B0018,h32|lr0.01|w1|p1|a1|it900,9.65943654406851,7.401463412563258,-4.423792704411091,0.3695149600523117,19.62422013282776
B0018,h32|lr0.01|w0.3|p0|a1|it900,4.989663606017818,3.868488546406413,-2.1467256345230368,0.519721435366708,19.362021446228027
B0018,h48|lr0.01|w0.3|p0|a1|it600,7.138339810129553,4.640606822200474,-4.834302223682143,0.5124074035168823,35.79921817779541
B0018,h32|lr0.01|w1|p0|a1|it600,5.2215246062492255,4.205915356545754,-1.9634324436843695,0.32310591748867235,13.906648635864258
B0018,h48|lr0.01|w0.3|p0|a1|it900,8.61251896777081,5.562564127467083,-5.86793420621889,0.5159913600026645,53.47645044326782
B0018,h16|lr0.01|w0.3|p0|a1|it900,5.391825180073119,4.294299377778291,-2.109943982435544,0.5240318440710615,16.702054500579834
B0018,h32|lr0.002|w0.3|p0|a1|it600,8.123383716699399,5.076749829533207,-6.002688791449174,0.9540027666291333,15.37161636352539
B0018,h48|lr0.01|w0.3|p0.5|a1|it600,8.21388768596344,5.232851113642026,-5.797510219476692,0.6085890237828774,35.97960376739502
B0018,h48|lr0.01|w3|p0|a1|it900,12.05305510685577,9.561799822657784,-4.759947603463322,0.131350365261534,54.3008131980896
B0018,h48|lr0.01|w0.3|p0|a0|it900,8.107407114922495,6.189116877946385,-3.601983594812853,0.5903302273937368,53.61203694343567
B0041,h32|lr0.005|w1|p1|a0|it900,10.668472204066495,6.383914943202658,-8.455569091768423,1.8965052542846428,16.839499473571777
B0041,h48|lr0.005|w1|p1|a0|it600,11.172811038698997,6.665035031763472,-8.845574654648209,1.9655993544829817,30.691527843475342
B0041,h16|lr0.002|w0.3|p0|a1|it900,10.664847821622587,6.393743504147113,-8.384202204470565,8.440205318742345,15.127071619033813
B0041,h32|lr0.005|w3|p0|a1|it900,12.573047385421058,7.478295970690891,-10.100865514730256,0.6166834826873193,16.943703651428223
B0041,h32|lr0.01|w0.3|p0|a1|it600,10.691627341055131,6.404525799673428,-8.44439487875136,5.978458219177174,11.35806393623352
B0041,h32|lr0.01|w1|p1|a1|it900,10.92834835824091,6.524880874100412,-8.694644262069632,1.7247873147180623,17.189361333847046
B0041,h32|lr0.01|w0.3|p0|a1|it900,10.83685610354671,6.489888138801724,-8.54054211960418,5.5185735586773506,17.994638204574585
B0041,h48|lr0.01|w0.3|p0|a1|it600,11.345930610209932,6.761638179843016,-8.957588598335366,5.7109481661540515,32.07946848869324
B0041,h32|lr0.005|w1|p1|a1|it900,11.265767778951206,6.743906865984608,-8.9287374193797,1.8913788785045027,17.21847653388977
B0041,h16|lr0.002|w0.3|p1|a1|it900,11.719648255025811,7.008275161516302,-9.265589960260147,8.0483603385954,15.374809503555298
B0041,h48|lr0.002|w1|p0|a1|it900,14.50901105535659,8.54907581429861,-11.671923970297412,3.051176752176439,46.79709815979004
B0041,h32|lr0.002|w0.3|p0|a1|it900,12.98109801170617,7.668436709963252,-10.455279668454342,8.911640696439047,17.60709500312805
B0041,h16|lr0.01|w0.3|p0|a1|it900,10.231097585039386,6.119457775553565,-8.071653234761202,5.735248426393524,17.197755575180054
B0041,h32|lr0.01|w0.3|p0.5|a1|it900,10.187973017519205,6.115689937643987,-8.010707013149725,3.9363009302472416,16.129745960235596
B0041,h16|lr0.002|w3|p0|a1|it900,14.034636856794059,8.314603378782023,-11.358814246439264,0.6666477364325936,14.76708197593689
B0041,h48|lr0.01|w0.3|p0|a0|it900,10.577196477613345,6.302045903194465,-8.293775208831722,6.264684300523455,42.60106635093689
B0041,h16|lr0.005|w3|p0|a1|it900,11.657939702958284,6.968981617727159,-9.302688611701166,0.5096918415297365,14.334681987762451
B0045,h32|lr0.005|w1|p1|a0|it900,10.972820720307483,6.544218215574333,-8.75096598114255,1.479657848347326,16.699963092803955
B0045,h48|lr0.005|w1|p1|a0|it600,12.022811463447114,6.94457801384004,-10.022341812345118,1.4557998432460806,23.776447534561157
B0045,h16|lr0.002|w0.3|p0|a1|it900,5.4008274426006535,4.249409973083624,-2.1553576764182276,6.36973835458765,15.183013916015625
B0045,h32|lr0.005|w3|p0|a1|it900,20.68884616436893,12.25500720731673,-16.78048448213161,0.6091901015989302,16.633174180984497
B0045,h32|lr0.01|w0.3|p0|a1|it600,11.657174628344148,7.083602637890496,-9.031591202124915,4.043029325921636,11.320907354354858
B0045,h32|lr0.01|w1|p1|a1|it900,11.711958077401093,6.803905736500783,-9.712829223474724,1.2765755944802444,16.886812686920166
B0045,h32|lr0.01|w0.3|p0|a1|it900,11.640587934970718,6.9732936958409795,-9.198275125573584,3.7429729186510095,17.177072048187256
B0045,h48|lr0.01|w0.3|p0|a1|it600,8.955864308178421,5.2839446151399345,-7.169862526137064,4.1547357924814285,25.68905472755432
B0045,h32|lr0.005|w1|p1|a1|it900,13.333593533186406,7.742244069347088,-11.071838551251924,1.7271490610456226,16.82237219810486
B0045,h16|lr0.002|w0.3|p1|a1|it900,8.53295549509085,5.138795253757518,-6.65174580882314,5.340847862023248,14.089985132217407
B0045,h48|lr0.002|w1|p0|a1|it900,24.323013368875184,14.440773932568824,-19.558309078771686,2.6592031716488913,37.4577853679657
B0045,h32|lr0.002|w0.3|p0|a1|it900,15.200529788269048,9.686401034093713,-10.875504446074585,5.765180059332021,17.431281566619873
B0045,h16|lr0.01|w0.3|p0|a1|it900,6.863554911903861,4.382343432265354,-4.8389428918721205,3.813454137466307,13.808595895767212
B0045,h32|lr0.01|w0.3|p0.5|a1|it900,9.41786179959121,6.170521963682999,-6.365055469116627,3.211775532451045,16.79442024230957
B0045,h16|lr0.002|w3|p0|a1|it900,17.26848459142444,9.741254126362291,-14.972822745816961,0.7881617262008289,14.099376201629639
B0045,h48|lr0.01|w0.3|p0|a0|it900,11.525538477368613,6.860166409375931,-9.117231816139116,4.163115125563769,35.55528473854065
B0045,h16|lr0.005|w3|p0|a1|it900,13.100128332706815,7.526047647879039,-11.071488000075776,0.5367078248455245,14.503559827804565
B0046,h32|lr0.005|w1|p1|a0|it900,18.36906314070887,10.6209358604872,-15.386106190249746,1.7900846170039637,16.396658182144165
B0046,h48|lr0.005|w1|p1|a0|it600,20.34418421679417,11.66429848375449,-17.218628783829296,1.9143399942875774,23.672165155410767
B0046,h16|lr0.002|w0.3|p0|a1|it900,14.05106062173835,8.213626929201515,-11.523466726074204,7.509900130609453,14.602053165435791
B0046,h32|lr0.005|w3|p0|a1|it900,31.54800445333817,18.18608798381518,-26.63696619504623,0.555659920713133,16.918819427490234
B0046,h32|lr0.01|w0.3|p0|a1|it600,21.251913195650307,12.882130762913022,-16.60395463110071,6.271502952375349,12.361953973770142
B0046,h32|lr0.01|w1|p1|a1|it900,23.233958935401123,13.580930646253153,-19.19226213274966,1.8646051423160748,17.021992206573486
B0046,h32|lr0.01|w0.3|p0|a1|it900,20.590727591280775,12.181274512611289,-16.670987822656137,5.841239871213685,16.101213216781616
B0046,h48|lr0.01|w0.3|p0|a1|it600,20.599991691934413,12.51549222610745,-15.9805105882577,5.87751395175576,26.186298847198486
B0046,h32|lr0.005|w1|p1|a1|it900,28.846618381500157,16.76082515446582,-24.055467305508706,1.996998190399729,17.112625122070312
B0046,h16|lr0.002|w0.3|p1|a1|it600,8.667554641795396,5.36356904486061,-6.465198301591779,9.665425690209869,9.993768215179443
B0046,h48|lr0.002|w1|p0|a1|it900,30.559024995516324,17.32899180097416,-26.24631185994585,3.3957646375461374,37.36277151107788
B0046,h16|lr0.002|w0.3|p0|a1|it600,6.908145045391021,4.609592197409119,-4.454438200937052,9.263952713904201,10.384007692337036
B0046,h16|lr0.01|w0.3|p0|a1|it900,17.492083230193938,10.374092041446017,-14.09632216431226,5.255044983178562,15.009292364120483
B0046,h32|lr0.01|w1|p0.5|a1|it600,21.528496677835665,12.88155425980769,-17.205126937959417,1.6981206193211307,11.412940502166748
B0046,h48|lr0.002|w3|p0|a1|it900,24.209205405517935,14.277116759779652,-19.683275862036385,0.7602746612013279,39.08584642410278
B0046,h48|lr0.002|w0.3|p0|a0|it600,15.9409855588984,10.080690272065917,-11.521740835686728,7.886230488265299,25.994495153427124
B0046,h16|lr0.005|w3|p1|a1|it600,20.006910007115916,11.96294153163599,-16.02908529082588,0.5144265223188841,10.206709861755371
B0047,h32|lr0.005|w1|p1|a0|it900,15.735692456352249,9.687090971490004,-11.996206047793914,1.3301002947636946,16.064865112304688
B0047,h48|lr0.005|w1|p1|a0|it600,17.62897465423313,10.664984351543051,-13.784783021838829,1.5618680672296912,25.585476875305176
B0047,h16|lr0.002|w0.3|p0|a1|it900,12.635671030793599,7.465756255310345,-10.193833542660052,6.573570202497173,14.516485691070557
B0047,h32|lr0.005|w3|p0|a1|it900,26.52007587309857,15.750138261488095,-21.45126946243603,0.5037387991376678,17.722130060195923
B0047,h32|lr0.01|w0.3|p0|a1|it600,19.148813097122066,11.801084685487842,-14.563402625842967,6.135208509347032,11.678382635116577
B0047,h32|lr0.01|w1|p1|a1|it900,22.12401926838951,13.691138828980435,-16.75564603011965,1.3657644230375745,18.17942476272583
B0047,h32|lr0.01|w0.3|p0|a1|it900,19.379361096096474,11.953219906240442,-14.687372853996076,5.1569453618493455,21.49324893951416
B0047,h48|lr0.01|w0.3|p0|a1|it600,21.06413585035393,12.833640000431181,-16.267444398295794,4.749943342371098,29.395236253738403
B0047,h32|lr0.005|w1|p1|a1|it900,25.940937974659754,15.709346802143177,-20.346758894722573,1.5511660561348877,18.937394380569458
B0047,h16|lr0.002|w0.3|p1|a1|it600,7.375637183063007,5.2400656219980295,-4.132061195751582,8.674792326552577,10.129440307617188
B0047,h48|lr0.002|w1|p0|a1|it900,28.509427054033853,16.309809503171284,-24.171368447534437,2.3443185666929676,43.85897254943848
B0047,h16|lr0.002|w0.3|p0|a1|it600,7.086069819723654,5.080909005239467,-3.8643128851918553,9.545731062868375,10.925268650054932
B0047,h16|lr0.01|w0.3|p0|a1|it900,17.243167965947233,10.54902322663561,-13.238919593008838,4.72049018108322,18.40664029121399
B0047,h32|lr0.01|w1|p0.5|a1|it600,20.461885675828178,12.668850854713574,-15.49869374996896,1.4519381020413253,12.10623550415039
B0047,h48|lr0.002|w3|p0|a1|it900,29.07733274241017,16.374251873594357,-25.220812089492213,0.6095733190212037,41.16964292526245
B0047,h48|lr0.002|w0.3|p0|a0|it600,18.26662679887854,10.873304148821939,-14.60308413901952,5.904603088283837,24.91182231903076
B0047,h16|lr0.005|w3|p1|a1|it600,22.606536371900198,13.255762073795688,-18.642392937288673,0.44213558121384844,10.813762426376343
B0048,h32|lr0.005|w1|p1|a0|it900,18.316615259181916,11.150627430716547,-14.22241265999928,1.7322463967528314,16.477752685546875
B0048,h48|lr0.005|w1|p1|a0|it600,20.03862330627407,11.980632746639548,-15.966477841197479,1.7017048375979982,26.584872007369995
B0048,h16|lr0.002|w0.3|p0|a1|it900,13.783179810085919,8.407682359744694,-10.598898582065123,7.4735818697312695,14.822403907775879
B0048,h32|lr0.005|w3|p0|a1|it900,29.74808944006127,17.7157351743333,-23.967008283892387,0.7268348804986621,18.493791580200195
B0048,h32|lr0.01|w0.3|p0|a1|it600,19.42848413257525,11.628763710689057,-15.466382528012826,5.842616066748874,12.384024620056152
B0048,h32|lr0.01|w1|p1|a1|it900,24.311331778297692,14.87226593679173,-18.766443492101377,1.7932380838676238,16.769348859786987
B0048,h32|lr0.01|w0.3|p0|a1|it900,22.140618134407106,13.336726896043032,-17.45759365207993,5.044376243309193,18.013418912887573
B0048,h48|lr0.01|w0.3|p0|a1|it600,23.26584034083255,13.676944035331035,-18.987222968486165,5.124182915410489,27.966206789016724
B0048,h32|lr0.005|w1|p1|a1|it900,26.720175777041447,16.33060516445517,-20.661103000098304,1.8629259231443644,18.089303493499756
B0048,h16|lr0.002|w0.3|p1|a1|it600,7.442790194026774,5.572819253777117,-3.593860585709116,9.901472135573384,10.583194971084595
B0048,h48|lr0.002|w1|p0|a1|it900,32.07236116293556,18.431897368252482,-27.05600526700087,2.9302418019426075,41.37009882926941
B0048,h16|lr0.002|w0.3|p0|a1|it600,7.348702526947225,5.345876990020527,-3.857301257220971,11.088460872270339,10.028178930282593
B0048,h16|lr0.01|w0.3|p0|a1|it900,19.31446068492612,11.796248207844402,-14.891865788738171,5.259979331295063,16.22546887397766
B0048,h32|lr0.01|w1|p0.5|a1|it600,21.218190270379,13.213568676370949,-15.922120767995315,1.4581423544306513,12.015429258346558
B0048,h48|lr0.002|w3|p0|a1|it900,31.766592547346463,18.23416636855194,-26.87495258460518,0.8186002525019395,40.98151206970215
B0048,h48|lr0.002|w0.3|p0|a0|it600,20.528920873953812,12.426820891090019,-16.012107158774803,6.603478911087416,25.99789309501648
B0048,h16|lr0.005|w3|p1|a1|it600,26.584335579144447,15.82662130968448,-21.453331779377503,0.6036144523154364,10.396690845489502"""


def ga_evaluations_table() -> pd.DataFrame:
    """All GA-PINN evaluations held in the session, in the format of the downloadable CSV (and of GA_FILE)."""
    rows = []
    for c, r in (st.session_state.get("ga_pinn") or {}).items():
        restored = r.get("restored")
        for k_, g in (r.get("genomes") or {}).items():
            d = (dict(restored[k_]) if restored is not None and k_ in restored
                 else {"fitness": float(r["fitness"][k_])})
            d.pop("battery", None)
            d.pop("genome", None)
            rows.append({"battery": c, "genome": te.ga_pinn_key(g).split("|", 1)[1], **d})
    return pd.DataFrame(rows)


def ga_save() -> None:
    """Write the GA-PINN evaluations next to the app (results/), so they load automatically after a restart."""
    try:
        E = ga_evaluations_table()
        if len(E):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            if GA_FILE.exists():                      # keep batteries of the file that are not in this session
                old = pd.read_csv(GA_FILE)
                old = old[~old["battery"].isin(E["battery"])]
                E = pd.concat([old, E], ignore_index=True)
            E.to_csv(GA_FILE, index=False)
    except Exception:
        pass                                          # read-only storage: the download / restore route still works


def ga_autoload() -> None:
    """Load the GA-PINN results from GA_FILE when the session has none (app start, restart, new version)."""
    if ga_loaded() or st.session_state.get("_ga_autoload_tried") == DATA_KEY:
        return
    st.session_state["_ga_autoload_tried"] = DATA_KEY
    try:
        src = GA_FILE if GA_FILE.exists() else io.StringIO(GA_EMBEDDED_CSV)
        got = ga_store_from_evaluations(pd.read_csv(src))
        cells = set(CT_RAW["Cell_ID"])
        store_ = st.session_state.setdefault("ga_pinn", {})
        for c, r in got.items():
            if c not in cells:
                continue
            grp_c, fz, train_c = frozen_context(c, TWIN_CORES[0])
            if not fz:
                continue
            ns, _ = te.frozen_live_window(CT_RAW, c, train_c)
            r["train"], r["n0"], r["autoloaded"] = tuple(train_c), int(ns), True
            store_[c] = r
    except Exception:
        pass


def ga_store_from_evaluations(E: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Rebuild finished GA-PINN searches from a downloaded evaluations CSV (battery, genome, fitness, ...): the winner and
    the ranked candidates need nothing else. Used after an app restart or an engine release, so the GA need not rerun."""
    names = list(te.GA_PINN_GENES)
    out: Dict[str, Dict[str, Any]] = {}
    for c, g in E.dropna(subset=["battery", "genome", "fitness"]).groupby("battery"):
        fit, genomes, details = {}, {}, {}
        for row in g.to_dict("records"):
            gen = te.ga_pinn_genome("PINN-ga|" + str(row["genome"]))
            gen["accel"] = int(gen["accel"])
            k = tuple(gen[n] for n in names)
            fit[k], genomes[k], details[k] = float(row["fitness"]), gen, row
        best = min(fit, key=fit.get)
        out[str(c)] = {"done": True, "generation": None, "history": [], "fitness": fit, "genomes": genomes,
                       "restored": details, "best_key": te.ga_pinn_key(genomes[best]), "best_fitness": fit[best]}
    return out


def ga_pinn_section(fleet: Optional[Dict[str, Any]]) -> None:
    """Professor's proposal (STATUS 5bu): a genetic algorithm searches the PINN configuration and the FORM of its
    physics for each battery, scored by nested leave-one-out on the training batteries only."""
    section("GA-PINN · genetic algorithm over the PINN and its physics")
    st.markdown("For each battery, a genetic algorithm searches six genes: **hidden size**, **learning rate**, "
                "**physics weight λ**, **SEI law exponent p** (1 = slows down as the layer grows, 0 = steady), an "
                "**acceleration term** (loss of active material speeds up with the total damage: the knee) and "
                "**training length**. Fitness = validation RMSE + 0.5 × |late-life bias| + small physics and time "
                "terms, computed by **nested leave-one-out**: only the training batteries' futures are scored, never "
                "the battery itself. The current twin-core settings are always in the first population. Evaluations "
                "are saved, so a stopped search continues where it stopped.")
    if not fleet or "scan" not in fleet:
        placeholder("Build the integrated twin first (button at the top of this page).")
        return
    rule = M3_EOL_RULES.get(st.session_state.get("m3_eol", list(M3_EOL_RULES)[0]), "nasa")
    lives, _ = m3_lives(fleet, rule, TWIN_CORES[0])
    cands = sorted(lives)
    if not cands:
        st.info("No battery reached end of life under the Mission 3 end-of-life rule.")
        return
    c1, c2, c3 = st.columns(3)
    pop = c1.number_input("Population", 4, 12, 6, key="ga_pop")
    gens = c2.number_input("Generations", 2, 8, 4, key="ga_gens")
    mins = c3.number_input("Time budget per press (minutes)", 2, 60, 10, key="ga_mins")
    sel = st.multiselect("Batteries", cands, default=cands, key="ga_cells")
    store_ = st.session_state.setdefault("ga_pinn", {})
    st.caption(("GA-PINN results load automatically from `results/" + GA_FILE.name + "`" if GA_FILE.exists() else
                "GA-PINN results load automatically from the copy embedded in the app (NASA search, 131 evaluations)")
               + f" · {sum(1 for r in store_.values() if r.get('done'))} batteries loaded.")
    with st.expander("Restore GA results from an evaluations CSV (if the automatic loading has no file)"):
        up = st.file_uploader("ga_pinn_evaluations__*.csv", type="csv", key="ga_restore")
        if up is not None and st.button("Restore", key="ga_restore_go", icon=":material/restore:"):
            try:
                E = pd.read_csv(up)
                got = ga_store_from_evaluations(E)
                for c, r in got.items():
                    if c not in cands:
                        continue
                    grp_c, fz, train_c = frozen_context(c, TWIN_CORES[0])
                    ns, _ = te.frozen_live_window(CT_RAW, c, train_c)
                    r["train"], r["n0"] = tuple(train_c), int(ns)
                    store_[c] = r
                ga_save()
                st.success(f"Restored {sum(c in cands for c in got)} batteries: GA-PINN is available without rerunning "
                           "the search, and from now on it loads automatically.")
            except Exception as exc:
                report_error("Could not restore the GA results", exc, debug)
    if st.button("Run / continue GA-PINN", key="ga_go", icon=":material/genetics:", type="primary"):
        deadline = time.time() + 60 * float(mins)
        msg = st.empty()
        for c in sel:
            prev = store_.get(c) or {}
            if prev.get("done") and (prev.get("restored") or prev.get("settings") == (int(pop), int(gens))):
                continue
            grp_c, fz, train_c = frozen_context(c, TWIN_CORES[0])
            ns, _ = te.frozen_live_window(CT_RAW, c, train_c)
            count = {"n": 0}

            def evaluate(g: Dict[str, Any], c=c, train_c=train_c, ns=ns) -> float:
                count["n"] += 1
                msg.caption(f"{c}: evaluation {count['n']} (genome {te.ga_pinn_key(g).split('|', 1)[1]})")
                return ga_fitness_cached(CT_RAW, imp, DATA_KEY, c, tuple(train_c), int(ns), float(eol_ah),
                                         tuple(sorted(g.items())))["fitness"]
            try:
                run = te.ga_pinn_run(evaluate, int(pop), int(gens), seed=0, deadline=deadline)
            except Exception as exc:
                report_error(f"GA-PINN failed for {c}", exc, debug)
                break
            run["settings"], run["train"], run["n0"] = (int(pop), int(gens)), tuple(train_c), int(ns)
            store_[c] = run
            if not run["done"]:
                st.info(f"Time budget reached during {c} (generation {run['generation'] + 1} of {int(gens)}). "
                        "Press again to continue: finished evaluations are reused.")
                break
        msg.empty()
        ga_save()
    rows, hist, evals = [], [], []
    start_key = tuple(sorted(te.GA_PINN_START.items()))
    for c in sel:
        r = store_.get(c)
        if not r:
            rows.append({"battery": c, "status": "not started"})
            continue
        if not r.get("fitness") or "train" not in r:
            rows.append({"battery": c, "status": "incomplete (press Run / continue)"})
            continue
        fit = r["fitness"]
        restored = r.get("restored")
        sk = tuple(te.GA_PINN_START[n] for n in te.GA_PINN_GENES)
        if restored is not None:
            start = float(fit.get(sk, np.nan))
        else:
            start = ga_fitness_cached(CT_RAW, imp, DATA_KEY, c, r["train"], r["n0"], float(eol_ah),
                                      start_key)["fitness"]
        best_k = min(fit, key=fit.get) if fit else None
        bg = r["genomes"][best_k] if best_k else {}
        rows.append({"battery": c, "status": ("restored" if restored is not None else "done") if r["done"]
                     else f"generation {r['generation'] + 1}",
                     "evaluations": len(fit), "fitness, current settings": start,
                     "fitness, best": fit[best_k] if best_k else np.nan,
                     "improvement (%)": 100 * (1 - fit[best_k] / start) if best_k and start > 0 else np.nan,
                     "hidden": bg.get("hidden"), "learning rate": bg.get("learning_rate"),
                     "λ physics": bg.get("physics_weight"), "SEI p": bg.get("sei_p"),
                     "acceleration": ("on" if bg.get("accel") else "off") if bg else None,
                     "iterations": bg.get("iterations")})
        for h in r["history"]:
            hist.append({"battery": c, **h})
        for k_, g in r["genomes"].items():
            if restored is not None:
                d = {kk: vv for kk, vv in restored[k_].items() if kk not in ("battery", "genome")}
            else:
                d = ga_fitness_cached(CT_RAW, imp, DATA_KEY, c, r["train"], r["n0"], float(eol_ah),
                                      tuple(sorted(g.items())))
            evals.append({"battery": c, "genome": te.ga_pinn_key(g).split("|", 1)[1], **d})
    R = pd.DataFrame(rows).set_index("battery")
    show_table(R.style.format("{:.3g}", subset=[c for c in R.columns if R[c].dtype.kind == "f"], na_rep="—"),
               note="Lower fitness is better. A finished search becomes the GA-PINN core of that battery (sidebar: "
                    "Twin core model → GA-PINN core); unfinished batteries use the default PINN core.")
    if hist:
        H = pd.DataFrame(hist)
        fig = go.Figure()
        for c, dh in H.groupby("battery"):
            fig.add_trace(go.Scatter(x=dh["generation"] + 1, y=dh["best fitness"], mode="lines+markers", name=c))
        fig.update_xaxes(title_text="Generation", dtick=1)
        fig.update_yaxes(title_text="Best fitness (lower is better)")
        show(style_fig(fig, P, 360, "GA-PINN: best fitness per generation"), key="ga_fig", data=H)
    if evals:
        download("Download all GA-PINN evaluations (CSV)", pd.DataFrame(evals).to_csv(index=False).encode(),
                 f"ga_pinn_evaluations__v{te.ENGINE_VERSION}.csv", "text/csv", key="ga_csv")


def _score_cells(rule: str, core: str) -> List[str]:
    """Batteries the scorecard streams for a core: Mission 3 groups, end of life reached in the record, a forecaster
    available (same filters as m3_lives, without streaming)."""
    out = []
    for c in [c for g in M3_GROUPS for c in te.protocol_cells(CT_RAW, g)]:
        eol_c = m3_soh_eol_for(c, rule)
        meas = CT_RAW[(CT_RAW["Cell_ID"] == c) & ~CT_RAW["outlier"]][["n", "SOH"]].dropna().sort_values("n")
        if not np.isfinite(eol_c) or len(meas) < 10:
            continue
        if te.eol_cycle_from_replay(meas.reset_index(drop=True), eol_c) is None:
            continue
        grp_c, fz, train_c = frozen_context(c, core)
        if frozen_unavailable_reason(c, grp_c, fz, train_c):
            continue
        out.append(c)
    return out


def twin_scorecard_section(fleet: Optional[Dict[str, Any]]) -> None:
    """Which forecaster makes the better digital twin? Seven criteria agreed before running (STATUS 5bq), computed
    for every twin core on every battery that reached end of life (leave-one-out choices); 4b = band coverage after
    leave-one-out calibration of the band width (5bu)."""
    section("Twin scorecard · frozen data model vs PINN twin core vs GA-PINN core")
    st.markdown("Seven criteria fixed **before** running: (1) end-of-life forecast error over the battery's life, "
                "(2) late-life bias, (3) decision value in the replacement study, (4) honest uncertainty (band "
                "coverage; 4b after calibration), (5) physical consistency, (6) one model for forecast, explanation "
                "and what-if, (7) speed. Same batteries, same live window, same re-anchoring for every core. "
                "Calibration (4b): the band width of each battery is scaled by a factor learned on the **other** "
                "batteries of its group so that their coverage is 80 %.")
    if core_refit(TWIN_CORES[1]):
        st.caption(f"PINN cores retrained every {core_refit(TWIN_CORES[1])} cycles (sidebar); the frozen model is "
                   "trained once, as validated in Mission 2.")
    if not fleet or "scan" not in fleet:
        placeholder("Build the integrated twin first (button at the top of this page).")
        return
    rule = M3_EOL_RULES.get(st.session_state.get("m3_eol", list(M3_EOL_RULES)[0]), "nasa")
    ga_sig = tuple(sorted((c, r.get("best_key")) for c, r in (st.session_state.get("ga_pinn") or {}).items()
                          if r.get("done")))
    skey = (DATA_KEY, rule, float(eol_ah), core_refit(TWIN_CORES[1]), ga_sig)
    saved = st.session_state.get("twin_scorecard")
    cores_sel = st.multiselect("Cores to score", list(TWIN_CORES), default=list(TWIN_CORES), key="score_cores",
                               help="Score fewer cores per press to split the work (e.g. frozen + PINN first).")
    budget = st.number_input("Time budget per press (minutes)", 2, 60, 15, key="score_mins")
    skey = skey + (tuple(cores_sel),)
    prog = st.session_state.setdefault("score_progress", {})       # (data, rule, refit, core, battery) -> seconds
    pkey = (DATA_KEY, rule, float(eol_ah), core_refit(TWIN_CORES[1]))
    todo = [(core, c) for core in cores_sel for c in _score_cells(rule, core)]
    if st.button("Score the twin cores", key="score_go", icon=":material/scoreboard:", type="primary",
                 help="Streams every battery that reached end of life through the selected cores, one at a time, "
                      "within the time budget; the table appears when all are streamed."):
        deadline = time.time() + 60 * float(budget)
        msg = st.empty()
        for i, (core, c) in enumerate(todo):
            k_ = pkey + (core, c)
            if k_ in prog:
                continue
            if time.time() > deadline:
                break
            msg.caption(f"Streaming {c} through {core} ({i + 1} of {len(todo)})…")
            grp_c, fz, train_c = frozen_context(c, core)
            ns, nm = te.frozen_live_window(CT_RAW, c, train_c)
            t0 = time.time()
            try:
                frozen_cached(CT_RAW, imp, DATA_KEY, c, tuple(train_c), fz[0], fz[1], float(m3_soh_eol_for(c, rule)),
                              core_refit(core), "capacity", float(eol_ah), ns, nm)
            except Exception as exc:
                st.warning(f"{c} · {core}: forecast failed ({exc})")
            prog[k_] = time.time() - t0
        msg.empty()
        n_done = sum((pkey + (core, c)) in prog for core, c in todo)
        if n_done < len(todo):
            st.info(f"{n_done} of {len(todo)} streamed before the time budget. Press again to continue: finished "
                    "batteries are saved.")
        else:
            res, scales = {}, {}
            for core in cores_sel:
                lives, notes = m3_lives(fleet, rule, core)
                rows, kept = [], []
                for c, (tl, grp, e) in lives.items():
                    grp_c, fz, train_c = frozen_context(c, core)
                    ns, nm = te.frozen_live_window(CT_RAW, c, train_c)
                    eol_c = float(m3_soh_eol_for(c, rule))
                    frames, _ = frozen_cached(CT_RAW, imp, DATA_KEY, c, tuple(train_c), fz[0], fz[1], eol_c,
                                              core_refit(core), "capacity", float(eol_ah), ns, nm)
                    sc = _scorecard_battery(tl, e, frames)
                    if sc:
                        rows.append({"core": core, "group": grp, "battery": c, "model key": fz[0], **sc})
                        kept.append((c, grp, e, eol_c, frames))
                for c, grp, e, eol_c, frames in kept:     # 4b: band scale learned on the OTHER batteries of the group
                    others = [x for x in kept if x[1] == grp and x[0] != c]
                    if others:
                        covs = [np.nanmean([_band_coverage(o[4], o[2], o[3], s) for o in others]) for s in BAND_SCALES]
                        s_c = float(BAND_SCALES[int(np.nanargmin(np.abs(np.array(covs) - 0.8)))])
                    else:
                        s_c = 1.0
                    scales[(core, c)] = s_c
                    for r_ in rows:
                        if r_["battery"] == c:
                            r_["4b · band coverage, calibrated"] = _band_coverage(frames, e, eol_c, s_c)
                            r_["band scale (calibrated)"] = s_c
                per = te.m3_maintenance_study(lives, te.M3Economics())[0] if len(lives) >= 3 else pd.DataFrame()
                tm = [v for (d_, r__, e__, f__, co, cc), v in prog.items()
                      if (d_, r__, e__, f__) == pkey and co == core]
                res[core] = {"rows": pd.DataFrame(rows), "per": per,
                             "secs": float(np.median(tm)) if tm else np.nan, "notes": notes}
            saved = {"key": skey, "res": res, "scales": scales}
            st.session_state["twin_scorecard"] = saved
    n_done = sum((pkey + (core, c)) in prog for core, c in todo)
    if todo:
        st.caption(f"Streamed: {n_done} of {len(todo)} battery × core pairs (each one saved; a press stops at the time "
                   "budget and the next press continues).")
    if not saved or saved.get("key") != skey:
        placeholder("Press the button to score the cores (cores already streamed reuse their saved results).")
        return
    res = saved["res"]
    allrows = pd.concat([r["rows"] for r in res.values()], ignore_index=True)
    if allrows.empty:
        st.info("No battery reached end of life with a core available.")
        return
    crit = [c for c in allrows.columns if c[:1].isdigit()]
    cores = [c for c in TWIN_CORES if c in res]
    out = []
    for g, dg in list(allrows.groupby("group")) + [("all", allrows)]:
        for cname in crit + ["3 · decision value: twin + age limit (share of gain)",
                             "3 · decision value: twin alone (share of gain)"]:
            row: Dict[str, Any] = {"group": g, "criterion": cname}
            for core in cores:
                if cname.startswith("3 ·"):
                    per = res[core]["per"]
                    cells = list(dg[dg["core"] == core]["battery"])
                    pol = "twin + age limit" if "age limit" in cname else "twin-informed"
                    row[core] = _group_share(per, pol, cells) if len(per) and cells else np.nan
                elif cname.startswith("2 ·"):
                    row[core] = float(dg[dg["core"] == core][cname].abs().mean())   # closer to 0 is better
                else:
                    row[core] = float(dg[dg["core"] == core][cname].mean())
            vals = {k: row[k] for k in cores if np.isfinite(row[k])}
            if len(vals) >= 2:
                if cname.startswith("4"):              # coverage: closest to the nominal 0.8 wins
                    score = {k: abs(v - 0.8) for k, v in vals.items()}
                elif cname.startswith("3 ·"):
                    score = {k: -v for k, v in vals.items()}
                else:                                  # 1, 2, 5: lower is better
                    score = dict(vals)
                order = sorted(score, key=score.get)
                tol = 0.01 if cname[:1] in ("3", "4", "5") else 0.25
                tied = [k for k in order if abs(score[k] - score[order[0]]) < tol]
                row["better"] = ("tie" if len(tied) == len(order) else
                                 order[0] if len(tied) == 1 else " = ".join(tied))   # e.g. both PINN cores beat frozen
            out.append(row)
    S = pd.DataFrame(out)
    S.loc[S["criterion"].str.startswith("2 ·"), "criterion"] = "2 · late-life bias, size (cycles; |bias|)"
    show_table(S.set_index(["group", "criterion"]).style.format("{:.2f}", subset=cores, na_rep="—"),
               note="Ties: within 0.25 cycles (1, 2) or 1 percentage point (3, 4, 5). Lower is better for 1, 2 (size "
                    "of the bias) and 5; higher for 3; for 4 and 4b the closest to the band's nominal 80 % wins. "
                    "Decision value uses the default economics; replacement margin and age chosen leave-one-out. "
                    "GA-PINN and self-tuning cores: batteries whose search has not finished use the default PINN "
                    "core (column "
                    "'model key' in the per-battery table).")
    t7 = {core: res[core]["secs"] for core in cores}
    st.markdown(f"**6 · One model for forecast, explanation and what-if:** {TWIN_CORES[0]}: no (forecast only; "
                f"explanation and what-if come from the separate shared physics) · the PINN cores "
                f"({', '.join(TWIN_CORES[1:])}): yes (mechanism states explain the fade, condition inputs give the "
                f"what-if).  \n"
                f"**7 · Speed:** every cycle is a re-anchoring for every core (instant); time to stream one battery "
                f"(trainings and retrainings included, median, measured when first computed): "
                + " · ".join((f"{k}: {v:.0f} s" if v >= 1 else f"{k}: < 1 s") if np.isfinite(v) else f"{k}: —"
                             for k, v in t7.items()))
    with st.expander("Per battery"):
        show_table(allrows.set_index(["core", "group", "battery"]).style.format(
            "{:.2f}", subset=[c for c in allrows.columns if allrows[c].dtype.kind == "f"]))
    download("Download the scorecard (CSV)", S.to_csv(index=False).encode(),
             f"twin_scorecard__v{te.ENGINE_VERSION}.csv", "text/csv", key="score_csv")
    download("Download the per-battery scores (CSV)", allrows.to_csv(index=False).encode(),
             f"twin_scorecard_batteries__v{te.ENGINE_VERSION}.csv", "text/csv", key="score_csv2")


def m3_policy_summary(per: pd.DataFrame) -> pd.DataFrame:
    """Per-policy summary of a replacement study (same definitions as te.m3_maintenance_study), recomputed here so
    the alarm-aware variants sit in the same table."""
    base = per[per["policy"] == "run to end of life"].set_index("battery")["profit rate"]
    best = per[per["policy"] == "perfect knowledge"].set_index("battery")["profit rate"]
    rows = []
    for name, g in per.groupby("policy"):
        g = g.set_index("battery")
        room = (best.reindex(g.index) - base.reindex(g.index)).sum()
        rows.append({"policy": name, "batteries": len(g), "mean profit rate": g["profit rate"].mean(),
                     "unplanned": int(g["unplanned"].sum()),
                     "mean unused life (cycles)": float((g["end of life"] - g["replace at"]).clip(lower=0).mean()),
                     "share of perfect-knowledge gain": ((g["profit rate"] - base.reindex(g.index)).sum() / room)
                     if room > 0 else np.nan})
    return pd.DataFrame(rows).set_index("policy").sort_values("mean profit rate", ascending=False)


def m3_maintenance_section(fleet: Dict[str, Any], p: Dict[str, Any]) -> None:
    section("Mission 3 · 2. Maintenance: when to replace?")
    st.markdown("Each battery's **real measured life** is replayed cycle by cycle, and replacement is decided with "
                "only what the twin knew at that cycle. Policies: **perfect knowledge** (best cycle in hindsight, the "
                "upper bound), **twin-informed** (order when the forecast end of life is within the lead time plus a "
                "safety margin), **twin + age limit**, **twin + alarm** and **twin + alarm + age limit** (an anomaly "
                "alarm, FLEET or OWN, also triggers the order), **best fixed age** (the usual practice) and **run to end of "
                "life** (always unplanned). Margin and age are chosen on the *other* batteries (leave-one-out). "
                "Revenue per cycle is proportional to SOH, so performance loss counts.")
    key = (DATA_KEY, p["rule"], float(eol_ah), twin_core(), core_refit())
    saved = st.session_state.get("m3_lives")
    if st.button("Run the replacement study (all batteries that reached end of life)", key="m3_maint_go",
                 type="primary", icon=":material/build:",
                 help="Replays the frozen forecast for each battery (saved results are reused)."):
        lives, notes = m3_lives(fleet, p["rule"])
        saved = {"key": key, "lives": lives, "notes": notes}
        st.session_state["m3_lives"] = saved
    if not saved or saved.get("key") != key:
        placeholder("Press the button: the first run streams every battery through its frozen model (minutes on "
                    "NASA, then saved).")
        return
    lives = saved["lives"]
    if len(lives) < 3:
        st.info("Fewer than three batteries reached end of life: the study needs leave-one-out choices.")
        if saved["notes"]:
            st.caption("Not in the study: " + "; ".join(saved["notes"]))
        return
    e = te.M3Economics(battery_cost_cycles=p["cost"], unplanned_factor=p["k"], lead_cycles=p["lead"])
    mkey = (key, p["cost"], p["k"], p["lead"])
    mc = st.session_state.get("m3_maint_cache")
    if mc and mc[0] == mkey:
        per, summ = mc[1], mc[2]
    else:
        with st.spinner("Replacement study (with and without the anomaly alarms)…"):
            per, _ = te.m3_maintenance_study(lives, e)
            # alarm-aware variants: an Alarm status (FLEET / OWN indicator) also triggers the order; the safety
            # margin is re-chosen leave-one-out with the alarms on
            per_a, _ = te.m3_maintenance_study(lives, e, use_alarm=True)
            per_a = per_a[per_a["policy"].isin(["twin-informed", "twin + age limit"])].copy()
            per_a["policy"] = per_a["policy"].map({"twin-informed": "twin + alarm",
                                                   "twin + age limit": "twin + alarm + age limit"})
            per = pd.concat([per, per_a], ignore_index=True)
            summ = m3_policy_summary(per)
        st.session_state["m3_maint_cache"] = (mkey, per, summ)
    st.session_state["m3_maint"] = {"summary": summ, "per": per, "prices": dict(p)}
    tw = summ.loc["twin-informed"] if "twin-informed" in summ.index else None
    k = st.columns(3)
    k[0].metric("Batteries", f"{len(lives)}", delta=", ".join(sorted({v[1] for v in lives.values()})), delta_color="off")
    if tw is not None:
        k[1].metric("Twin: share of the possible gain", f"{100 * tw['share of perfect-knowledge gain']:.0f} %")
        fx = summ.loc["best fixed age", "share of perfect-knowledge gain"] if "best fixed age" in summ.index else np.nan
        k[2].metric("Best fixed age", fmt(100 * fx, ".0f", "%"))
    show_table(summ.style.format({"mean profit rate": "{:.3f}", "mean unused life (cycles)": "{:.1f}",
                                  "share of perfect-knowledge gain": "{:.0%}"}),
               note="Share of the gain = (profit of the policy − run to end of life) / (perfect knowledge − run to "
                    "end of life), summed over batteries.")
    # per group: groups differ in kind (Cold lives are short and start just as the forecast does)
    rows_g = []
    for gname, g in per.groupby("group"):
        base = g[g["policy"] == "run to end of life"].set_index("battery")["profit rate"]
        best = g[g["policy"] == "perfect knowledge"].set_index("battery")["profit rate"]
        room = (best - base).sum()
        for pol, gp in g.groupby("policy"):
            gp = gp.set_index("battery")
            rows_g.append({"group": gname, "policy": pol, "batteries": len(gp),
                           "unplanned": int(gp["unplanned"].sum()),
                           "share of the possible gain": ((gp["profit rate"] - base.reindex(gp.index)).sum() / room)
                           if room > 0 else np.nan})
    by_g = pd.DataFrame(rows_g).pivot_table(index="policy", columns="group",
                                            values=["share of the possible gain", "unplanned"])
    st.markdown("**By group**")
    show_table(by_g.style.format("{:.0%}", subset=[c for c in by_g.columns if c[0] == "share of the possible gain"])
               .format("{:.0f}", subset=[c for c in by_g.columns if c[0] == "unplanned"]),
               note="Margin and age are still chosen leave-one-out; the share is computed within each group.")
    st.session_state["m3_maint"]["by_group"] = by_g
    show_table(per.pivot_table(index=["group", "battery", "end of life"], columns="policy",
                               values="replace at").style.format("{:.0f}"))
    with st.expander("Sweep over the assumed prices (27 settings)", icon=":material/tune:"):
        if st.button("Run the price sweep", key="m3_maint_sweep_go", icon=":material/play_arrow:",
                     help="27 replacement studies (battery cost x unplanned factor x lead time); a few minutes."):
            rows = []
            bar = st.progress(0.0, text="Price sweep…")
            combos = list(itertools.product((30, 60, 120), (1.5, 3, 5), (5, 10, 20)))
            for j, (cost, kk, lead) in enumerate(combos):
                _, s = te.m3_maintenance_study(lives, te.M3Economics(battery_cost_cycles=cost, unplanned_factor=kk,
                                                                      lead_cycles=lead))
                r = {"battery cost": cost, "unplanned ×": kk, "lead": lead}
                for pol in s.index:
                    r[pol] = s.loc[pol, "share of perfect-knowledge gain"]
                rows.append(r)
                bar.progress((j + 1) / len(combos))
            bar.empty()
            st.session_state["m3_maint_sweep"] = (key, pd.DataFrame(rows))
        sv = st.session_state.get("m3_maint_sweep")
        if sv and sv[0] == key:
            sw = sv[1]
            if "twin-informed" in sw and "best fixed age" in sw:
                ok = sw["twin-informed"].notna() & sw["best fixed age"].notna()
                st.caption(f"Twin above the best fixed age in "
                           f"{(sw.loc[ok, 'twin-informed'] > sw.loc[ok, 'best fixed age']).sum()} of {ok.sum()} "
                           f"settings with room to gain.")
            show_table(sw.set_index(["battery cost", "unplanned ×", "lead"]).style.format("{:.0%}", na_rep="—"))
        else:
            st.caption("Press the button to run the sweep (it is not run automatically: it takes minutes).")
    if saved["notes"]:
        st.caption("Not in the study: " + "; ".join(saved["notes"]))
    download("Download the replacement study (CSV)", per.to_csv(index=False).encode(),
             f"m3_maintenance_study__v{te.ENGINE_VERSION}.csv", "text/csv", key="m3_maint_csv")


def fig_m3_lifecycle(summ: pd.DataFrame, P: Palette) -> go.Figure:
    order = ["do nothing", "usual practice", "twin replacement", "twin strategy", "upper bound"]
    s = summ.reindex(order)
    col = {"do nothing": "#9AA3AD", "usual practice": "#E69F00", "twin replacement": "#56B4E9",
           "twin strategy": "#0072B2", "upper bound": "#6B7A8F"}
    fig = go.Figure(go.Bar(x=order, y=s["mean profit rate (per hour)"], marker_color=[col[o] for o in order],
                           text=[f"{v:.0%}" if np.isfinite(v) else "" for v in s["share of the possible gain"]],
                           textposition="outside", hovertemplate="%{x}: %{y:.3f} per hour<extra></extra>"))
    fig.update_yaxes(title_text="Mean lifecycle profit rate (per hour)")
    return style_fig(fig, P, 420, "Lifecycle profit by strategy (labels: share of the possible gain)")


def m3_lifecycle_section(fleet: Dict[str, Any], p: Dict[str, Any]) -> None:
    section("Mission 3 · 3. Lifecycle: what benefit is expected?")
    st.markdown("Operation and replacement together, in one number per battery: operating profit per hour over a "
                "battery's life, replacement included. **Do nothing**: own operation, run to end of life. **Usual "
                "practice**: own operation, best fixed replacement age. **Twin replacement**: own operation, "
                "twin-informed replacement. **Twin strategy**: the operation that worked best on the *other* "
                "batteries of the group, plus twin-informed replacement. **Upper bound**: best operation and "
                "replacement in hindsight. The measured life stays the reality; another operation stretches or "
                "shortens it by the physics' life ratio, and the twin's replacement timing is assumed as good relative "
                "to end of life as on the measured operation.")
    saved = st.session_state.get("m3_lives")
    h = fleet.get("main")
    if not saved or h is None or len(saved.get("lives") or {}) < 3:
        placeholder("Run the replacement study above first: the lifecycle study uses its replayed lives.")
        return
    try:
        lives = saved["lives"]
        # only the batteries this study needs: those tested at the temperatures of the batteries that reached end of
        # life (a battery's operation options are the combinations run at its temperature); Pulsed / Hot lives of up
        # to ~1700 simulated cycles are skipped -> about 5x less work on Streamlit Cloud
        tested = te.whatif_tested(h, te.condition_groups(CT_RAW))["by_battery"]
        temps = {float(tested.loc[c, "T_C tested"]) for c in lives if c in tested.index}
        keep = sorted(c for c in h["data"] if c in tested.index and float(tested.loc[c, "T_C tested"]) in temps)
        h = dict(h, data={c: h["data"][c] for c in keep})
        lkey = (saved.get("key"), p["cost"], p["buy"], p["k"], p["lead"], p["rule"])
        lc = st.session_state.get("m3_life_cache")
        if lc and lc[0] == lkey:
            per, summ = lc[1], lc[2]
            depth = lc[3] if len(lc) > 3 else None
        else:
            st.caption(f"Combines operation and replacement for the {len(lives)} batteries of the replacement study, "
                       f"simulating the {len(keep)} batteries tested at their temperatures. It runs only when you "
                       f"press the button (about a minute on Streamlit Cloud) and is kept until a price changes.")
            if not st.button("Run the lifecycle study", key="m3_life_go", type="primary", icon=":material/insights:"):
                return
            depth = m3_depth_cached(CT_RAW, DATA_KEY_RAW, tuple(sorted(fleet["main"]["data"]))) \
                if store is not None else None
            with st.spinner("Lifecycle study (operation x replacement)…"):
                per, summ = te.m3_lifecycle_study(
                    h, ct, lives, te.condition_groups(CT_RAW), depth,
                    te.M3OperationEconomics(buy_price_ratio=p["buy"], battery_cost_cycles=p["cost"]),
                    unplanned_factor=p["k"], lead_cycles=p["lead"],
                    soh_eols={c: m3_soh_eol_for(c, p["rule"]) for c in lives})
            st.session_state["m3_life_cache"] = (lkey, per, summ, depth)
    except Exception as exc:
        report_error("Lifecycle study failed", exc, debug)
        return
    if not len(per):
        placeholder("No battery of the replacement study is covered by the shared physics.")
        return
    st.session_state["m3_life"] = {"summary": summ, "per": per, "prices": dict(p)}
    gcol = "vs usual practice (% of operating profit per hour)"
    head = []
    for gname in summ.index.get_level_values(0).unique():
        s_ = summ.loc[gname]
        head.append({"group": gname, "batteries": int(s_["batteries"].iloc[0]),
                     "usual practice (per hour)": s_.loc["usual practice", "mean profit rate (per hour)"],
                     "twin strategy vs usual (%)": s_.loc["twin strategy", gcol],
                     "from replacement (%)": s_.loc["twin replacement", gcol],
                     "from operation (%)": s_.loc["twin strategy", gcol] - s_.loc["twin replacement", gcol],
                     "upper bound vs usual (%)": s_.loc["upper bound", gcol]})
    head = pd.DataFrame(head).set_index("group")
    groups_avail = [g_ for g_ in head.index if g_ != "all"] + ["all"]
    default = "Reference" if "Reference" in groups_avail else groups_avail[0]
    gsel = st.segmented_control("Group", groups_avail, default=default, key="m3_life_group") or default
    s_ = summ.loc[gsel]
    g = s_[gcol]
    k = st.columns(3)
    k[0].metric(f"Twin strategy vs usual practice · {gsel}", f"{g['twin strategy']:+.0f} %",
                help="Difference in mean lifecycle profit rate, as % of the usual practice's operating profit per hour.")
    k[1].metric("from replacement", f"{g['twin replacement']:+.0f} %")
    k[2].metric("from operation", f"{g['twin strategy'] - g['twin replacement']:+.0f} %")
    if (s_["mean profit rate (per hour)"] < 0).any():
        st.caption("Negative profit rates: at these prices a battery of this group does not pay back its own cost "
                   "within its life (short lives). Percentages there compare losses; read them with care.")
    show(fig_m3_lifecycle(s_, P), key="m3_life_fig", data=s_.reset_index())
    show_table(head.style.format({"usual practice (per hour)": "{:.3f}", "twin strategy vs usual (%)": "{:+.1f}",
                                  "from replacement (%)": "{:+.1f}", "from operation (%)": "{:+.1f}",
                                  "upper bound vs usual (%)": "{:+.1f}"}),
               note="One line per group; 'all' pools groups of different kinds and is shown for completeness only.")
    cols = ["group", "battery", "end of life", "own option", "chosen option", "life ratio (chosen / own)",
            "do nothing", "usual practice", "twin replacement", "twin strategy", "upper bound"]
    show_table(per[cols].set_index(["group", "battery"]).style.format(
        {c: "{:.3f}" for c in cols[5:]} | {"end of life": "{:.0f}"}),
        note="Profit rates per hour. Negative values: the battery's life is too short to pay back its cost at these "
             "prices (Cold batteries).")
    with st.expander(f"Sweep over the assumed prices · {gsel}", icon=":material/tune:"):
        if st.button("Run the lifecycle price sweep", key="m3_life_sweep_go", icon=":material/play_arrow:",
                     help="27 lifecycle studies (battery cost x unplanned factor x lead time): this takes several "
                          "minutes; all groups are computed at once."):
            rows = []
            bar = st.progress(0.0, text="Lifecycle price sweep…")
            combos = list(itertools.product((30, 60, 120), (1.5, 3, 5), (5, 10, 20)))
            for j, (cost, kk, lead) in enumerate(combos):
                try:
                    _, s = te.m3_lifecycle_study(h, ct, lives, te.condition_groups(CT_RAW), depth,
                                                 te.M3OperationEconomics(buy_price_ratio=p["buy"],
                                                                         battery_cost_cycles=cost),
                                                 unplanned_factor=kk, lead_cycles=lead,
                                                 soh_eols={c: m3_soh_eol_for(c, p["rule"]) for c in lives})
                except Exception:
                    continue
                for gname in s.index.get_level_values(0).unique():
                    gg = s.loc[gname, "vs usual practice (% of operating profit per hour)"]
                    rows.append({"group": gname, "battery cost": cost, "unplanned ×": kk, "lead": lead,
                                 "twin strategy (%)": gg["twin strategy"],
                                 "twin replacement (%)": gg["twin replacement"], "upper bound (%)": gg["upper bound"]})
                bar.progress((j + 1) / len(combos))
            bar.empty()
            st.session_state["m3_life_sweep"] = (saved.get("key"), p["buy"], p["rule"], pd.DataFrame(rows))
        sv = st.session_state.get("m3_life_sweep")
        if sv and sv[:3] == (saved.get("key"), p["buy"], p["rule"]) and len(sv[3]):
            sw = sv[3][sv[3]["group"] == gsel].drop(columns="group")
            if len(sw):
                st.caption(f"{gsel}: twin strategy above usual practice in {(sw['twin strategy (%)'] > 0).sum()} of "
                           f"{len(sw)} settings; median {sw['twin strategy (%)'].median():+.0f} %.")
                show_table(sw.set_index(["battery cost", "unplanned ×", "lead"]).style.format("{:+.1f}"))
        else:
            st.caption("Press the button to run the sweep (it is not run automatically: it takes several minutes).")
    download("Download the lifecycle study (CSV)", per.to_csv(index=False).encode(),
             f"m3_lifecycle_study__v{te.ENGINE_VERSION}.csv", "text/csv", key="m3_life_csv")


def m3_forecast_section(fleet: Dict[str, Any]) -> None:
    """The forecaster behind the replacement decision (STATUS 5ce): one battery, one 'today', the end-of-life forecast
    of the chosen cores against what happened. Uses the same saved streams as Mission 3 and the twin scorecard."""
    section("Mission 3 · Forecast behind the replacement decision")
    st.markdown(f"The maintenance and lifecycle studies below use the twin core chosen in the sidebar: "
                f"**{twin_core()}**" + (f", retrained every {core_refit()} cycles" if core_refit() else "")
                + ". Here you see its forecast for one battery at a chosen *today*, next to other cores, against "
                  "what really happened. Nothing after *today* is used by any forecast.")
    rule = M3_EOL_RULES.get(st.session_state.get("m3_eol", list(M3_EOL_RULES)[0]), "nasa")
    cands = _score_cells(rule, TWIN_CORES[0])
    if not cands:
        st.info("No battery reached end of life under the Mission 3 end-of-life rule.")
        return
    c1, c2 = st.columns([1, 2])
    cell = c1.selectbox("Battery", cands, key="m3f_cell")
    cores = c2.multiselect("Cores to compare", list(TWIN_CORES), default=[TWIN_CORES[0], TWIN_CORES[2]],
                           key="m3f_cores")
    eol_c = float(m3_soh_eol_for(cell, rule))
    meas = CT_RAW[(CT_RAW["Cell_ID"] == cell) & ~CT_RAW["outlier"]][["n", "SOH"]].dropna().sort_values("n")
    e_true = te.eol_cycle_from_replay(meas.reset_index(drop=True), eol_c)
    grp_c, fz0, train0 = frozen_context(cell, TWIN_CORES[0])
    ns, nm = te.frozen_live_window(CT_RAW, cell, train0)
    hi_ = int(e_true) if e_true else int(meas["n"].max())
    today = st.slider("Today (cycle)", int(ns), max(int(ns) + 1, hi_), int(min(ns + 10, hi_)), key="m3f_today")
    fkey = (DATA_KEY, rule, float(eol_ah), core_refit(TWIN_CORES[1]), cell, tuple(cores))
    got = st.session_state.get("m3f_frames")
    if st.button("Show the forecasts", key="m3f_go", icon=":material/show_chart:",
                 help="Streams the battery through each core once (saved; PINN cores take minutes the first time "
                      "unless the twin scorecard already streamed them)."):
        got = {"key": fkey, "frames": {}}
        for core in cores:
            g_, fz, train_c = frozen_context(cell, core)
            with st.spinner(f"{cell} · {core}…"):
                try:
                    fr, _ = frozen_cached(CT_RAW, imp, DATA_KEY, cell, tuple(train_c), fz[0], fz[1], eol_c,
                                          core_refit(core), "capacity", float(eol_ah), ns, nm)
                    got["frames"][core] = (fr, model_label(fz[0]))
                except Exception as exc:
                    st.warning(f"{core}: forecast failed ({exc})")
        st.session_state["m3f_frames"] = got
    if not got or got.get("key") != fkey:
        placeholder("Choose a battery and the cores, then press the button.")
        return
    colors = {TWIN_CORES[0]: "#56B4E9", TWIN_CORES[1]: "#E69F00", TWIN_CORES[2]: "#009E73"}
    fig = go.Figure()
    past, fut_m = meas[meas["n"] <= today], meas[meas["n"] > today]
    fig.add_trace(go.Scatter(x=past["n"], y=100 * past["SOH"], mode="markers", name="Measured (up to today)",
                             marker=dict(size=5, color="#9AA3AD")))
    fig.add_trace(go.Scatter(x=fut_m["n"], y=100 * fut_m["SOH"], mode="markers", name="Measured later (unseen)",
                             marker=dict(size=5, color="#9AA3AD", opacity=0.35)))
    rows = []
    for core, (fr, label) in got["frames"].items():
        ks = [k for k in fr if k <= today]
        if not ks:
            continue
        fut, med, lo, hi = (np.asarray(x, float) for x in fr[max(ks)][:4])
        col = colors.get(core, "#CC79A7")
        fig.add_trace(go.Scatter(x=np.r_[fut, fut[::-1]], y=100 * np.r_[hi, lo[::-1]], fill="toself", opacity=0.15,
                                 line=dict(width=0), fillcolor=col, name=f"{core} · 80 % band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fut, y=100 * med, mode="lines", line=dict(color=col, width=3), name=core))
        hit = np.nonzero(med <= eol_c)[0]
        e_pred = float(fut[hit[0]]) if len(hit) else np.nan
        if np.isfinite(e_pred):
            fig.add_trace(go.Scatter(x=[e_pred], y=[100 * eol_c], mode="markers", showlegend=False,
                                     marker=dict(symbol="diamond", size=13, color=col),
                                     hovertemplate=f"{core}: end of life at %{{x:.0f}}<extra></extra>"))
        rows.append({"core": core, "model": label, "forecast made at cycle": max(ks),
                     "predicted end of life (cycle)": e_pred, "real end of life (cycle)": e_true,
                     "error (cycles, + = late)": (e_pred - e_true) if (e_true and np.isfinite(e_pred)) else np.nan})
    fig.add_hline(y=100 * eol_c, line_dash="dash", line_color="#D55E00",
                  annotation_text=f"end of life {100 * eol_c:.0f} %", annotation_position="bottom left")
    fig.add_vline(x=today, line_dash="dot", line_color="#666", annotation_text="today")
    if e_true:
        fig.add_vline(x=e_true, line_dash="dash", line_color="#D55E00", annotation_text="real end of life",
                      annotation_position="top left")
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text="SOH (%)")
    show(style_fig(fig, P, 480, f"{cell} · end-of-life forecast at cycle {today}"), key="m3f_fig")
    if rows:
        show_table(pd.DataFrame(rows).set_index("core").style.format(
            {"predicted end of life (cycle)": "{:.0f}", "real end of life (cycle)": "{:.0f}",
             "error (cycles, + = late)": "{:+.0f}"}, na_rep="—"),
            note="Move today to replay the battery's life: each core's forecast uses only the cycles up to today. "
                 "Diamonds: predicted end of life; dashed lines: threshold and real end of life.")


def m3_intro() -> None:
    """Mission 3 at a glance: the three questions, and where each is answered on this page."""
    section("Mission 3 · Operation and maintenance")
    st.markdown(
        "Mission 3 turns the twin into decisions. **(1) How should a battery be operated?** The operation study "
        "(1) compares the operations NASA tested; the operating window (1b) explores a grid of conditions for one "
        "battery; the general window (1c) does it for every battery. **(2) When should it be replaced?** The "
        "replacement study (2) compares the twin with a fixed age and run-to-failure. **(3) What is the benefit?** "
        "The lifecycle study (3) combines operation and replacement. The **summary** at the end of the page collects "
        "the answers for the meeting and the thesis.")


def m3_general_window_section(fleet: Dict[str, Any]) -> None:
    """Mission 3 · 1c. The operating window for EVERY battery, each simulated as new, over a grid inside NASA's tested
    ranges; scenarios are scored relative to each battery's best, then averaged (the Colab study of 11 Nov 2026)."""
    section("Mission 3 · 1c. General operating window: the best conditions for every battery")
    st.markdown(
        "Every battery of the shared physics is simulated **as a new battery** under every scenario of a grid inside "
        "the ranges NASA tested. Each scenario's lifetime energy is expressed **relative to that battery's best** "
        "(1 = its best), then averaged over the batteries: a scenario that is good for most batteries is a general "
        "recommendation.")
    st.warning("Not a validated prediction: the ageing of each battery is scaled to its own recorded fade (a fit to "
               "its whole life), and the shared physics is fitted on all batteries. Use the RANKING of conditions; "
               "the absolute cycle counts are indicative. The data cannot identify an effect of discharge current.",
               icon=":material/warning:")
    h = fleet.get("main")
    if h is None:
        placeholder("The general window needs the shared physics (fleet-wide parts).")
        return
    groups = te.condition_groups(CT_RAW)
    tested = te.whatif_tested(h, groups)
    v = tested["values"]
    c1, c2, c3 = st.columns(3)
    n_t = c1.select_slider("Temperatures", [3, 5, 7, 9], value=9, key="gen_nt")
    n_i = c2.select_slider("Currents", [2, 3, 5, 7], value=7, key="gen_ni")
    n_v = c3.select_slider("Cut-offs", [2, 4, 6, 8], value=8, key="gen_nv")
    T_sc = np.round(np.linspace(min(v["T_C"]), max(v["T_C"]), n_t), 1)
    I_sc = np.round(np.linspace(min(v["I_dis"]), max(v["I_dis"]), n_i), 2)
    cells = sorted(h["data"])
    gkey = (DATA_KEY, n_t, n_i, n_v)
    if st.button(f"Simulate {n_t * n_i * n_v} scenarios x {len(cells)} batteries", key="gen_go", type="primary",
                 icon=":material/grid_view:"):
        try:
            depth = m3_depth_cached(CT_RAW, DATA_KEY, tuple(cells)) if store is not None else None
            dv = depth.reset_index(drop=("battery" in depth.columns)) if depth is not None and len(depth) else None
            v_lo = max(2.0, float(dv["V"].min())) if dv is not None else 2.7
            V_sc = np.round(np.linspace(v_lo, 2.7, n_v), 2)
            rows, bar = [], st.progress(0.0, text="Simulating…")
            for j, b in enumerate(cells):
                try:
                    W = te.m3_operating_window(h, b, 0, 1.0, m3_soh_eol_for(b, "nasa"), T_sc, I_sc, V_sc, depth=depth,
                                               groups=groups)["table"]
                    W = W.assign(battery=b, group=str(groups.loc[b, "Group"]) if b in groups.index else "")
                    W["relative energy"] = W["energy to EOL (Wh)"] / W["energy to EOL (Wh)"].max()
                    W["in top 10 %"] = W["relative energy"] >= W["relative energy"].quantile(0.9)
                    rows.append(W)
                except Exception:
                    pass
                bar.progress((j + 1) / len(cells), text=f"{b} done")
            bar.empty()
            st.session_state["m3_gen"] = {"key": gkey, "all": pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()}
        except Exception as exc:
            report_error("general operating window", exc, False)
    R = st.session_state.get("m3_gen")
    if not R or R["key"] != gkey or not len(R["all"]):
        placeholder("Press the button (about 10 s per battery on Streamlit Cloud).")
        return
    A = R["all"]
    G = (A.groupby(["T_C", "I_dis", "V_cut"])
         .agg(**{"mean relative energy": ("relative energy", "mean"), "worst battery": ("relative energy", "min"),
                 "share of batteries in their top 10 %": ("in top 10 %", "mean"),
                 "median cycles to EOL": ("remaining cycles", "median"),
                 "cycles, 10 % battery": ("remaining cycles", lambda x: x.quantile(0.1)),
                 "cycles, 90 % battery": ("remaining cycles", lambda x: x.quantile(0.9)),
                 "median energy to EOL (Wh)": ("energy to EOL (Wh)", "median")})
         .reset_index().sort_values("mean relative energy", ascending=False).reset_index(drop=True))
    st.session_state["m3_gen"]["summary"] = G
    best = G.iloc[0]
    k = st.columns(4)
    k[0].metric("Best general operation", f"{best['T_C']:g} °C · {best['I_dis']:g} A · {best['V_cut']:g} V")
    k[1].metric("Median cycles to EOL", f"{best['median cycles to EOL']:.0f}",
                delta=f"{best['cycles, 10 % battery']:.0f}–{best['cycles, 90 % battery']:.0f} (10–90 %)", delta_color="off")
    k[2].metric("Top-10 % choice for", f"{100 * best['share of batteries in their top 10 %']:.0f} % of batteries")
    k[3].metric("Lives simulated", f"{len(A):,}")
    show_table(G.head(5).style.format({"T_C": "{:g}", "I_dis": "{:g}", "V_cut": "{:g}", "mean relative energy": "{:.3f}",
                                       "worst battery": "{:.3f}", "share of batteries in their top 10 %": "{:.0%}",
                                       "median cycles to EOL": "{:.0f}", "cycles, 10 % battery": "{:.0f}",
                                       "cycles, 90 % battery": "{:.0f}", "median energy to EOL (Wh)": "{:.0f}"}),
               note="Top 5 scenarios by mean relative lifetime energy (1 = each battery's best).")
    T_mid = sorted(G["T_C"].unique())[len(G["T_C"].unique()) // 2]
    tsel = st.select_slider("Temperature of the maps (°C)", sorted(G["T_C"].unique()), value=T_mid, key="gen_tmap")
    M1 = G[np.isclose(G["T_C"], tsel)].pivot_table(index="V_cut", columns="I_dis", values="mean relative energy")
    M2 = G[np.isclose(G["T_C"], tsel)].pivot_table(index="V_cut", columns="I_dis", values="median cycles to EOL")
    m1, m2 = st.columns(2)
    for col_, M, title, cs, fmt_ in ((m1, M1, "Mean relative lifetime energy", "Viridis", ".2f"),
                                     (m2, M2, "Median cycles to end of life", "Magma", ".0f")):
        fig = go.Figure(go.Heatmap(z=M.to_numpy(), x=[f"{x:g} A" for x in M.columns], y=[f"{y:g} V" for y in M.index],
                                   colorscale=cs, texttemplate=f"%{{z:{fmt_}}}", hovertemplate="%{y} · %{x}: %{z}<extra></extra>"))
        fig.update_layout(title=f"{title} at {tsel:g} °C", height=360, margin=dict(l=10, r=10, t=40, b=10))
        col_.plotly_chart(fig, width="stretch", key=f"gen_map_{title}")
    sel = A[np.isclose(A["I_dis"], best["I_dis"]) & np.isclose(A["V_cut"], best["V_cut"])].groupby("T_C")["remaining cycles"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(sel.quantile(0.9).index) + list(sel.quantile(0.1).index[::-1]),
                             y=list(sel.quantile(0.9).values) + list(sel.quantile(0.1).values[::-1]), fill="toself",
                             line=dict(width=0), opacity=0.25, name="10–90 % of batteries", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sel.median().index, y=sel.median().values, mode="lines+markers", name="median"))
    fig.update_layout(title=f"Temperature effect at {best['I_dis']:g} A · {best['V_cut']:g} V", height=330,
                      xaxis_title="temperature (°C)", yaxis_title="cycles to end of life", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch", key="gen_temp")
    download("Download all simulated lives (CSV)", A.to_csv(index=False).encode(), "general_operating_window.csv",
             "text/csv", key="gen_dl")


def m3_summary_section() -> None:
    """Mission 3 · summary: the three answers and one decision per battery, from the studies above (same run)."""
    section("Mission 3 · Summary: the answers, and one decision per battery")
    ss = st.session_state
    maint, life, ops, gen = ss.get("m3_maint"), ss.get("m3_life"), ss.get("m3_ops"), ss.get("m3_gen")
    missing = [n for n, x in (("operation study (1)", ops), ("general window (1c)", gen), ("replacement study (2)", maint),
                              ("lifecycle study (3)", life)) if x is None or (isinstance(x, pd.DataFrame) and not len(x))]
    if missing:
        st.info("Run these studies above to complete the summary: " + ", ".join(missing) + ".")
    answers = []
    if gen is not None and "summary" in gen:
        b = gen["summary"].iloc[0]
        answers.append(("How should a battery be operated?",
                        f"In general: **{b['T_C']:g} °C, {b['V_cut']:g} V cut-off** (median {b['median cycles to EOL']:.0f} "
                        f"cycles to end of life; a top-10 % choice for {100 * b['share of batteries in their top 10 %']:.0f} % "
                        "of the batteries). A shallower cut-off helps; cold operation is very harmful. Current: not "
                        "identifiable from the data."))
    if maint is not None and "summary" in maint:
        s = maint["summary"]
        def share(pol):
            return f"{100 * s.loc[pol, 'share of perfect-knowledge gain']:.0f} %" if pol in s.index else "–"
        answers.append(("When should it be replaced?",
                        f"With the **twin plus an age limit**: {share('twin + age limit')} of the possible gain, against "
                        f"{share('best fixed age')} for the best fixed age and 0 % for run to failure (twin alone "
                        f"{share('twin-informed')}). The age limit catches lives too short to forecast in time."))
    if life is not None and "summary" in life:
        summ = life["summary"]
        gcol = "vs usual practice (% of operating profit per hour)"
        parts = []
        for gname in summ.index.get_level_values(0).unique():
            if gname == "all":
                continue
            try:
                parts.append(f"{gname} {summ.loc[(gname, 'twin strategy'), gcol]:+.0f} %")
            except Exception:
                pass
        answers.append(("What is the benefit?", "Twin strategy (replacement + operation) against usual practice, profit "
                        "per hour: **" + ", ".join(parts) + "**. Prices are assumptions: the ranking is meaningful, the "
                        "absolute profits are not."))
    for q, a in answers:
        st.markdown(f"**{q}** {a}")
    if maint is not None and "per" in maint:
        per = maint["per"]
        rows = []
        for (bat, grp), g in per.groupby(["battery", "group"]):
            pol = g.set_index("policy")
            r = {"battery": bat, "group": grp,
                 "end of life (cycle)": float(g["end of life"].iloc[0]) if "end of life" in g else np.nan}
            for name, col in (("twin + age limit", "replace at (twin + age limit)"), ("best fixed age", "replace at (fixed age)")):
                if name in pol.index:
                    r[col] = float(pol.loc[name, "replace at"])
                    r[f"failure before ({name})"] = bool(pol.loc[name, "unplanned"])
            if ops is not None and len(ops) and "battery" in ops:
                o = ops[ops["battery"] == bat]
                if len(o):
                    b_ = o.sort_values("profit rate (per hour)", ascending=False).iloc[0]
                    r["best tested operation"] = f"{b_['I_dis']:g} A · {b_['V_cut']:g} V"
            rows.append(r)
        D = pd.DataFrame(rows).set_index(["group", "battery"]).sort_index()
        show_table(D, note="One decision per battery: when the twin (with an age limit) replaces it against the fixed "
                           "age, whether the battery failed first, and its most profitable tested operation.")
        lines = [f"- **{q}** {a}" for q, a in answers]
        Dr = D.reset_index().astype(str)
        table_md = "| " + " | ".join(Dr.columns) + " |\n|" + "---|" * len(Dr.columns) + "\n" + \
            "\n".join("| " + " | ".join(r) + " |" for r in Dr.to_numpy())
        report = "# Mission 3 · summary\n\n" + "\n".join(lines) + "\n\n" + table_md + "\n"
        cA, cB = st.columns(2)
        with cA:
            download("Download the decision table (CSV)", D.reset_index().to_csv(index=False).encode(),
                     "mission3_decisions.csv", "text/csv", key="m3_sum_csv")
        with cB:
            download("Download the summary (Markdown)", report.encode(), "mission3_summary.md", "text/markdown",
                     key="m3_sum_md")


def view_ops() -> None:
    recommendations("ops")
    m3_intro()
    anomaly_section()
    p = m3_prices()
    fleet = twin_fleet_parts(build=False)
    if fleet is None:
        section("Mission 3 · Optimisation")
        if st.button("Build the twin's fleet-wide parts", key="m3_build", type="primary", icon=":material/hub:",
                     help="Shared physics and anomaly scan for the whole fleet, once per session (shared with the "
                          "Integrated twin page and saved in the result backup)."):
            fleet = twin_fleet_parts(build=True)
        else:
            placeholder("Mission 3 uses the twin's fleet-wide parts (shared physics, anomaly scan). Build them once.")
            return
    m3_operation_section(fleet, p)
    m3_window_section(fleet)
    m3_general_window_section(fleet)
    m3_forecast_section(fleet)
    m3_maintenance_section(fleet, p)
    m3_lifecycle_section(fleet, p)
    m3_summary_section()


# =============================================================================
# Dispatch: only the active view runs
# =============================================================================
sidebar_guide()
_VIEW_FN: Dict[str, Callable[[], None]] = {V_HOME: view_home, V_DATA: view_data, V_MODELS: view_models,
                                           V_UPDATE: view_update, V_LIVE: view_replay, V_TWIN: view_twin, V_OPS: view_ops,
                                           V_STUDY: view_study}
try:
    ga_autoload()
    if twin_core() == TWIN_CORES[2] and not ga_loaded():
        with st.sidebar:
            st.caption(":orange[No GA-PINN results found (results/ga_pinn_evaluations.csv): run the GA or restore it on "
                       "the Integrated twin page. Until then every battery uses the PINN twin core.]")
except Exception:
    pass
try:
    _VIEW_FN.get(view, view_data)()
except Exception as exc:
    report_error(f"{view} failed", exc, debug)
