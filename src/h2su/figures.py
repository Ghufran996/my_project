"""Figure helpers: one style, PNG at 1000 dpi (D8), raw data kept next to the figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import CONFIGS, FIGURES

COL_W_IN = 3.543   # ECM single column: 3543 px at 1000 dpi
FULL_W_IN = 7.48   # ECM full page width: 7480 px at 1000 dpi


# Brief 7 item 6 (m4): shared site palette and markers; pairwise gray-level difference (ITU-R BT.601 luma, 0-255) >= 30 is asserted in tests/test_site_palette.py
SITES = ["Cabauw", "Carpentras", "Gobabeb"]
SITE_COLORS = {"Cabauw": "#0072B2", "Carpentras": "#D55E00", "Gobabeb": "#000000"}
SITE_MARKERS = {"Cabauw": "o", "Carpentras": "s", "Gobabeb": "^"}


def gray_level(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def apply_style() -> None:
    plt.style.use(str(CONFIGS / "paper.mplstyle"))


def save_png(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=1000.5, format="png")
    plt.close(fig)
    return path


def plot_ver1(csv_path: Path, name: str = "ver1_rauls_heatup") -> Path:
    """VER-1: hydrogen produced during potentiostatic (1.80 V) heat-up 25->75 degC vs published Table 5 of L26."""
    apply_style()
    df = pd.read_csv(csv_path)
    d = df[(df.attempt != "heating_only_check") & (df.thermal_basis == "tables")].copy()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL_W_IN, 2.9))
    pub = d.drop_duplicates("P_nom_kW").sort_values("P_nom_kW")
    axa.plot(pub.P_nom_kW, pub.mH2_published_kg, "o-", color="black", label="L26 Table 5 (published)", zorder=3)
    styles = {"A_as_printed": ("s", "#0072B2", "Model A: as printed"),
              "B_sigma_scaled_to_stated_operating_point": ("^", "#D55E00", "Model B: Eq. (10) rescaled (diagnostic)")}
    for att, (mk, col, lab) in styles.items():
        s = d[d.attempt == att]
        first = True
        for _, r in s.iterrows():
            off = 1.0
            if r.P_nom_kW == 100:
                off = 0.88 if r.n_stacks == 1 else 1.14
            axa.plot(r.P_nom_kW * off, r.mH2_model_kg, mk, color=col, label=lab if first else None, zorder=2)
            axb.plot(r.P_nom_kW * off, r.RE_pct, mk, color=col, zorder=2)
            if not r.reached_75C:
                axa.annotate("n.r.", (r.P_nom_kW * off, r.mH2_model_kg), textcoords="offset points", xytext=(5, -2), fontsize=6)
                axb.annotate("n.r.", (r.P_nom_kW * off, r.RE_pct), textcoords="offset points", xytext=(5, -2), fontsize=6)
            first = False
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel("Nominal electrical power (kW)")
    axa.set_ylabel("H$_2$ produced during heat-up (kg)")
    axa.legend(loc="upper left", fontsize=6)
    axb.axhspan(-10, 10, color="#999999", alpha=0.25, lw=0, label="±10 % acceptance band (SC-R)")
    axb.axhline(0, color="black", lw=0.5)
    axb.set_xscale("log")
    axb.set_xlabel("Nominal electrical power (kW)")
    axb.set_ylabel("Relative error to Table 5 (%)")
    axb.plot([], [], " ", label="n.r.: 75 °C not reached within 6 h")
    axb.legend(loc="upper right", fontsize=6)
    axa.minorticks_off()
    axb.minorticks_off()
    for ax, lab in ((axa, "A"), (axb, "B")):
        ax.text(-0.14, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=9)
    return save_png(fig, name)
