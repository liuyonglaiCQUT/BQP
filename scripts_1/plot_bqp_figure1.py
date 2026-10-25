"""Generate a compact, English-only BQP overview for a single paper column."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results_bqp" / "figures"

BLUE = "#174A9C"
RED = "#D62728"
GREEN = "#2CA02C"
BLACK = "#111111"
GRAY = "#555555"


def serif_font(size: float, *, bold: bool = False) -> FontProperties:
    """Use Matplotlib's portable serif fallback chain."""
    return FontProperties(family="serif", weight="bold" if bold else "normal", size=size)


def box(ax, x, y, width, height, *, dashed=False, linewidth=0.9):
    patch = Rectangle(
        (x, y),
        width,
        height,
        facecolor="white",
        edgecolor=BLACK,
        linewidth=linewidth,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=1,
    )
    ax.add_patch(patch)


def draw_figure():
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        }
    )

    regular = serif_font(7.0)
    small = serif_font(6.2)
    tiny = serif_font(5.7)
    bold = serif_font(7.2, bold=True)
    bold_small = serif_font(6.5, bold=True)

    # Compact aspect ratio, close to the issue-visualization style in the reference.
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # One held-out fact is the common source of both evaluation records.
    box(ax, 0.035, 0.805, 0.93, 0.155, dashed=True, linewidth=1.0)
    ax.text(0.5, 0.922, "One Test Triple (source fact)", ha="center", va="center", color=BLUE, fontproperties=bold)
    ax.text(0.5, 0.852, r"$\langle h_i,\ r_i,\ t_i\rangle$", ha="center", va="center", color=BLACK, fontsize=8.3)

    # The two records produced by standard bidirectional evaluation.
    box(ax, 0.035, 0.545, 0.45, 0.215)
    box(ax, 0.515, 0.545, 0.45, 0.215)

    ax.text(0.26, 0.722, "Tail Query", ha="center", va="center", color=BLUE, fontproperties=bold)
    ax.text(0.26, 0.657, r"$\langle h_i,\ r_i,\ ?\rangle$", ha="center", va="center", fontsize=7.6)
    ax.text(0.26, 0.590, r"reciprocal rank:  $T_i=1/R_i^{\mathrm{tail}}$", ha="center", va="center", color=GRAY, fontproperties=small)

    ax.text(0.74, 0.722, "Head Query", ha="center", va="center", color=BLUE, fontproperties=bold)
    ax.text(0.74, 0.657, r"$\langle ?,\ r_i,\ t_i\rangle$", ha="center", va="center", fontsize=7.6)
    ax.text(0.74, 0.590, r"reciprocal rank:  $H_i=1/R_i^{\mathrm{head}}$", ha="center", va="center", color=GRAY, fontproperties=small)

    ax.text(
        0.5,
        0.486,
        r"Same point estimate:  $\widehat{\mathrm{MRR}}=\mathrm{mean}_i[(T_i+H_i)/2]$",
        ha="center",
        va="center",
        color=BLACK,
        fontproperties=small,
    )

    # Two uncertainty procedures: only the right-hand one preserves the source pair.
    box(ax, 0.035, 0.105, 0.45, 0.315)
    box(ax, 0.515, 0.105, 0.45, 0.315)

    ax.text(0.26, 0.380, "Direction-wise Resampling", ha="center", va="center", color=BLUE, fontproperties=bold_small)
    ax.text(0.26, 0.318, r"Draw $T_i^*$ and $H_j^*$ separately", ha="center", va="center", color=BLACK, fontproperties=regular)
    ax.text(0.26, 0.263, r"indices $i$ and $j$ vary independently", ha="center", va="center", color=GRAY, fontproperties=tiny)
    ax.text(0.268, 0.180, "Source pairing:", ha="right", va="center", color=BLACK, fontproperties=small)
    ax.text(0.275, 0.180, "BROKEN", ha="left", va="center", color=RED, fontproperties=bold_small)
    ax.text(0.26, 0.132, r"$\mathrm{Cov}(T,H)$ is discarded", ha="center", va="center", color=RED, fontproperties=tiny)

    ax.text(0.74, 0.380, "Source-triple Resampling", ha="center", va="center", color=BLUE, fontproperties=bold_small)
    ax.text(0.74, 0.318, r"Draw $(T_i^*,H_i^*)$ jointly", ha="center", va="center", color=BLACK, fontproperties=regular)
    ax.text(0.74, 0.263, r"one index $i$ selects both directions", ha="center", va="center", color=GRAY, fontproperties=tiny)
    ax.text(0.748, 0.180, "Source pairing:", ha="right", va="center", color=BLACK, fontproperties=small)
    ax.text(0.755, 0.180, "RETAINED", ha="left", va="center", color=GREEN, fontproperties=bold_small)
    ax.text(0.74, 0.132, r"$\mathrm{Cov}(T,H)$ is preserved", ha="center", va="center", color=GREEN, fontproperties=tiny)

    ax.text(
        0.5,
        0.045,
        "Same MRR, different uncertainty",
        ha="center",
        va="center",
        color=BLACK,
        fontproperties=bold,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "bqp_figure1_overview"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
