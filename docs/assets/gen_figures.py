"""
gen_figures.py
==============
Regenerate the static figures used in the documentation.

Run from the repo root with the ``ccdaf`` environment::

    python docs/assets/gen_figures.py

Outputs land in ``docs/assets/img/``. Only diagrams/plots that can be produced
without a running GUI live here; GUI screenshots are captured by hand.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

# Region label -> (name, colour). Mirrors region_tagger.LABELS and the app's
# colour map; body is the background label.
REGIONS = [
    (11, "LSPV", "#e41a1c"),
    (13, "LIPV", "#377eb8"),
    (15, "RSPV", "#4daf4a"),
    (17, "RIPV", "#984ea3"),
    (19, "LAA",  "#ff7f00"),
    (1,  "body", "#d9d9d9"),
]


def region_legend():
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=140)
    ax.set_axis_off()
    for i, (lbl, name, colour) in enumerate(REGIONS):
        y = len(REGIONS) - 1 - i
        ax.add_patch(Rectangle((0.0, y + 0.15), 0.7, 0.7, facecolor=colour,
                               edgecolor="#333333", linewidth=0.8))
        ax.text(0.95, y + 0.5, f"{lbl:>2}  {name}", va="center", ha="left",
                fontsize=12, family="monospace")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(REGIONS))
    ax.set_title("elemTag region labels", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "region-legend.png", bbox_inches="tight")
    plt.close(fig)


def electrode_convergence():
    # Residual max|d0 - d1| after each Newton step, measured on a 5%-radius
    # + noise move (see docs/concepts.md). The default is 3 iterations.
    residual = [6.785e-2, 5.499e-3, 1.391e-4, 1.110e-16, 1.388e-16, 1.110e-16]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
    ax.semilogy(range(len(residual)), residual, "o-", color="#b3202f")
    ax.axvline(3, color="#377eb8", ls="--", lw=1.2,
               label="EAM_SDF_ITERATIONS = 3")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel(r"max $|d_0 - d_1|$  (distance-to-wall residual)")
    ax.set_title("Electrode displacement converges in ~2 steps", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "electrode-convergence.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region_legend()
    electrode_convergence()
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))
