"""Draws the run-to-run variation figure used on slide 10 of the presentation.

Slide 10 carries the project's main finding: test macro F1 moves as much between
identical runs as the distance to the target the project was trying to reach. As
three numbers in three boxes that has to be read and explained. Plotted on one axis
it can be seen, because the run-to-run spread and the gap to target are distances on
the same scale and can be drawn as such.

Reads reports/improvement_experiments_notebook_run.json (S5, the five fixed-seed
retrains), reports/cross_validation_results.json (the five folds) and
reports/significance_tests.json (the interval). Retrains nothing, so it is
deterministic. Writes reports/run_spread.png.

Run from the repository root:

    python run_spread_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Deck palette, so the figure does not look pasted in from elsewhere.
NAVY = "#1E2A44"
BODY = "#222A38"
MUTED = "#6C7483"
TEAL = "#0E7C86"
RED = "#B23A48"
PANEL = "#F1F4F9"

TARGET = 0.75
DELIVERED = 0.6754
OUT = Path("reports/run_spread.png")

plt.rcParams.update({
    # Calibri is the deck face but is not installed here; Helvetica Neue is the
# closest humanist sans available on macOS and reads the same at slide size.
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def main():
    exp = json.loads(Path("reports/improvement_experiments_notebook_run.json").read_text())
    cv = json.loads(Path("reports/cross_validation_results.json").read_text())
    sig = json.loads(Path("reports/significance_tests.json").read_text())

    repeats = np.array(exp["S5_seed_ensemble"]["individual_test_macro_f1"], dtype=float)
    bert = np.array(cv["bert"]["per_fold_macro_f1"], dtype=float)
    svm = np.array(cv["svm"]["per_fold_macro_f1"], dtype=float)
    bert_lo, bert_hi = sig["bert_mean"]["ci95_mean"]

    fig, ax = plt.subplots(figsize=(10.6, 4.68), dpi=200)

    rows = [
        (4.0, repeats, TEAL, "Same config, retrained 5×", "seed held at 42"),
        (3.0, bert, NAVY, "5-fold CV · BERT", "corrected loss"),
        (2.0, svm, MUTED, "5-fold CV · SVM", "TF-IDF + LinearSVC"),
    ]

    for y, vals, colour, label, sub in rows:
        ax.plot([vals.min(), vals.max()], [y, y], color=colour, lw=7,
                alpha=.18, solid_capstyle="round", zorder=1)
        ax.scatter(vals, np.full_like(vals, y), s=62, color=colour,
                   zorder=3, clip_on=False, edgecolor="white", linewidth=1.1)
        ax.scatter([vals.mean()], [y], marker="D", s=52, color=colour,
                   zorder=4, clip_on=False, edgecolor="white", linewidth=1.2)
        ax.text(0.5775, y + .19, label, ha="left", va="bottom",
                fontsize=11.5, color=BODY, fontweight="bold")
        ax.text(0.5775, y - .30, sub, ha="left", va="bottom", fontsize=9.4, color=MUTED)

    # 95% interval on BERT's cross-validated mean: the interval that excludes the
    # target. Drawn just above its row so it reads as an annotation on that row
    # rather than as another set of measured points.
    ci_y = 3.42
    ax.plot([bert_lo, bert_hi], [ci_y, ci_y], color=NAVY, lw=1.5, zorder=2)
    for xb in (bert_lo, bert_hi):
        ax.plot([xb, xb], [ci_y - .10, ci_y + .10], color=NAVY, lw=1.5, zorder=2)
    ax.text((bert_lo + bert_hi) / 2, ci_y + .16,
            "95% CI  [0.645, 0.729]   —  excludes the target",
            ha="center", va="bottom", fontsize=9.4, color=NAVY)

    # Reference lines
    ax.axvline(TARGET, color=RED, ls="--", lw=1.7, zorder=2)
    ax.text(TARGET + .0022, 4.80, "TARGET  0.75", color=RED, fontsize=10.6,
            fontweight="bold", ha="left", va="center")
    ax.axvline(DELIVERED, color=MUTED, ls=":", lw=1.5, zorder=2)
    ax.text(DELIVERED - .0022, 4.80, "delivered  0.6754", color=MUTED, fontsize=9.6,
            ha="right", va="center")

    # The punchline: spread and gap are distances on this same axis, so draw them as such.
    def bracket(y, x0, x1, colour, text):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=1.9,
                                    shrinkA=0, shrinkB=0))
        ax.plot([x0, x0], [y - .09, y + .09], color=colour, lw=1.9)
        ax.plot([x1, x1], [y - .09, y + .09], color=colour, lw=1.9)
        ax.text((x0 + x1) / 2, y - .34, text, ha="center", va="top",
                fontsize=10.6, color=colour, fontweight="bold")

    bracket(1.06, repeats.min(), repeats.max(), TEAL,
            "run-to-run spread  0.0521")
    bracket(0.28, bert.mean(), TARGET, RED,
            "gap to target  0.063")

    ax.set_xlim(.5775, .7735)
    ax.set_ylim(-.42, 5.10)
    ax.set_yticks([])
    ax.set_xticks(np.arange(.60, .7601, .025))
    ax.set_xticklabels([f"{v:.3f}" for v in np.arange(.60, .7601, .025)],
                       fontsize=9.6, color=MUTED)
    ax.set_xlabel("test macro F1", fontsize=10.4, color=BODY, labelpad=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D6DCE6")
    ax.tick_params(axis="x", length=3, color="#D6DCE6")
    ax.grid(axis="x", color=PANEL, lw=1, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED,
                              markeredgecolor="white", label="individual run / fold"),
                       Line2D([], [], marker="D", ls="", color=MUTED,
                              markeredgecolor="white", label="mean")],
              loc="upper right", frameon=False, fontsize=9.2, ncol=2,
              handletextpad=.3, columnspacing=1.4,
              # below the axis entirely: inside the axes it collided with the
              # tick labels around the target line
              bbox_to_anchor=(1.0, -0.105))

    fig.tight_layout(pad=.7)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"repeat runs   {repeats.min():.4f}–{repeats.max():.4f}  spread {np.ptp(repeats):.4f}")
    print(f"gap to target {TARGET - bert.mean():.4f}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
