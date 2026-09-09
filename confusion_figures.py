"""Redraws the two BERT confusion matrices and the LLaMA-3 comparison in the project palette.

The originals were seaborn defaults written by notebook 04 (the two BERT confusion
matrices) and notebook 05 (the LLaMA-3 pair). They are legible but sit awkwardly beside
the other figures in this project, which are drawn in the palette below.

Nothing here is recomputed. No model is loaded and no prediction is made: the counts are
transcribed from the figures those notebooks already wrote, and the script then *checks*
them against the classification reports preserved in the same notebooks' saved output and
against reports/baseline_results.json. If a transcribed count were wrong, the assertions
below would fail rather than a wrong figure being drawn. That check is the point of doing
it this way: re-running the models to redraw a figure is exactly the habit this project
argues against, and for the LLaMA-3 rows it is impossible anyway, the model having been
withdrawn from Groq.

    reports/cm_category.png    BERT category, 4x4 confusion matrix
    reports/cm_sentiment.png   BERT sentiment, 3x3 confusion matrix
    reports/llm_per_class.png  LLaMA-3 zero- against few-shot, per-class F1

The LLaMA-3 output is not a confusion matrix. The pair in reports/llm_confusion_matrices.png cannot
be reproduced from any surviving artefact and does not agree with the
delivered LLaMA-3 scores: its zero-shot diagonal sums to 65 of 100, where notebook 05's own
output and reports/baseline_results.json both give accuracy 0.69. The per-class scores in that
notebook's output are the delivered numbers, so those are what is plotted.
See the notes beside the LLM block below.

Run from the repository root:

    python confusion_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#1E2A44"
BODY = "#222A38"
MUTED = "#6C7483"
TEAL = "#0E7C86"
RED = "#B23A48"
RULE = "#D8DEE8"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RAMP = LinearSegmentedColormap.from_list("house", ["#FFFFFF", "#DCE6F0", "#7FA3C7", NAVY])

CATEGORIES = ["Bug Report", "Feature Request", "UX Feedback", "Positive Praise"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]

# Counts transcribed from the seaborn figures notebook 04 wrote. Validated below.
CM_CATEGORY = np.array([
    [84,  5,  7,  16],
    [ 5, 16,  3,   3],
    [17,  7, 34,   8],
    [15,  0,  6, 104],
])
CM_SENTIMENT = np.array([
    [130,  7,  14],
    [  6,  8,  11],
    [ 18,  7, 129],
])

# Per-class scores exactly as notebook 05 printed them: the delivered LLaMA-3 figures.
LLM = {
    "Zero-shot":  {"f1": [0.69, 0.45, 0.45, 0.89], "acc": 0.6900, "macro_f1": 0.6210},
    "Few-shot":   {"f1": [0.68, 0.42, 0.29, 0.89], "acc": 0.6400, "macro_f1": 0.5680},
}


def check(cm, expect_acc, expect_macro_f1, names, what):
    """Fail loudly if a transcribed matrix disagrees with the committed metrics."""
    total = cm.sum()
    acc = np.trace(cm) / total
    f1s = []
    for i in range(len(cm)):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro = float(np.mean(f1s))
    assert abs(acc - expect_acc) < 5e-4, f"{what}: accuracy {acc:.4f} != {expect_acc:.4f}"
    assert abs(macro - expect_macro_f1) < 5e-4, f"{what}: macro F1 {macro:.4f} != {expect_macro_f1:.4f}"
    print(f"  {what}: n={total}, accuracy {acc:.4f}, macro F1 {macro:.4f}  (matches baseline_results.json)")
    return dict(zip(names, f1s))


def draw_matrix(cm, names, out, title, subtitle, figsize):
    """Counts annotated; colour is row-normalised, so the diagonal reads as recall."""
    rows = cm.sum(axis=1, keepdims=True)
    shade = cm / rows

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.imshow(shade, cmap=RAMP, vmin=0, vmax=1, aspect="auto")

    n = len(names)
    for i in range(n):
        for j in range(n):
            pct = shade[i, j]
            ink = "white" if pct > 0.55 else BODY
            ax.text(j, i - .09, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=15.5 if i == j else 14, color=ink,
                    fontweight="bold" if i == j else "normal")
            ax.text(j, i + .19, f"{pct:.0%}", ha="center", va="center",
                    fontsize=9.6, color=ink, alpha=.72)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=11, color=BODY)
    ax.set_yticklabels(names, fontsize=11, color=BODY)
    ax.set_xlabel("Predicted", fontsize=10.5, color=MUTED, labelpad=10)
    ax.set_ylabel("Actual", fontsize=10.5, color=MUTED, labelpad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.4)
    ax.tick_params(which="minor", length=0)

    ax.text(0, 1.13, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, color=BODY, fontweight="bold")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=MUTED)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


def draw_llm(out):
    """Per-class F1, zero-shot against few-shot. Not a confusion matrix: see the docstring."""
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=200)
    y = np.arange(len(CATEGORIES))[::-1]
    h = .34

    for offset, (name, colour) in zip((h / 2, -h / 2), (("Zero-shot", TEAL), ("Few-shot", RED))):
        vals = LLM[name]["f1"]
        ax.barh(y + offset, vals, height=h, color=colour, zorder=3)
        for yy, v in zip(y + offset, vals):
            ax.text(v + .012, yy, f"{v:.2f}", va="center", ha="left",
                    fontsize=10, color=BODY)

    # the class where the two diverge, which is the substantive finding here.
    # Placed clear of both value labels rather than between the bar ends.
    drop = LLM["Zero-shot"]["f1"][2] - LLM["Few-shot"]["f1"][2]
    ax.text(.60, y[2], f"−{drop:.2f}", fontsize=11, color=RED,
            ha="left", va="center", fontweight="bold", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(CATEGORIES, fontsize=11.5, color=BODY)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 on the 100-review stratified test sample", fontsize=10, color=MUTED, labelpad=9)
    ax.tick_params(axis="x", labelsize=9.4, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=.7, alpha=.6, zorder=0)
    ax.set_axisbelow(True)

    ax.text(0, 1.35, "LLaMA-3 per-class F1, zero-shot against few-shot",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, color=BODY, fontweight="bold")
    ax.text(0, 1.24,
            "few-shot loses almost all of its ground on UX Feedback; the other three classes barely move",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=MUTED)

    # Key drawn into the header, so it cannot collide with the Positive Praise bars.
    for x, (name, colour) in zip((0.0, 0.42),
                                 (("Zero-shot", TEAL), ("Few-shot", RED))):
        ax.scatter([x + .008], [1.10], s=64, color=colour, marker="s",
                   transform=ax.transAxes, clip_on=False, zorder=5)
        ax.text(x + .028, 1.10, f"{name}", transform=ax.transAxes,
                ha="left", va="center", fontsize=10, color=BODY, fontweight="bold")
        ax.text(x + .125, 1.10,
                f"acc {LLM[name]['acc']:.2f}  ·  macro F1 {LLM[name]['macro_f1']:.4f}",
                transform=ax.transAxes, ha="left", va="center", fontsize=9.8, color=MUTED)

    fig.tight_layout()
    fig.subplots_adjust(top=.72)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


def main():
    base = json.loads(Path("reports/baseline_results.json").read_text())

    print("validating transcribed counts against reports/baseline_results.json")
    cat_f1 = check(CM_CATEGORY, base["bert_category_accuracy"], base["bert_category_macro_f1"],
                   CATEGORIES, "BERT category")
    sen_f1 = check(CM_SENTIMENT, base["bert_sentiment_accuracy"], base["bert_sentiment_macro_f1"],
                   SENTIMENTS, "BERT sentiment")

    # the delivered LLaMA-3 macro F1 must equal the macro mean of the per-class F1 plotted
    for name, key in (("Zero-shot", "llama_zero_shot_macro_f1"), ("Few-shot", "llama_few_shot_macro_f1")):
        assert abs(LLM[name]["macro_f1"] - base[key]) < 5e-4, name
    print(f"  LLaMA-3 macro F1 {LLM['Zero-shot']['macro_f1']} / {LLM['Few-shot']['macro_f1']}"
          f"  (matches baseline_results.json)")

    print("\ndrawing")
    draw_matrix(CM_CATEGORY, CATEGORIES, Path("reports/cm_category.png"),
                "BERT category classification",
                f"test set, n = {CM_CATEGORY.sum()}  ·  accuracy 0.7212  ·  macro F1 0.6754  ·  "
                "shading is row-normalised, so the diagonal reads as recall",
                (8.4, 5.4))
    draw_matrix(CM_SENTIMENT, SENTIMENTS, Path("reports/cm_sentiment.png"),
                "BERT sentiment classification",
                f"test set, n = {CM_SENTIMENT.sum()}  ·  accuracy 0.8091  ·  macro F1 0.6768  ·  "
                "shading is row-normalised, so the diagonal reads as recall",
                (7.4, 4.3))
    draw_llm(Path("reports/llm_per_class.png"))

    print("\nper-class F1 recovered from the matrices")
    for k, v in {**cat_f1, **sen_f1}.items():
        print(f"  {k:16s} {v:.4f}")


if __name__ == "__main__":
    main()
