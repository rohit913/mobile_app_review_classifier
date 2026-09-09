"""Redraws the single-review LIME explanation, writing reports/lime_attribution_sample.png.

Notebook 06 saves LIME's own HTML widget to reports/lime_explanation_sample.html and a
screenshot of its default plot to reports/lime_explanation_sample.png. That default plot
is fine for checking a result in a notebook but reads poorly at figure size: pure red/green
on white, no value labels, no indication of which class is being explained and no
predicted probability, next to figures drawn in the project's own palette.

This script reads the exact attributions out of the saved HTML, which is the same
artefact the notebook wrote, and redraws them. No value is recomputed and no model is
loaded, so nothing here can drift from what notebook 06 produced: the numbers are parsed
from a committed file, and running this twice gives the same image. That matters for this
project in particular, where re-running things to regenerate a figure is exactly the
failure mode this project documents.

The review is the same Bug Report case notebook 06 explains with SHAP, so the two
outputs show one review under both methods.

Run from the repository root:

    python lime_figure.py
"""
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("reports/lime_explanation_sample.html")
OUT = Path("reports/lime_attribution_sample.png")

# Same palette as run_spread_figure.py, so the two read as one set.
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


def parse_lime_html(path: Path):
    """Pull the attributions, review text, class names and probabilities out of LIME's HTML.

    LIME's save_to_file writes literal JavaScript calls rather than a data block, so the
    values are read from those calls: exp.show_raw_text carries the word/offset/weight
    triples and the review string, PredictProba carries the class names and probabilities.
    Both are JSON-compatible array literals, so they parse without evaluating any script.
    """
    html = path.read_text(encoding="utf8", errors="replace")

    m = re.search(r"exp\.show_raw_text\(\s*(\[.*?\])\s*,\s*(\d+)\s*,\s*(\".*?\")", html, re.S)
    if not m:
        raise SystemExit(f"{path}: could not find LIME's show_raw_text payload")
    triples = json.loads(m.group(1))
    label_idx = int(m.group(2))
    review = ast.literal_eval(m.group(3))

    p = re.search(r"PredictProba\(\s*\w+\s*,\s*(\[.*?\])\s*,\s*(\[.*?\])\s*\)", html, re.S)
    if not p:
        raise SystemExit(f"{path}: could not find LIME's PredictProba payload")
    class_names = json.loads(p.group(1))
    probs = json.loads(p.group(2))

    words = [(w, float(v)) for w, _offset, v in triples]
    return words, class_names[label_idx], probs[label_idx], review


def main():
    words, cls, prob, review = parse_lime_html(SRC)
    words = sorted(words, key=lambda x: abs(x[1]))          # smallest at the bottom

    fig, ax = plt.subplots(figsize=(9.6, 3.75), dpi=200)

    labels = [w for w, _ in words]
    values = [v for _, v in words]
    ys = range(len(values))

    ax.barh(list(ys), values,
            color=[TEAL if v > 0 else RED for v in values],
            height=.58, zorder=3)

    span = max(abs(v) for v in values)
    pad = span * .06
    for y, v in zip(ys, values):
        ax.text(v + (pad if v > 0 else -pad), y, f"{v:+.3f}",
                va="center", ha="left" if v > 0 else "right",
                fontsize=10.2, color=BODY, zorder=4)

    ax.axvline(0, color=NAVY, lw=1.1, zorder=2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=12, color=BODY)
    ax.set_xlim(-span * 1.5, span * 1.5)
    ax.set_ylim(-.72, len(values) - .28)

    ax.set_xlabel("LIME weight  ·  contribution to the predicted class",
                  fontsize=10, color=MUTED, labelpad=9)
    ax.tick_params(axis="x", labelsize=9.4, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=.7, alpha=.6, zorder=0)
    ax.set_axisbelow(True)

    ax.text(0, 1.16, f"“{review}”",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.4, color=BODY, fontweight="bold")
    ax.text(0, 1.045,
            f"predicted {cls}, {prob:.1%} confidence   ·   "
            f"teal pushes toward {cls}, red pushes away",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=MUTED)

    fig.tight_layout()
    fig.subplots_adjust(top=.80)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"  review    : {review}")
    print(f"  predicted : {cls} ({prob:.4f})")
    for w, v in sorted(words, key=lambda x: -abs(x[1])):
        print(f"  {w:>10s} {v:+.6f}")


if __name__ == "__main__":
    main()
