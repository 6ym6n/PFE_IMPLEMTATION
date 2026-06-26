"""Generate the shared chart PNGs for the Strategy-D deck + report.

``make_charts(outdir)`` writes three figures used by both ``make_pdf.py`` and
``make_pptx.py``:
    chart_results.png     grouped bars: our methods vs published SOTA, per city
    chart_scale.png       the comparability scale (NYC vs Flickr literature)
    chart_validation.png  ours vs Chen 2016 scatter (the faithfulness check)
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flickr_results_data import (  # noqa: E402
    CHEN_PAIRS_F1,
    CITIES,
    FLICKR_OURS_RANGE,
    NYC_RANGE,
    OURS_PAIRS_F1,
    POINTER_PAIRS_F1,
    PUB_CITIES,
    PUBLISHED_PAIRS_F1,
)

NAVY = "#1B2A4E"
INDIGO = "#3F51B5"
TEAL = "#0E7C7B"
ORANGE = "#E07A1F"
GREY = "#888888"
RED = "#C0392B"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#444444",
    "figure.dpi": 150,
})


def _results_chart(path: str) -> None:
    """Grouped bars: Random / Markov / Pointer (ours) vs AR-Trip SOTA, per city."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    methods = [
        ("Random (ours)", OURS_PAIRS_F1["Random"], GREY),
        ("Markov (ours)", OURS_PAIRS_F1["Markov"], TEAL),
        ("Pointer (ours)", POINTER_PAIRS_F1["Pointer (beam)"], ORANGE),
        ("AR-Trip (SOTA, pub.)", {c: PUBLISHED_PAIRS_F1["AR-Trip"][0][i] for i, c in enumerate(PUB_CITIES)}, NAVY),
    ]
    n = len(PUB_CITIES)
    w = 0.2
    for k, (label, data, color) in enumerate(methods):
        xs = [i + (k - 1.5) * w for i in range(n)]
        ys = [data[c] for c in PUB_CITIES]
        bars = ax.bar(xs, ys, width=w, label=label, color=color, edgecolor="white", linewidth=0.6)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333333")
    ax.set_xticks(range(n))
    ax.set_xticklabels(PUB_CITIES)
    ax.set_ylabel("pairs-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Itinerary pairs-F1: our methods vs published SOTA (Flickr)")
    ax.axhspan(NYC_RANGE[0], NYC_RANGE[1], color=RED, alpha=0.10)
    ax.text(n - 0.5, (NYC_RANGE[0] + NYC_RANGE[1]) / 2, " our NYC band\n (0.26–0.29)",
            va="center", ha="left", fontsize=8, color=RED)
    ax.legend(loc="upper left", ncol=2, fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _scale_chart(path: str) -> None:
    """The comparability scale: 3 rows (Flickr literature / ours-Flickr / ours-NYC)."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    h = 0.45
    # row 2: Flickr literature band
    ax.barh(2, 0.85 - 0.26, left=0.26, height=h, color=TEAL, alpha=0.30, edgecolor=TEAL)
    ax.text(0.855, 2, "  pub. methods", va="center", ha="left", fontsize=9, color=TEAL)
    # row 1: ours on Flickr — classical span + pointer span
    ax.barh(1, FLICKR_OURS_RANGE[1] - FLICKR_OURS_RANGE[0], left=FLICKR_OURS_RANGE[0],
            height=h, color=NAVY, alpha=0.55, edgecolor=NAVY)
    ax.barh(1, 0.49 - 0.31, left=0.31, height=h * 0.5, color=ORANGE, edgecolor="white")
    ax.text(0.595, 1, "  classical 0.23–0.59", va="center", ha="left", fontsize=8.5, color=NAVY)
    ax.text(0.40, 1, "pointer", va="center", ha="center", fontsize=8, color="white", fontweight="bold")
    # row 0: NYC band
    ax.barh(0, NYC_RANGE[1] - NYC_RANGE[0], left=NYC_RANGE[0], height=h, color=RED, alpha=0.45, edgecolor=RED)
    ax.text(0.30, 0, "  (0.26–0.29)", va="center", ha="left", fontsize=9, color=RED)

    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Flickr literature\n(same task)", "OURS — Flickr\n(Strategy D)", "OURS — Foursquare NYC\n(not comparable)"],
                       fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("pairs-F1")
    ax.set_title("Same task, different benchmark → different scale")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _validation_chart(path: str) -> None:
    """Scatter: ours vs Chen 2016 pairs-F1 (points near y=x = faithful reproduction)."""
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.plot([0.2, 0.55], [0.2, 0.55], "--", color=GREY, label="y = x (perfect match)")
    for method, color, marker in [("Random", GREY, "o"), ("PoiPopularity", TEAL, "s")]:
        xs = [CHEN_PAIRS_F1[method][c] for c in CITIES]
        ys = [OURS_PAIRS_F1[method][c] for c in CITIES]
        ax.scatter(xs, ys, color=color, marker=marker, s=70, label=method, zorder=5, edgecolor="white")
    ax.set_xlabel("Chen 2016 published pairs-F1")
    ax.set_ylabel("our pairs-F1")
    ax.set_title("Faithfulness check: we reproduce Chen 2016")
    ax.set_xlim(0.2, 0.56)
    ax.set_ylim(0.2, 0.56)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_charts(outdir: str) -> dict:
    """Generate all charts into ``outdir``; return ``{name: path}``."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "results": os.path.join(outdir, "chart_results.png"),
        "scale": os.path.join(outdir, "chart_scale.png"),
        "validation": os.path.join(outdir, "chart_validation.png"),
    }
    _results_chart(paths["results"])
    _scale_chart(paths["scale"])
    _validation_chart(paths["validation"])
    return paths


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = make_charts(os.path.join(here, "assets"))
    for k, v in out.items():
        print("wrote", v)
