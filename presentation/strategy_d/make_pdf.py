"""Detailed companion document for Strategy D (Flickr itinerary recommendation).

Reuses the thesis report's reportlab styling (``report/make_report.py``) for a
consistent look, adds Strategy-D-specific vector diagrams, and embeds the shared
matplotlib charts. Run:

    py -3.11 presentation/strategy_d/make_pdf.py

Output: presentation/strategy_d/strategy_d_explained.pdf
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "report"))
sys.path.insert(0, HERE)

import make_report as mr  # noqa: E402  (reuse styles/helpers/diagram base)
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, TableStyle,
)

import charts  # noqa: E402
from flickr_results_data import (  # noqa: E402
    CHEN_PAIRS_F1, CITIES, DATASET_STATS, OURS_F1, OURS_PAIRS_F1, PAIRS_EXAMPLE,
    POINTER_F1, POINTER_PAIRS_F1, PUB_CITIES, PUBLISHED_PAIRS_F1,
)

S = mr.STYLES
NAVY, INDIGO, TEAL, ORANGE = mr.NAVY, mr.INDIGO, mr.TEAL, mr.ORANGE
GREY_DARK, GREY_MED, GREY_LIGHT, WHITE = mr.GREY_DARK, mr.GREY_MED, mr.GREY_LIGHT, mr.WHITE
BG_G, BG_C, BG_U, BG_S, BG_H = (mr.BG_BOX_GRAPH, mr.BG_BOX_CTX, mr.BG_BOX_USER,
                                mr.BG_BOX_SEQ, mr.BG_BOX_HEAD)

OUTPUT = os.path.join(HERE, "strategy_d_explained.pdf")
ASSETS = charts.make_charts(os.path.join(HERE, "assets"))


def body(t):
    return mr.body(t)


def bullet(t):
    return mr.bullet(t)


def section(t, lvl=1):
    return mr.section(t, lvl)


def img(path, width_cm):
    """An Image flowable scaled to ``width_cm`` keeping aspect ratio."""
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width_cm * cm
    return Image(path, width=w, height=w * ih / iw)


# ===========================================================================
# Custom vector diagrams (subclass the report's Diagram base)
# ===========================================================================
class PhaseMap(mr.Diagram):
    """Phase 1 -> Phase 2A/B (NYC) -> Phase 2D (Flickr, this work)."""

    def __init__(self):
        super().__init__(16.5 * cm, 5.2 * cm)

    def draw(self):
        c = self.canv
        bw, bh = 4.9 * cm, 2.5 * cm
        y = 1.4 * cm
        boxes = [
            ("Phase 1\nNext-POI prediction", "Foursquare NYC\nGCN + GRU + user\nHR@k / NDCG / MRR",
             0.2 * cm, BG_G, INDIGO),
            ("Phase 2A / 2B\nItinerary (NYC)", "frozen rollout / pointer\npairs-F1 ≈ 0.26–0.29\n(NOT comparable)",
             5.75 * cm, BG_H, ORANGE),
            ("Phase 2D — THIS WORK\nItinerary (Flickr)", "Chen-2016 protocol\npairs-F1 0.23–0.59\nliterature-comparable",
             11.3 * cm, BG_S, TEAL),
        ]
        for title, sub, x, fill, stroke in boxes:
            self._box(c, x, y, bw, bh, fill, stroke=stroke, line_width=1.6)
            self._label(c, title.split("\n")[0], x + bw / 2, y + bh - 0.55 * cm, size=10.5, color=stroke)
            self._label(c, title.split("\n")[1], x + bw / 2, y + bh - 1.05 * cm, size=10.5, color=stroke)
            for i, ln in enumerate(sub.split("\n")):
                self._label(c, ln, x + bw / 2, y + bh - 1.5 * cm - i * 0.40 * cm, size=8.5,
                            font="Helvetica", color=GREY_DARK)
        for i in range(2):
            x1 = 0.2 * cm + bw + i * 5.55 * cm
            self._arrow(c, x1, y + bh / 2, x1 + 0.65 * cm, y + bh / 2, color=NAVY, line_width=1.6)


class TrajBuild(mr.Diagram):
    """Geo-tagged photos -> POI visits -> ordered, dedup'd trajectory."""

    def __init__(self):
        super().__init__(16.5 * cm, 4.6 * cm)

    def draw(self):
        c = self.canv
        y = 1.5 * cm
        steps = [
            ("Flickr photos", "(lat, lon, time)\nper user", 0.2 * cm, BG_G),
            ("POI visits", "snap to nearest POI;\nmerge same-POI run", 4.5 * cm, BG_C),
            ("order by time", "within one seqID\n= one trajectory", 8.8 * cm, BG_U),
            ("trajectory", "p1 → p2 → p3 → p4\n(distinct, loop-free)", 13.1 * cm, BG_S),
        ]
        bw, bh = 3.2 * cm, 1.8 * cm
        for title, sub, x, fill in steps:
            self._box(c, x, y, bw, bh, fill, stroke=NAVY)
            self._label(c, title, x + bw / 2, y + bh - 0.5 * cm, size=10, color=NAVY)
            for i, ln in enumerate(sub.split("\n")):
                self._label(c, ln, x + bw / 2, y + 0.7 * cm - i * 0.4 * cm, size=8, font="Helvetica",
                            color=GREY_DARK)
        for i in range(3):
            x1 = 0.2 * cm + bw + i * 4.3 * cm
            self._arrow(c, x1, y + bh / 2, x1 + 1.1 * cm, y + bh / 2, color=NAVY)


class QueryRoute(mr.Diagram):
    """Query (start, end, K) -> recommend the ordered middle."""

    def __init__(self):
        super().__init__(16.5 * cm, 4.4 * cm)

    def draw(self):
        c = self.canv
        # query box
        self._box(c, 0.2 * cm, 1.4 * cm, 5.2 * cm, 1.9 * cm, BG_U, stroke=INDIGO)
        self._label(c, "QUERY", 2.8 * cm, 1.4 * cm + 1.45 * cm, size=10, color=INDIGO)
        self._label(c, "start s, end e, length K", 2.8 * cm, 1.4 * cm + 0.95 * cm, size=9.5,
                    font="Helvetica", color=GREY_DARK)
        self._label(c, "e.g. (Castle, Museum, 5)", 2.8 * cm, 1.4 * cm + 0.45 * cm, size=8.5,
                    font="Helvetica-Oblique", color=GREY_MED)
        self._arrow(c, 5.4 * cm, 2.35 * cm, 6.3 * cm, 2.35 * cm, color=NAVY, line_width=1.6)
        # model
        self._box(c, 6.3 * cm, 1.6 * cm, 3.0 * cm, 1.5 * cm, BG_S, stroke=TEAL)
        self._label(c, "recommender", 6.3 * cm + 1.5 * cm, 2.35 * cm - 2, size=10, color=TEAL)
        self._arrow(c, 9.3 * cm, 2.35 * cm, 10.2 * cm, 2.35 * cm, color=NAVY, line_width=1.6)
        # route: s ? ? ? e
        labels = ["s", "?", "?", "?", "e"]
        fills = [BG_U, WHITE, WHITE, WHITE, BG_U]
        for i, (lab, fl) in enumerate(zip(labels, fills)):
            x = 10.3 * cm + i * 1.2 * cm
            self._box(c, x, 1.8 * cm, 1.0 * cm, 1.1 * cm, fl, stroke=ORANGE, radius=4)
            self._label(c, lab, x + 0.5 * cm, 2.35 * cm - 3, size=11, color=ORANGE)
        self._label(c, "ordered route of K POIs (loop-free, ends at e)",
                    13.0 * cm, 1.4 * cm, size=8.5, font="Helvetica-Oblique", color=GREY_MED)


class PairsF1(mr.Diagram):
    """Worked pairs-F1: truth [A,B,C,D] vs pred [A,C,B,D] -> 5/6."""

    def __init__(self):
        super().__init__(16.5 * cm, 7.6 * cm)

    def draw(self):
        c = self.canv
        ex = PAIRS_EXAMPLE

        def seq_row(label, seq, y, color):
            self._label(c, label, 0.2 * cm, y, size=10, color=color, anchor="left", font="Helvetica-Bold")
            for i, p in enumerate(seq):
                x = 3.6 * cm + i * 1.2 * cm
                self._box(c, x, y - 0.3 * cm, 0.95 * cm, 0.8 * cm, BG_S if color == TEAL else BG_H,
                          stroke=color, radius=4)
                self._label(c, p, x + 0.475 * cm, y - 0.05 * cm, size=11, color=color)

        seq_row("ground truth Y:", ex["truth"], 7.0 * cm, TEAL)
        seq_row("prediction Ŷ:", ex["pred"], 5.9 * cm, ORANGE)

        self._label(c, "ordered pairs (i before j):", 0.2 * cm, 4.9 * cm, size=9.5, color=NAVY,
                    anchor="left", font="Helvetica-Bold")
        self._label(c, "Y:  " + "  ".join(ex["truth_pairs"]), 0.4 * cm, 4.35 * cm, size=9.5,
                    color=TEAL, anchor="left", font="Helvetica")
        self._label(c, "Ŷ:  " + "  ".join(ex["pred_pairs"]), 0.4 * cm, 3.8 * cm, size=9.5,
                    color=ORANGE, anchor="left", font="Helvetica")
        self._label(c, "shared, SAME order:  " + "  ".join(ex["shared"]) + "   →  5 of 6",
                    0.4 * cm, 3.05 * cm, size=9.5, color=NAVY, anchor="left", font="Helvetica-Bold")
        self._label(c, "(B,C) is in Y but Ŷ has (C,B) — wrong order, so it does NOT count.",
                    0.4 * cm, 2.5 * cm, size=8.5, color=GREY_MED, anchor="left", font="Helvetica-Oblique")
        # computation
        self._box(c, 0.4 * cm, 0.5 * cm, 15.6 * cm, 1.5 * cm, GREY_LIGHT, stroke=NAVY, radius=4)
        self._label(c, "precision = 5/6,   recall = 5/6   →   pairs-F1 = 2·P·R/(P+R) = 5/6 ≈ 0.833",
                    8.2 * cm, 1.25 * cm, size=11, color=NAVY)


class LOOCV(mr.Diagram):
    """Leave-one-trajectory-out cross-validation."""

    def __init__(self):
        super().__init__(16.5 * cm, 6.2 * cm)

    def draw(self):
        c = self.canv
        # column of trajectories
        n = 6
        for i in range(n):
            y = 5.2 * cm - i * 0.78 * cm
            is_test = i == 2
            self._box(c, 0.3 * cm, y, 3.3 * cm, 0.62 * cm, BG_H if is_test else BG_S,
                      stroke=ORANGE if is_test else TEAL, radius=4)
            txt = "t₃  ← held out (TEST)" if is_test else f"t{['₁','₂','₃','₄','₅','₆'][i]}"
            self._label(c, txt, 0.5 * cm, y + 0.18 * cm, size=9,
                        color=ORANGE if is_test else TEAL, anchor="left",
                        font="Helvetica-Bold" if is_test else "Helvetica")
        self._label(c, "… all N trajectories", 0.5 * cm, 5.2 * cm - n * 0.78 * cm + 0.1 * cm,
                    size=8, font="Helvetica-Oblique", color=GREY_MED, anchor="left")
        # arrows to train/predict
        self._arrow(c, 3.7 * cm, 3.0 * cm, 5.0 * cm, 3.0 * cm, color=NAVY)
        self._box(c, 5.0 * cm, 2.3 * cm, 4.2 * cm, 1.4 * cm, BG_S, stroke=TEAL)
        self._label(c, "train on the other N−1", 5.0 * cm + 2.1 * cm, 3.25 * cm - 2, size=9.5, color=TEAL)
        self._label(c, "(length≥3 trajectories)", 5.0 * cm + 2.1 * cm, 2.75 * cm - 2, size=8,
                    font="Helvetica", color=GREY_DARK)
        self._arrow(c, 9.2 * cm, 3.0 * cm, 10.5 * cm, 3.0 * cm, color=NAVY)
        self._box(c, 10.5 * cm, 2.3 * cm, 5.7 * cm, 1.4 * cm, BG_H, stroke=ORANGE)
        self._label(c, "predict t₃ from (s, e, K);", 10.5 * cm + 2.85 * cm, 3.25 * cm - 2, size=9.5,
                    color=ORANGE)
        self._label(c, "score F1 & pairs-F1 vs t₃", 10.5 * cm + 2.85 * cm, 2.75 * cm - 2, size=9.5,
                    color=ORANGE)
        self._label(c, "Repeat holding out every trajectory once  →  report the mean per city.",
                    8.2 * cm, 1.1 * cm, size=10, color=NAVY)


class MethodArch(mr.Diagram):
    """Classical baselines (top) + GCN-pointer architecture (bottom)."""

    def __init__(self):
        super().__init__(16.5 * cm, 9.4 * cm)

    def draw(self):
        c = self.canv
        # ---- classical (top band) ----
        self._label(c, "Classical baselines (counting, CPU)", 0.2 * cm, 8.9 * cm, size=10.5,
                    color=NAVY, anchor="left", font="Helvetica-Bold")
        self._box(c, 0.3 * cm, 7.0 * cm, 3.4 * cm, 1.5 * cm, BG_S, stroke=TEAL)
        self._label(c, "training", 0.3 * cm + 1.7 * cm, 8.05 * cm, size=9.5, color=TEAL)
        self._label(c, "trajectories", 0.3 * cm + 1.7 * cm, 7.55 * cm, size=9.5, color=TEAL)
        self._label(c, "(fold)", 0.3 * cm + 1.7 * cm, 7.2 * cm, size=8, font="Helvetica", color=GREY_DARK)
        for i, (t1, t2) in enumerate([("popularity", "visit counts"),
                                      ("Markov P(j|i)", "transition counts")]):
            x = 4.6 * cm + i * 4.0 * cm
            self._box(c, x, 7.0 * cm, 3.6 * cm, 1.5 * cm, BG_C, stroke=ORANGE)
            self._label(c, t1, x + 1.8 * cm, 8.05 * cm, size=9.5, color=ORANGE)
            self._label(c, t2, x + 1.8 * cm, 7.5 * cm, size=8.5, font="Helvetica", color=GREY_DARK)
            self._arrow(c, x - 0.4 * cm, 7.75 * cm, x, 7.75 * cm, color=NAVY)
        self._box(c, 12.8 * cm, 7.0 * cm, 3.4 * cm, 1.5 * cm, BG_H, stroke=ORANGE)
        self._label(c, "greedy / beam", 12.8 * cm + 1.7 * cm, 8.05 * cm, size=9.5, color=ORANGE)
        self._label(c, "decode (loop-free,", 12.8 * cm + 1.7 * cm, 7.55 * cm, size=8.5,
                    font="Helvetica", color=GREY_DARK)
        self._label(c, "reserve end)", 12.8 * cm + 1.7 * cm, 7.2 * cm, size=8.5, font="Helvetica",
                    color=GREY_DARK)
        self._arrow(c, 12.4 * cm, 7.75 * cm, 12.8 * cm, 7.75 * cm, color=NAVY)

        # divider
        c.setStrokeColor(GREY_LIGHT)
        c.setLineWidth(0.8)
        c.line(0.2 * cm, 6.3 * cm, 16.3 * cm, 6.3 * cm)

        # ---- neural (bottom) ----
        self._label(c, "Learned model: GCN encoder + GRU pointer decoder (GPU)", 0.2 * cm, 5.8 * cm,
                    size=10.5, color=NAVY, anchor="left", font="Helvetica-Bold")
        # graph -> GCN -> H
        self._box(c, 0.3 * cm, 3.9 * cm, 3.0 * cm, 1.4 * cm, BG_G, stroke=INDIGO)
        self._label(c, "POI graph", 0.3 * cm + 1.5 * cm, 4.85 * cm, size=9.5, color=INDIGO)
        self._label(c, "kNN ∪ co-visit", 0.3 * cm + 1.5 * cm, 4.35 * cm, size=8, font="Helvetica",
                    color=GREY_DARK)
        self._arrow(c, 3.3 * cm, 4.6 * cm, 3.9 * cm, 4.6 * cm, color=NAVY)
        self._box(c, 3.9 * cm, 3.9 * cm, 2.7 * cm, 1.4 * cm, BG_G, stroke=INDIGO)
        self._label(c, "2-layer GCN", 3.9 * cm + 1.35 * cm, 4.6 * cm - 2, size=9.5, color=INDIGO)
        self._arrow(c, 6.6 * cm, 4.6 * cm, 7.2 * cm, 4.6 * cm, color=NAVY)
        self._box(c, 7.2 * cm, 3.9 * cm, 3.0 * cm, 1.4 * cm, BG_G, stroke=INDIGO)
        self._label(c, "POI features H", 7.2 * cm + 1.5 * cm, 4.85 * cm, size=9.5, color=INDIGO)
        self._label(c, "(|V| × d)", 7.2 * cm + 1.5 * cm, 4.35 * cm, size=8, font="Helvetica",
                    color=GREY_DARK)
        # query -> GRU -> pointer
        self._box(c, 0.3 * cm, 1.6 * cm, 3.0 * cm, 1.4 * cm, BG_U, stroke=INDIGO)
        self._label(c, "query (s, e)", 0.3 * cm + 1.5 * cm, 2.55 * cm, size=9.5, color=INDIGO)
        self._label(c, "seeds h₀", 0.3 * cm + 1.5 * cm, 2.05 * cm, size=8, font="Helvetica",
                    color=GREY_DARK)
        self._arrow(c, 3.3 * cm, 2.3 * cm, 3.9 * cm, 2.3 * cm, color=NAVY)
        self._box(c, 3.9 * cm, 1.6 * cm, 3.3 * cm, 1.4 * cm, BG_S, stroke=TEAL)
        self._label(c, "GRU decoder", 3.9 * cm + 1.65 * cm, 2.3 * cm - 2, size=9.5, color=TEAL)
        self._arrow(c, 7.2 * cm, 2.3 * cm, 7.8 * cm, 2.3 * cm, color=NAVY)
        self._box(c, 7.8 * cm, 1.6 * cm, 4.0 * cm, 1.4 * cm, BG_H, stroke=ORANGE)
        self._label(c, "pointer:  ⟨state, H_v⟩", 7.8 * cm + 2.0 * cm, 2.55 * cm, size=9.5, color=ORANGE)
        self._label(c, "mask visited, reserve e", 7.8 * cm + 2.0 * cm, 2.05 * cm, size=8,
                    font="Helvetica", color=GREY_DARK)
        self._arrow(c, 11.8 * cm, 2.3 * cm, 12.4 * cm, 2.3 * cm, color=NAVY)
        self._box(c, 12.4 * cm, 1.6 * cm, 3.8 * cm, 1.4 * cm, BG_H, stroke=ORANGE)
        self._label(c, "next POI", 12.4 * cm + 1.9 * cm, 2.55 * cm, size=9.5, color=ORANGE)
        self._label(c, "(+ Markov prior lever)", 12.4 * cm + 1.9 * cm, 2.05 * cm, size=8,
                    font="Helvetica", color=GREY_DARK)
        # H feeds pointer
        self._arrow(c, 8.7 * cm, 3.9 * cm, 9.8 * cm, 3.0 * cm, color=INDIGO, line_width=0.9)


# ===========================================================================
# Tables
# ===========================================================================
def _pf1_table(methods_dict, cities, headerlabel="pairs-F1"):
    head = [headerlabel] + cities
    rows = [head]
    for m, d in methods_dict.items():
        rows.append([m] + [f"{d[c]:.3f}" for c in cities])
    return rows


def stats_table_rows():
    rows = [["City", "#POIs", "#users", "#traj", "#traj≥3 (eval)"]]
    for c in CITIES:
        p, u, t, e = DATASET_STATS[c]
        rows.append([c, str(p), f"{u:,}", f"{t:,}", str(e)])
    return rows


# ===========================================================================
# Page furniture
# ===========================================================================
def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setStrokeColor(GREY_LIGHT)
        c.setLineWidth(0.6)
        c.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        c.setFont(mr._UNICODE_FONT_NAME, 8)
        c.setFillColor(GREY_MED)
        c.drawString(2 * cm, A4[1] - 1.1 * cm, "Strategy D — Itinerary recommendation on Flickr")
        c.drawRightString(A4[0] - 2 * cm, A4[1] - 1.1 * cm, "pairs-F1 comparable to the literature")
        c.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
        c.drawCentredString(A4[0] / 2, 1.0 * cm, f"— {doc.page} —")
    c.restoreState()


# ===========================================================================
# Story
# ===========================================================================
def build_story():
    s = []
    # cover
    s.append(Spacer(1, 3.5 * cm))
    s.append(Paragraph("Literature-Comparable Itinerary Recommendation", S["Title"]))
    s.append(Paragraph("Strategy D — the Flickr photo-trajectory benchmark", S["Subtitle"]))
    s.append(Spacer(1, 0.7 * cm))
    s.append(Paragraph("From non-comparable Foursquare-NYC scores to pairs-F1 on the "
                       "same scale as Chen 2016, DeepTrip, SelfTrip and AR-Trip", S["Subtitle"]))
    s.append(Spacer(1, 2.6 * cm))
    s.append(Paragraph("Detailed technical document · companion to the slide deck", S["Author"]))
    s.append(Paragraph("Repository: github.com/6ym6n/PFE_IMPLEMTATION", S["Author"]))
    s.append(PageBreak())

    # exec summary
    s.append(section("Executive summary", 1))
    s.append(body(
        "Phase 2 of this thesis built an <b>itinerary recommender</b> — given a start POI, "
        "an end POI and a desired length K, produce an <b>ordered route</b> of POIs — and "
        "evaluated it on Foursquare NYC, scoring <b>pairs-F1 ≈ 0.26–0.29</b>. The published "
        "trip-recommendation literature reports pairs-F1 ≈ 0.6–0.8, which made our numbers "
        "look weak. <b>They are not weak — they were simply not comparable.</b> The literature "
        "is evaluated on a different benchmark (the small <b>Flickr</b> photo-trajectory city "
        "datasets) under a specific protocol."
    ))
    s.append(body(
        "<b>Strategy D</b> re-runs the itinerary task on those exact datasets under the exact "
        "published protocol, so the numbers land on the <b>same scale</b> as the papers. "
        "Using the canonical trajectory files (Chen 2016's own preprocessing output) our "
        "trajectories are <i>identical</i> to the literature's, and a <b>Random baseline "
        "reproduces Chen 2016 within noise</b> — proving the pipeline is faithful. Our classical "
        "methods reach <b>pairs-F1 0.23–0.59</b> and the learned GCN+pointer model 0.31–0.49: "
        "both squarely on the published 0.26–0.85 scale. This document explains every piece — "
        "data, task, metric, protocol, methods, results and honest limitations."
    ))
    s.append(Spacer(1, 0.2 * cm))
    s.append(img(ASSETS["scale"], 15.5))
    s.append(mr.caption("Figure 1 — The whole point in one picture: the same itinerary task on a "
                        "different benchmark sits on a different scale. Strategy D puts us on the "
                        "literature's scale."))

    # 1. context
    s.append(section("1. Where Strategy D fits", 1))
    s.append(body(
        "The thesis addresses <b>two formally distinct sub-tasks</b>, each with its own "
        "field-standard benchmark and metrics (this separation is exactly what the tour-"
        "recommendation surveys prescribe): <b>next-POI prediction</b> on Foursquare/Gowalla "
        "check-ins (ranking metrics), and <b>itinerary / tour recommendation</b> on Flickr "
        "photo-trajectories (F1 and pairs-F1). Using a different dataset per task is correct and "
        "standard; the only error would be to cross-compare them."
    ))
    s.append(PhaseMap())
    s.append(mr.caption("Figure 2 — Phase 1 (next-POI) feeds Phase 2 (itinerary). Phase 2A/2B ran "
                        "on Foursquare NYC and produced an internal A-vs-B study; Phase 2D (this "
                        "work) runs on Flickr for literature comparability."))

    # 2. comparability
    s.append(section("2. The comparability problem", 1))
    s.append(body(
        "Two pairs-F1 numbers may be compared <b>only if three things match</b>: the "
        "<b>dataset</b> (same trajectories, same POI vocabulary size — a 30-POI city and a "
        "5,000-POI city have completely different chance levels), the <b>protocol</b> (same "
        "split, same query, same length filter), and the <b>metric</b> (the exact same pairs-F1 "
        "definition). Foursquare NYC differs from the Flickr benchmark on all three, so its "
        "0.26–0.29 is incomparable to the literature's 0.6–0.8 — the gap is a data/protocol "
        "artefact, not a model-quality gap (cf. Krichene &amp; Rendle, KDD 2020, on why "
        "cross-protocol metrics are inconsistent)."
    ))
    s.append(bullet("<b>Dataset:</b> Flickr cities have 27–88 POIs; NYC has ~5,000. Recovering the "
                    "order of 5 of 30 POIs is a very different problem from 5 of 5,000."))
    s.append(bullet("<b>Protocol:</b> the literature uses leave-one-trajectory-out CV with the "
                    "origin and destination given; our NYC study used a fixed test split."))
    s.append(bullet("<b>Metric scale:</b> both report pairs-F1, but on different vocabularies the "
                    "achievable range differs."))

    # 3. datasets
    s.append(section("3. The Flickr datasets", 1))
    s.append(body(
        "The benchmark is five city datasets mined from geo-tagged Flickr photos (Lim et al.; "
        "the standard data for classical and deep tour recommendation). We use the canonical "
        "<code>traj-{City}.csv</code> + <code>poi-{City}.csv</code> files — <b>Chen 2016's own "
        "preprocessing output</b>, mirrored in the <code>tour-cikm16</code> and DeepTrip repos — "
        "so our trajectories are <i>identical</i> to the published ones and there is zero "
        "preprocessing ambiguity. A <b>trajectory</b> is the sequence of POIs of one photo-trip "
        "(<code>trajID</code> / <code>seqID</code>), ordered by time, with consecutive duplicate POIs merged. Every "
        "trajectory in this data is <b>loop-free</b> (distinct POIs), which the pairs-F1 metric "
        "assumes."
    ))
    s.append(TrajBuild())
    s.append(mr.caption("Figure 3 — From raw geo-tagged photos to an ordered, loop-free POI "
                        "trajectory. The evaluated set keeps trajectories of length ≥ 3."))
    s.append(Spacer(1, 0.2 * cm))
    s.append(mr.make_table(stats_table_rows(),
                           col_widths=[4 * cm, 2.6 * cm, 2.8 * cm, 2.8 * cm, 4 * cm]))
    s.append(mr.caption("Table 1 — Dataset statistics. The right column (length ≥ 3) is the "
                        "evaluated set; Chen 2016's total-trajectory counts match the #traj column "
                        "exactly, confirming we have the same data."))

    # 4. task
    s.append(section("4. The task, precisely", 1))
    s.append(body(
        "A <b>query</b> is <i>q = (start POI s, end POI e, length K)</i>. The recommender must "
        "output an ordered, loop-free route of exactly K POIs that starts at s and ends at e, "
        "recovering the intermediate POIs (and their order) of the user's real trajectory. The "
        "ground truth is the held-out trajectory itself. This is the standard PersTour / Chen "
        "2016 convention: <i>recover the visited sequence given its endpoints and length.</i>"
    ))
    s.append(QueryRoute())
    s.append(mr.caption("Figure 4 — The query gives the endpoints and the length; the model fills "
                        "the ordered middle. Start and end are given, so a perfect recovery scores "
                        "1.0 and only the middle ordering is at stake."))

    # 5. metric
    s.append(section("5. The metric: F1 and pairs-F1", 1))
    s.append(body(
        "Two metrics, both in [0, 1]. <b>Point-F1</b> (set-F1) measures <i>did we pick the right "
        "POIs</i>, ignoring order: the F1 of the predicted POI set against the true set. "
        "<b>pairs-F1</b> (Chen 2016, the primary metric) measures <i>order</i>: it is the F1 over "
        "<b>ordered pairs</b> of POIs. For every pair (a, b), it counts as correct only if a "
        "appears before b in <i>both</i> the prediction and the ground truth. The harmonic mean "
        "of pair-precision and pair-recall is pairs-F1."
    ))
    s.append(PairsF1())
    s.append(mr.caption("Figure 5 — A worked pairs-F1. Swapping the two middle POIs breaks exactly "
                        "one ordered pair (B,C), so 5 of 6 pairs survive → pairs-F1 = 5/6 ≈ 0.833. "
                        "We reuse the thesis's unit-tested implementation, pinned against Chen "
                        "2016's reference calc_pairsF1."))

    # 6. protocol
    s.append(section("6. The protocol: leave-one-trajectory-out CV", 1))
    s.append(body(
        "These datasets are tiny (47–634 evaluated trajectories per city), so the field uses "
        "<b>leave-one-trajectory-out cross-validation</b>: hold out one trajectory for testing, "
        "train on all the others, predict the held-out one from its (s, e, K) query, score it, "
        "and repeat for every trajectory — then average. We train each fold on the other "
        "<b>length ≥ 3</b> trajectories (matching the papers' folds); the held-out trajectory "
        "never influences training, its graph, or its statistics (no leakage)."
    ))
    s.append(LOOCV())
    s.append(mr.caption("Figure 6 — Leave-one-out CV. Each trajectory is the test case exactly "
                        "once; the reported number is the mean over all folds."))

    # 7. methods
    s.append(section("7. The methods we built", 1))
    s.append(body(
        "Two families. <b>Classical baselines</b> (fast, CPU, faithful re-implementations of "
        "Chen 2016's): <b>Random</b> (random order of the middle POIs — the floor), "
        "<b>PoiPopularity</b> (pick the most-visited POIs, ordered by popularity), <b>Markov</b> "
        "(a first-order POI→POI transition model decoded greedily) and <b>MarkovPath</b> (the "
        "same model decoded by beam search for the most-likely loop-free path). The <b>learned "
        "model</b> is a light, self-contained <b>GCN encoder + GRU pointer decoder</b> (no "
        "PyTorch-Geometric needed): a 2-layer GCN over a per-fold POI graph (geographic kNN ∪ "
        "training co-visit edges) produces POI features; a GRU seeded by the query rolls out, "
        "scoring each POI by inner product (a pointer), masked to stay loop-free with the end "
        "reserved for the last hop. A fresh model is trained on every leave-one-out fold."
    ))
    s.append(MethodArch())
    s.append(mr.caption("Figure 7 — Top: the classical counting baselines. Bottom: the GCN+pointer "
                        "architecture. The optional Markov-prior lever blends the classical "
                        "transition signal into the pointer's logits at decode time."))

    # 8. validation
    s.append(section("8. Did we do it right? Reproducing Chen 2016", 1))
    s.append(body(
        "The decisive check: a <b>Random</b> baseline has no modelling choices, so it can only "
        "match Chen 2016's Random if our data, protocol and metric are all identical to the "
        "paper's. They match within noise on every city — and PoiPopularity matches closely too "
        "(near-exactly on Glasgow, Edinburgh, Melbourne). This is the evidence the comparison is "
        "valid."
    ))
    faith = [["pairs-F1", "Toronto", "Osaka", "Glasgow", "Edinburgh", "Melbourne"]]
    for m in ("Random", "PoiPopularity"):
        faith.append([f"{m} — ours"] + [f"{OURS_PAIRS_F1[m][c]:.3f}" for c in CITIES])
        faith.append([f"{m} — Chen 2016"] + [f"{CHEN_PAIRS_F1[m][c]:.3f}" for c in CITIES])
    s.append(mr.make_table(faith, col_widths=[4.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm]))
    s.append(Spacer(1, 0.2 * cm))
    s.append(img(ASSETS["validation"], 10.5))
    s.append(mr.caption("Figure 8 — Our pairs-F1 vs Chen 2016's. Points on the dashed y = x line "
                        "are exact reproductions; Random sits on the line, confirming a faithful "
                        "protocol."))

    # 9. results
    s.append(section("9. Results — ours vs the literature", 1))
    s.append(body(
        "Our measured pairs-F1 (leave-one-out, length ≥ 3). All values are on the published "
        "0–1 scale, versus 0.26–0.29 on Foursquare NYC:"
    ))
    s.append(mr.make_table(_pf1_table(OURS_PAIRS_F1, CITIES),
                           col_widths=[4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm]))
    s.append(mr.caption("Table 2 — Our classical pairs-F1 per city (leave-one-out, length ≥ 3)."))
    s.append(Spacer(1, 0.2 * cm))
    s.append(img(ASSETS["results"], 15.5))
    s.append(mr.caption("Figure 9 — Our methods beside the published SOTA (AR-Trip). Everything is "
                        "on the literature's scale; the red band is where our NYC itinerary numbers "
                        "sat."))
    s.append(Spacer(1, 0.2 * cm))
    s.append(body("The directly-comparable published numbers (same protocol, 0–1 scale):"))
    pub_rows = [["pairs-F1 (published)"] + PUB_CITIES]
    for m, (vals, _fam, _yr) in PUBLISHED_PAIRS_F1.items():
        pub_rows.append([m] + [f"{v:.3f}" for v in vals])
    t = mr.make_table(pub_rows, col_widths=[6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
    s.append(t)
    s.append(mr.caption("Table 3 — Published pairs-F1 (curated comparable subset). No single SOTA "
                        "across all cities: AR-Trip leads on Toronto/Glasgow/Edinburgh, SelfTrip on "
                        "Osaka (0.851)."))

    # 10. learned model honest
    s.append(section("10. The learned model — measured result and the honest finding", 1))
    ptr = [["", "Toronto", "Osaka", "Glasgow", "Edinburgh", "Melbourne"]]
    ptr.append(["Pointer (beam) pairs-F1"] + [f"{POINTER_PAIRS_F1['Pointer (beam)'][c]:.3f}" for c in CITIES])
    ptr.append(["Pointer point-F1"] + [f"{POINTER_F1['Pointer (beam)'][c]:.3f}" for c in CITIES])
    ptr.append(["ref: Markov (ours)"] + [f"{OURS_PAIRS_F1['Markov'][c]:.3f}" for c in CITIES])
    s.append(mr.make_table(ptr, col_widths=[4.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.3 * cm, 2.3 * cm]))
    s.append(Spacer(1, 0.2 * cm))
    s.append(body(
        "<b>Honest reading.</b> The light from-scratch pointer reaches pairs-F1 <b>0.31–0.49 — on "
        "the published scale and clearly above Random</b> — and its point-F1 (~0.68–0.73) shows it "
        "recovers the <i>right POIs</i>. But its <b>ordering trails the simple Markov baseline on "
        "every city</b>, and it is far below the neural SOTA band (0.66–0.85). This is the same "
        "lesson as Strategy B on NYC: <i>naively training a dedicated pointer on the tiny per-fold "
        "data does not beat a strong simple model.</i> The published neural methods reach SOTA "
        "with machinery our light model omits — self-supervised / contrastive pre-training on the "
        "unlabelled trajectories, trajectory augmentation, and adversarial training."
    ))
    s.append(body(
        "<b>Levers to close the gap</b> (implemented, opt-in in <code>PointerConfig</code>, "
        "reported side-by-side in the notebook): a decode-time <b>Markov transition prior</b> "
        "(blends the fold's empirical log P(j|i) into the pointer logits — directly targets the "
        "weak ordering; a quick Osaka check lifts pairs-F1 from 0.42 to 0.44, edging past Markov), "
        "an optional <b>user embedding</b>, and <b>early stopping</b> on an internal validation "
        "split."
    ))

    # 11. honest deviations
    s.append(section("11. Honest deviations and limitations", 1))
    s.append(bullet("<b>Markov ≠ Chen's Markov.</b> Ours is a raw empirical first-order transition "
                    "matrix (Laplace-smoothed); Chen's is feature-factored. Ours therefore differs "
                    "and on the data-rich cities exceeds Chen's published Markov — a modelling "
                    "difference, not a protocol bug. Random and PoiPopularity are the clean "
                    "reproductions."))
    s.append(bullet("<b>PoiRank / Rank+Markov are cited, not re-implemented</b> (they need a "
                    "per-query rankSVM); their numbers come straight from Chen 2016."))
    s.append(bullet("<b>Excluded as not comparable:</b> POIBERT (80/20 split, percentage F1, no "
                    "pairs-F1), DLIR 2025 (8-hour split, different F1 ≈ 0.49), TourEmbedding (no "
                    "pairs-F1, percentage-scale F1, only the first POI given, no leave-one-out)."))
    s.append(bullet("<b>Reproducibility:</b> classical numbers reproduce to ±0.01 across NumPy "
                    "versions (RNG/tie-break on these tiny datasets) — well within the papers' "
                    "±0.15–0.25 std deviations."))

    # 12. how to run
    s.append(section("12. How to run", 1))
    s.append(body("<b>Locally (CPU, classical — real numbers in minutes):</b>"))
    s.append(mr.code("py -3.11 -m src.flickr.run_flickr --data_dir data/flickr"))
    s.append(body("<b>Colab (GPU, the learned pointer):</b> open <code>colab_flickr.ipynb</code> — "
                  "it clones the repo, downloads the five cities, runs the classical baselines, "
                  "then trains the GCN+pointer per leave-one-out fold (pure and Markov-prior+user "
                  "variants) and prints the combined comparison vs the published numbers. No "
                  "PyTorch-Geometric needed."))
    s.append(body("<b>Tests:</b> <code>py -3.11 -m pytest</code> — 79 cases, including the pairs-F1 "
                  "protocol pinned against Chen 2016's reference, the leave-one-out splitter, and "
                  "baseline / pointer invariants."))

    # appendix: module map
    s.append(section("Appendix A — Code map", 1))
    impl = [["Module", "What it does"],
            ["src/flickr/data.py", "load Chen (traj-/poi-) & Lim (userVisits-/POI-) formats → FlickrCity"],
            ["src/flickr/evaluate.py", "leave-one-out CV harness; F1 + pairs-F1; loop-free filter"],
            ["src/flickr/baselines.py", "Random / PoiPopularity / Markov / MarkovPath + factories"],
            ["src/flickr/pointer.py", "self-contained GCN+pointer; per-fold trainer; Markov-prior/user levers"],
            ["src/flickr/published.py", "curated comparable literature numbers + exclusion notes"],
            ["src/flickr/run_flickr.py", "orchestration + ours-vs-published tables + CLI"],
            ["tests/test_flickr.py", "loader, trajectory build, pairs-F1 vs Chen, splitter, baselines"]]
    s.append(mr.make_table(impl, col_widths=[4.6 * cm, 11.4 * cm]))

    s.append(section("Appendix B — Key references", 1))
    for r in [
        "Chen, Ong, Xie. <i>Learning Points and Routes to Recommend Trajectories.</i> CIKM 2016 — origin of pairs-F1 + our data.",
        "Gao et al. <i>DeepTrip.</i> SIGSPATIAL 2019. · Gao et al. <i>SelfTrip.</i> KBS 2022.",
        "Rashid et al. <i>DeepAltTrip.</i> TKDE 2021. · Shu et al. <i>AR-Trip.</i> SIGIR 2024.",
        "Lim et al. <i>Tour Recommendation and Trip Planning using LBSNs: A Survey.</i> KAIS 2019.",
        "Krichene, Rendle. <i>On Sampled Metrics for Item Recommendation.</i> KDD 2020 — why cross-protocol metrics are inconsistent.",
    ]:
        s.append(bullet(r))
    return s


def main():
    doc = BaseDocTemplate(OUTPUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2 * cm, bottomMargin=2 * cm,
                          title="Strategy D — Itinerary recommendation on Flickr")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="default", frames=frame, onPage=on_page)])
    doc.build(build_story())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
