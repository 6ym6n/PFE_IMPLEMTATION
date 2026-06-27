"""Trimmed, two-pillar variant of the model deck.

Keeps the two pillars explicit — PILLAR 1 = Frozen Rollout (build the itinerary,
Foursquare NYC) and PILLAR 2 = Flickr Benchmark (validate vs the literature) — and
collapses the Trained Pointer to a single "rigor" slide. Reuses make_model_pptx.py's
slide functions + helpers. Run:

    py -3.11 presentation/strategy_d/make_model_lean_pptx.py

Output: presentation/strategy_d/model_slides_lean.pptx
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_model_pptx as mm  # noqa: E402  (reuse its prs, helpers, slide funcs)
from flickr_results_data import AVSB_NYC  # noqa: E402

OUTPUT = os.path.join(HERE, "model_slides_lean.pptx")
NAVY, INDIGO, TEAL, ORANGE, RED = mm.NAVY, mm.INDIGO, mm.TEAL, mm.ORANGE, mm.RED
GREY, GREY_MED = mm.GREY, mm.GREY_MED
BG_G, BG_C, BG_U, BG_S, BG_H, BG_LIGHT = mm.BG_G, mm.BG_C, mm.BG_U, mm.BG_S, mm.BG_H, mm.BG_LIGHT


def s_two_pillars(n):
    s = mm.slide()
    mm.header(s, "The approach — two pillars", "overview")
    mm.box(s, 4.75, 1.45, 3.85, 0.95,
           [("the next-POI engine", 14, INDIGO, True), ("GCN + GRU + user + context", 10, GREY, False)],
           BG_G, INDIGO)
    mm.arrow(s, 6.0, 2.4, 3.5, 3.05)
    mm.box(s, 0.7, 3.05, 5.7, 2.7,
           [("PILLAR 1", 13, ORANGE, True), ("Frozen Rollout", 18, ORANGE, True),
            ("build the itinerary —", 13, GREY, False), ("decode the engine, loop-free", 12, GREY, False),
            ("(Foursquare NYC)", 12, GREY_MED, False)], BG_H, ORANGE)
    mm.box(s, 6.95, 3.05, 5.7, 2.7,
           [("PILLAR 2", 13, TEAL, True), ("Flickr Benchmark", 18, TEAL, True),
            ("validate vs the literature —", 13, GREY, False), ("same dataset + protocol", 12, GREY, False),
            ("as the published papers", 12, GREY_MED, False)], BG_S, TEAL)
    mm.text(s, 0.7, 6.05, 12.0, 0.9,
            [[("Pillar 2 is the ", 15, GREY, False, False), ("only valid bridge", 15, NAVY, True, False),
              (" to other papers' numbers — same scale, same rules. (NYC scores cannot be compared "
               "directly to the literature.)", 15, GREY, False, False)]])
    mm.footer(s, n)


def s_frozen_results(n):
    s = mm.slide()
    mm.header(s, "Frozen Rollout — results (Foursquare NYC)", "pillar 1 result")
    g = AVSB_NYC["A — frozen rollout (greedy)"]
    b = AVSB_NYC["A — frozen rollout (beam 3)"]
    rows = [["Decoder", "pairs-F1", "set-F1"],
            ["greedy", f"{g[0]:.3f}", f"{g[1]:.3f}"],
            ["beam (3)", f"{b[0]:.3f}", f"{b[1]:.3f}"]]
    mm.table(s, rows, 0.7, 1.6, 6.3, 1.8, font=14)
    mm.bullets(s, 0.7, 3.8, 12.0, 2.2, [
        [("Simplest, most interpretable", NAVY, True),
         (" — an inference-time decoder over the frozen engine; no extra training.", GREY, False)],
        [("Reuses the personalized engine directly", NAVY, True),
         (" — the itinerary inherits user preferences + context.", GREY, False)],
        [("Beam barely beats greedy", NAVY, True),
         (" — the engine is myopic; better decoding can't fix that (motivates Pillar 2 + future work).",
          GREY, False)],
    ], size=16)
    mm.text(s, 0.7, 6.25, 12.0, 0.6,
            [[("Note: these NYC numbers are ", 14, GREY, False, True),
              ("not comparable to the papers", 14, RED, True, True),
              (" — that is Pillar 2's job.", 14, GREY, False, True)]])
    mm.footer(s, n)


def s_rigor(n):
    s = mm.slide()
    mm.header(s, "Rigor — what else I tested", "honest checks")
    mm.bullets(s, 0.7, 1.5, 12.0, 3.0, [
        [("Integration (the Trained Pointer).", NAVY, True),
         (" A dedicated pointer trained end-to-end did NOT beat the Frozen Rollout — the engine's far "
          "denser supervision wins. (Directly tests Halder/DLIR's integration claim.)", GREY, False)],
        [("Simple baselines are strong.", NAVY, True),
         (" On the Flickr benchmark a Markov transition model is the strongest of my methods — "
          "quantified, not hidden.", GREY, False)],
        [("Personalization.", NAVY, True),
         (" User-embedding ablation: mixed (Glasgow +0.038; Osaka / Toronto ~neutral) — helps where "
          "users recur, cold-start otherwise.", GREY, False)],
    ], size=15.5, gap=12)
    fr = AVSB_NYC["A — frozen rollout (beam 3)"]
    tp = AVSB_NYC["B-v2 — pointer (+ context)"]
    rows = [["NYC, len≥3", "pairs-F1"], ["Frozen Rollout", f"{fr[0]:.3f}"], ["Trained Pointer", f"{tp[0]:.3f}"]]
    mm.table(s, rows, 0.7, 4.95, 4.3, 1.5, font=13, highlight_row=1)
    mm.text(s, 5.4, 5.25, 7.2, 1.1,
            [[("Takeaway: ", 15, NAVY, True, False),
              ("the simple, decoupled Frozen Rollout is the right headline — the learned / integrated "
               "variants don't beat it on this data.", 15, GREY, False, False)]])
    mm.footer(s, n)


def build():
    mm.s_title()            # 1
    mm.s_goal(2)            # 2
    s_two_pillars(3)        # 3  (NEW: two pillars)
    mm.s_arch(4)            # 4  the model
    mm.s_parts(5)           # 5
    mm.s_title_fit(6)       # 6
    mm.s_engine_results(7)  # 7  engine works
    mm.s_strategyA(8)       # 8  Frozen Rollout method
    mm.s_example(9)         # 9  worked example
    s_frozen_results(10)    # 10 Frozen Rollout result (pillar 1)
    mm.s_validation(11)     # 11 Flickr Benchmark (pillar 2)
    s_rigor(12)             # 12 (NEW: Trained Pointer collapsed here)
    mm.s_future(13)         # 13
    mm.s_conclusion(14)     # 14
    mm.prs.save(OUTPUT)
    print(f"Wrote {OUTPUT}  ({len(mm.prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
