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
from flickr_results_data import (  # noqa: E402
    AVSB_NYC, OURS_PAIRS_F1, PHASE1_TIER, PUBLISHED_PAIRS_F1,
)

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


def s_nextpoi_comparison(n):
    s = mm.slide()
    mm.header(s, "Next-POI engine — comparison with published methods", "Foursquare NYC, HR@1")
    fam = {"LSTM": "sequence (RNN)", "STGCN": "spatio-temporal GCN",
           "Ours (GCN+GRU+user)": "this work", "STAN": "self-attention",
           "GETNext": "transformer", "STHGCN": "hypergraph", "LLM4POI": "frozen LLM"}
    rows = [["Method", "HR@1", "family"]]
    for m, v in PHASE1_TIER:
        rows.append([m, f"{v:.3f}", fam.get(m, "")])
    hl = next(i for i, (m, _v) in enumerate(PHASE1_TIER) if m.startswith("Ours")) + 1
    mm.table(s, rows, 2.1, 1.55, 9.1, 3.7, font=13, highlight_row=hl)
    mm.text(s, 0.7, 5.6, 12.0, 1.1,
            [[("Full-vocabulary HR@1 on Foursquare NYC, against the LLM4POI benchmark table. ", 15, GREY, False, False),
              ("Our engine sits in the LSTM/STGCN tier", 15, NAVY, True, False),
              (" — an honest, leakage-free baseline, below the transformer / LLM state of the art.", 15, GREY, False, False)]])
    mm.footer(s, n)


def s_itinerary_comparison(n):
    s = mm.slide()
    mm.header(s, "Itinerary recommendation on Flickr — the literature", "pairs-F1, same protocol")
    cities4 = ["Toronto", "Osaka", "Glasgow", "Edinburgh"]
    rows = [["Method", "Tor", "Osa", "Gla", "Edi", "source"]]
    rows.append(["Markov  (ours)"] + [f"{OURS_PAIRS_F1['Markov'][c]:.2f}" for c in cities4] + ["reproduces Chen"])
    for name in ["PoiRank (Chen)", "Rank+Markov (Chen)", "DeepTrip", "CTLTR", "SelfTrip", "AR-Trip"]:
        vals, _fam, yr = PUBLISHED_PAIRS_F1[name]
        rows.append([name] + [f"{v:.2f}" for v in vals] + [str(yr)])
    mm.table(s, rows, 0.7, 1.55, 12.0, 3.9, font=12, highlight_row=1)
    mm.text(s, 0.7, 5.75, 12.0, 1.1,
            [[("All on the ", 14, GREY, False, False), ("same benchmark + protocol", 14, NAVY, True, False),
              (" (Flickr, leave-one-out, endpoints given, length≥3). Our classical methods reproduce the "
               "classical literature (Chen 2016); the neural line (DeepTrip 2019 → AR-Trip 2024) is the "
               "state of the art to aim for.", 14, GREY, False, False)]])
    mm.footer(s, n)


def build():
    mm.s_title()            # 1
    mm.s_goal(2)            # 2
    s_two_pillars(3)        # 3  (NEW: two pillars)
    mm.s_arch(4)            # 4  the model
    mm.s_parts(5)           # 5
    mm.s_title_fit(6)       # 6
    s_nextpoi_comparison(7) # 7  next-POI vs published methods
    mm.s_strategyA(8)       # 8  Frozen Rollout method
    mm.s_example(9)         # 9  worked example
    s_frozen_results(10)    # 10 Frozen Rollout result (pillar 1)
    mm.s_validation(11)     # 11 Flickr Benchmark (pillar 2)
    s_itinerary_comparison(12)  # 12 itinerary vs published methods (literature)
    mm.s_future(13)         # 13
    mm.s_conclusion(14)     # 14
    mm.prs.save(OUTPUT)
    print(f"Wrote {OUTPUT}  ({len(mm.prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
