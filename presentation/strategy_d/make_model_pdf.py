"""Detailed companion document for the MODEL-centric presentation (model_slides.pptx).

Explains, in plain language with diagrams and worked examples, every slide of the
jury deck: the goal, THE model (the next-POI engine, component by component), why it
carries the title, the engine's results, Strategy A (the itinerary method) with a
worked example and results, the validation, and the supporting experiments
(integration, Markov, personalization). Run:

    py -3.11 presentation/strategy_d/make_model_pdf.py

Output: presentation/strategy_d/model_explained.pdf
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "report"))
sys.path.insert(0, HERE)

import make_report as mr  # noqa: E402
import make_pdf as mpdf  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from flickr_results_data import (  # noqa: E402
    AVSB_NYC, CITIES, OURS_PAIRS_F1, PERSONALIZATION, PHASE1_METRICS, PHASE1_TIER,
    POINTER_PAIRS_F1,
)

S = mr.STYLES
NAVY, INDIGO, TEAL, ORANGE = mr.NAVY, mr.INDIGO, mr.TEAL, mr.ORANGE
GREY_DARK, GREY_MED, GREY_LIGHT, WHITE = mr.GREY_DARK, mr.GREY_MED, mr.GREY_LIGHT, mr.WHITE
BG_G, BG_C, BG_U, BG_S, BG_H = (mr.BG_BOX_GRAPH, mr.BG_BOX_CTX, mr.BG_BOX_USER,
                                mr.BG_BOX_SEQ, mr.BG_BOX_HEAD)
RED = mr.colors.HexColor("#C0392B")
ASSETS = mpdf.ASSETS
img = mpdf.img
body, bullet, section = mpdf.body, mpdf.bullet, mpdf.section
OUTPUT = os.path.join(HERE, "model_explained.pdf")


def callout(html, fill=BG_S, stroke=TEAL):
    p = Paragraph(html, S["Body"])
    t = Table([[p]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill), ("BOX", (0, 0), (-1, -1), 1.2, stroke),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


class StrategyARollout(mr.Diagram):
    """route=[s] -> engine scores -> mask/reserve -> append -> loop."""

    def __init__(self):
        super().__init__(16.4 * cm, 4.4 * cm)

    def draw(self):
        c = self.canv
        steps = [("route = [start]", BG_U, INDIGO), ("engine scores\nall POIs", BG_G, INDIGO),
                 ("mask visited;\nreserve the end", BG_C, ORANGE), ("append the\nbest POI", BG_H, ORANGE)]
        bw = 3.4 * cm
        y = 2.4 * cm
        for i, (t, fill, stroke) in enumerate(steps):
            x = 0.3 * cm + i * 4.0 * cm
            self._box(c, x, y, bw, 1.3 * cm, fill, stroke=stroke)
            for k, ln in enumerate(t.split("\n")):
                self._label(c, ln, x + bw / 2, y + 0.85 * cm - k * 0.45 * cm, size=10,
                            color=stroke if k == 0 else GREY_DARK,
                            font="Helvetica-Bold" if k == 0 else "Helvetica")
            if i < 3:
                self._arrow(c, x + bw, y + 0.65 * cm, x + bw + 0.6 * cm, y + 0.65 * cm)
        # loop back
        lastx = 0.3 * cm + 3 * 4.0 * cm + bw
        self._arrow(c, lastx, y, lastx, y - 0.9 * cm, color=TEAL)
        self._arrow(c, lastx, y - 0.9 * cm, 0.3 * cm + bw / 2, y - 0.9 * cm, color=TEAL)
        self._arrow(c, 0.3 * cm + bw / 2, y - 0.9 * cm, 0.3 * cm + bw / 2, y, color=TEAL)
        self._label(c, "repeat until length K, ending at e", 8.2 * cm, y - 1.25 * cm, size=9,
                    color=TEAL, font="Helvetica-Oblique")


class WorkedExample(mr.Diagram):
    """Edinburgh: query (15,16,K=4) -> build [15,3,13,16]."""

    def __init__(self):
        super().__init__(16.4 * cm, 5.2 * cm)

    def draw(self):
        c = self.canv
        self._box(c, 0.3 * cm, 4.0 * cm, 6.0 * cm, 0.9 * cm, BG_U, stroke=INDIGO)
        self._label(c, "QUERY:  start 15 · end 16 · K = 4", 3.3 * cm, 4.35 * cm, size=10, color=INDIGO)
        rows = [("step 1", ["15"], "engine → best next = 3"),
                ("step 2", ["15", "3"], "engine → best next = 13"),
                ("step 3", ["15", "3", "13"], "last hop → reserved end 16"),
                ("result", ["15", "3", "13", "16"], "= the real trip ✓")]
        for i, (label, seq, note) in enumerate(rows):
            y = 3.0 * cm - i * 0.78 * cm
            last = i == 3
            self._label(c, label, 0.3 * cm, y + 0.2 * cm, size=9.5, color=NAVY, anchor="left",
                        font="Helvetica-Bold")
            for j, p in enumerate(seq):
                x = 2.0 * cm + j * 0.95 * cm
                self._box(c, x, y, 0.78 * cm, 0.6 * cm, BG_S if last else GREY_LIGHT,
                          stroke=TEAL if last else NAVY, radius=4)
                self._label(c, p, x + 0.39 * cm, y + 0.2 * cm, size=11, color=TEAL if last else NAVY)
            self._label(c, note, 7.0 * cm, y + 0.2 * cm, size=9.5, color=GREY_MED, anchor="left",
                        font="Helvetica-Oblique")


def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setStrokeColor(GREY_LIGHT); c.setLineWidth(0.6)
        c.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        c.setFont(mr._UNICODE_FONT_NAME, 8); c.setFillColor(GREY_MED)
        c.drawString(2 * cm, A4[1] - 1.1 * cm, "The Model — presentation explained")
        c.drawRightString(A4[0] - 2 * cm, A4[1] - 1.1 * cm, "next-POI engine + Frozen Rollout")
        c.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
        c.drawCentredString(A4[0] / 2, 1.0 * cm, f"— {doc.page} —")
    c.restoreState()


def build_story():
    s = []
    s.append(Spacer(1, 3.4 * cm))
    s.append(Paragraph("The Model", S["Title"]))
    s.append(Paragraph("A personalized, context-aware itinerary recommender — explained", S["Subtitle"]))
    s.append(Spacer(1, 0.6 * cm))
    s.append(Paragraph("The next-POI engine (GCN + GRU + user embedding + context), decoded into "
                       "tourist itineraries by the Frozen Rollout", S["Subtitle"]))
    s.append(Spacer(1, 2.4 * cm))
    s.append(Paragraph("Companion document to the model-centric jury deck", S["Author"]))
    s.append(Paragraph("Repository: github.com/6ym6n/PFE_IMPLEMTATION", S["Author"]))
    s.append(PageBreak())

    s.append(section("Executive summary — the model in a nutshell", 1))
    s.append(body(
        "This document explains the deck I present to the jury. The headline is a single "
        "<b>model</b> and a single <b>method</b>. The model is a <b>next-POI engine</b>: a graph "
        "neural network (GCN) over a POI graph, a sequence model (GRU) over the visited route, a "
        "<b>user embedding</b> (the title's <i>user preferences</i>), and per-step <b>context</b> "
        "(Δdistance, Δtime — the title's <i>contextual data</i>), combined into a score over all "
        "POIs. The method is the <b>Frozen Rollout</b>: roll that engine out — repeatedly append the "
        "best next POI, masking visited ones and reserving the end — to build a full itinerary, "
        "with no extra training."
    ))
    s.append(body(
        "Everything else is a supporting experiment: training a dedicated <i>integrated</i> "
        "itinerary model (the Trained Pointer) — which did <b>not</b> beat the simpler Frozen Rollout; a "
        "literature-comparable benchmark on the Flickr datasets (the Flickr Benchmark) — which validates "
        "the approach and shows a simple Markov baseline is strong; and a personalization "
        "ablation. The takeaway I defend: <b>a strong, honest, personalized engine, decoded "
        "simply into itineraries — the best and most interpretable of the methods I built.</b>"
    ))

    s.append(section("1. The goal", 1))
    s.append(body(
        "A tourist with limited time wants a <b>personalized day plan</b>: an ordered route of "
        "POIs (places worth visiting). The plan must reflect <b>who they are</b> (preferences) and "
        "the <b>context</b> (distances, time). My approach does both with one model — a "
        "personalized next-POI engine — decoded into an itinerary."
    ))

    s.append(section("2. The model in one picture", 1))
    s.append(body(
        "My <b>model</b> is the next-POI engine; my <b>itinerary method</b> (the Frozen Rollout) decodes "
        "it. The engine scores “what to visit next”; the Frozen Rollout calls it repeatedly to build the "
        "whole route."
    ))
    s.append(mpdf.PhaseMap())
    s.append(mr.caption("Figure 1 — The engine (the model) is reused everywhere: the Frozen Rollout decodes it "
                        "into NYC itineraries; the Flickr Benchmark validates the family."))

    s.append(section("3. The model — architecture", 1))
    s.append(body(
        "The engine fuses three signals into a score over every POI. (1) A <b>POI graph</b> — "
        "places linked if geographically close or often visited together — encoded by a 2-layer "
        "<b>GCN</b> into a feature vector per POI. (2) Per-step <b>context</b> (Δd, Δt) encoded by "
        "small MLPs. (3) A <b>user embedding</b>. A <b>GRU</b> reads the route so far; an <b>MLP "
        "head</b> combines its state with the user to score the next POI."
    ))
    s.append(mr.ArchitectureDiagram())
    s.append(mr.caption("Figure 2 — The model I built: graph + context + user → GRU → a next-POI score. "
                        "Trained once on Foursquare NYC; reused for every itinerary."))

    s.append(section("4. How each part works", 1))
    s.append(bullet("<b>POI graph + GCN.</b> Places are nodes; edges join nearby or co-visited POIs. "
                    "The GCN mixes each POI's vector with its neighbours' (2 hops), giving each POI a "
                    "“fingerprint” aware of its surroundings."))
    s.append(bullet("<b>Context (Δd, Δt).</b> How far and how long since the previous place; small "
                    "networks lift these scalars into vectors the model can use."))
    s.append(bullet("<b>User embedding.</b> A small learned “taste profile” per tourist — the title's "
                    "<i>user preferences</i>."))
    s.append(bullet("<b>GRU sequence model.</b> Reads the places visited so far and summarises them "
                    "into a single state vector."))
    s.append(bullet("<b>Scoring head.</b> Combines the state and the user, and outputs a score for "
                    "every POI — “what to visit next”."))

    s.append(section("5. The model carries the thesis title", 1))
    s.append(body(
        "The title — “…Based on <b>User Preferences</b> and <b>Contextual Data</b>” — is a "
        "description of the architecture, not an aspiration: <b>user preferences</b> = the user "
        "embedding; <b>contextual data</b> = the Δd / Δt context features. Both live inside the "
        "model."
    ))

    s.append(section("6. The engine works (Phase 1)", 1))
    s.append(body(
        "Before building itineraries, the engine must be a sound next-POI model. On Foursquare NYC "
        f"(full vocabulary) it reaches <b>HR@1 = {PHASE1_METRICS['HR@1']:.3f}</b> "
        f"(HR@10 = {PHASE1_METRICS['HR@10']:.3f}, MRR = {PHASE1_METRICS['MRR']:.3f}) — the honest "
        "LSTM/STGCN tier of the LLM4POI benchmark. A strong, leakage-free engine."
    ))
    tier = [["Model", "HR@1"]] + [[m, f"{v:.3f}"] for m, v in PHASE1_TIER]
    s.append(mr.make_table(tier, col_widths=[8 * cm, 3 * cm]))
    s.append(mr.caption("Table 1 — HR@1 tier (LLM4POI). Ours sits between STGCN and STAN."))

    s.append(section("7. Frozen Rollout — the itinerary method", 1))
    s.append(body(
        "The Frozen Rollout turns the engine into an itinerary by <b>rolling it out</b>: start at the start "
        "POI, ask the engine to score the next POI, block already-visited POIs (loop-free) and keep "
        "the end POI in reserve, append the best one, and repeat until the route has the desired "
        "length K and ends at the end POI. <b>No new training</b> — it is an inference-time decoder "
        "over the frozen engine, so the itinerary inherits the engine's personalization and context."
    ))
    s.append(StrategyARollout())
    s.append(mr.caption("Figure 3 — Frozen Rollout: build the route one POI at a time by rolling out the "
                        "engine. Always loop-free, exactly K stops, start→…→end."))

    s.append(section("8. Frozen Rollout — worked example (Edinburgh)", 1))
    s.append(body(
        "A real Edinburgh trip. The query gives start = 15, end = 16, K = 4; the engine fills the "
        "middle. Here it recovers the true route exactly:"
    ))
    s.append(WorkedExample())
    s.append(mr.caption("Figure 4 — The engine proposes POI 3, then 13; the last hop is the reserved "
                        "end 16 → route [15, 3, 13, 16], matching the real trip (pairs-F1 = 1.0)."))

    s.append(section("9. Frozen Rollout — results, and why it is the headline", 1))
    s.append(body(
        "On Foursquare NYC (length ≥ 3), the Frozen Rollout is the strongest of my itinerary methods — and "
        "crucially it <b>beats the dedicated integrated model</b> (the Trained Pointer):"
    ))
    rows = [["Method (NYC, len≥3)", "pairs-F1", "kind"]]
    _relabel = {"A — frozen rollout (greedy)": "Frozen Rollout (greedy)",
                "A — frozen rollout (beam 3)": "Frozen Rollout (beam 3)",
                "B-v1 — pointer (no context)": "Trained Pointer (no context)",
                "B-v2 — pointer (+ context)": "Trained Pointer (+ context)"}
    for m in _relabel:
        pf1, _x, _y, kind = AVSB_NYC[m]
        rows.append([_relabel[m], f"{pf1:.3f}", kind])
    s.append(mr.make_table(rows, col_widths=[8.5 * cm, 3.5 * cm, 4 * cm]))
    s.append(callout(
        "<b>Decoding the strong engine beats training a separate model.</b> The Frozen Rollout is the best "
        "(~0.29 pairs-F1), the simplest, and the most interpretable — and it reuses the personalized "
        "engine directly, so the itinerary inherits user preferences and context. That is why it is "
        "the model + method I lead with."))

    s.append(section("10. Validation — on the literature's scale", 1))
    s.append(body(
        "Because the NYC itinerary scores are not comparable to the published literature (a "
        "different benchmark), I re-run the family on the standard <b>Flickr</b> datasets under the "
        "exact published protocol (the Flickr Benchmark). A <b>Random baseline reproduces Chen 2016</b> "
        "(proving the setup is fair), and my methods land on the published 0.3–0.85 pairs-F1 scale "
        "— with the honest note that a simple <b>Markov</b> baseline is the strongest of mine on "
        "this tiny data."
    ))
    s.append(img(ASSETS["results"], 15.0))
    s.append(mr.caption("Figure 5 — Our methods beside the published SOTA. The Flickr Benchmark validates the "
                        "approach; it does not replace the headline engine + Frozen Rollout."))

    s.append(section("11. What else I tested (rigor)", 1))
    s.append(bullet("<b>Integration (the Trained Pointer).</b> A dedicated pointer trained end-to-end did NOT "
                    "beat the Frozen Rollout — the engine's far denser training signal wins. This directly "
                    "tests Halder/DLIR's integration claim and nuances it."))
    s.append(bullet("<b>Simple baselines.</b> On the small Flickr data, a Markov transition model is "
                    "the strongest of my methods — quantified, not hidden."))
    s.append(body("<b>Personalization.</b> I switch the user embedding off vs on (identical otherwise) "
                  "on the Flickr benchmark. The result is honest and mixed:"))
    pers = [["City", "no user", "with user", "delta", "recurring users"]]
    rec = {"Osaka": "28%", "Glasgow": "32%", "Toronto": "55%"}
    for c in ("Osaka", "Glasgow", "Toronto"):
        d = PERSONALIZATION[c]
        pers.append([c, f"{d['no_user']:.3f}", f"{d['user']:.3f}",
                     f"{d['user'] - d['no_user']:+.3f}", rec[c]])
    pers.append(["Edinburgh, Melbourne", "—", "—", "(GPU run)", "59%, 61%"])
    s.append(mr.make_table(pers, col_widths=[4.6 * cm, 2.6 * cm, 2.8 * cm, 2.4 * cm, 3.6 * cm]))
    s.append(body(
        "A clear gain on Glasgow (+0.038), roughly neutral on Osaka and Toronto. The reason is "
        "<b>cold-start</b>: in leave-one-out even recurring users have very few trips, so the "
        "embedding is weakly estimated. <i>Personalization is a real lever, not a guaranteed win on "
        "small data</i> — and it is strongest where the engine (Foursquare, dense per-user history) "
        "lives."
    ))

    s.append(section("12. Limitations & future work", 1))
    s.append(bullet("<b>Light context</b> — Δd/Δt only; add time-of-day, opening hours, queuing."))
    s.append(bullet("<b>No scheduling</b> — add explicit time budgets (the DLIR direction)."))
    s.append(bullet("<b>Integrate properly</b> — share/warm-start the engine with the trained model "
                    "to lift it past the Frozen Rollout."))
    s.append(bullet("<b>Stronger personalization</b> — richer user features; finish the Edinburgh / "
                    "Melbourne ablation on GPU."))

    s.append(section("13. In one sentence", 1))
    s.append(callout(
        "<b>A personalized, context-aware next-POI engine (GCN + GRU + user + context), decoded into "
        "tourist itineraries by rolling it out (the Frozen Rollout) — the strongest, simplest, and most "
        "honest of the methods I built; integration (the Trained Pointer) and the Flickr Benchmark are experiments "
        "that support it.</b>", fill=BG_C, stroke=TEAL))
    return s


def main():
    doc = BaseDocTemplate(OUTPUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2 * cm, bottomMargin=2 * cm,
                          title="The Model — presentation explained")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="default", frames=frame, onPage=on_page)])
    doc.build(build_story())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
