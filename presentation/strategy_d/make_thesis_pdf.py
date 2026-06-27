"""Detailed plain-language explainer for the WHOLE thesis (companion to the deck).

Covers the full arc: the Smart Visit Module goal, the next-POI engine (Phase 1),
the decoupled-vs-integrated itinerary experiment (Strategy A vs B), the
comparability problem, the Flickr benchmark (Strategy D), and the positioning vs
Halder/DLIR. Reuses the report styling + the Strategy-D diagrams/charts. Run:

    py -3.11 presentation/strategy_d/make_thesis_pdf.py

Output: presentation/strategy_d/thesis_explained.pdf
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "report"))
sys.path.insert(0, HERE)

import make_report as mr  # noqa: E402
import make_pdf as mpdf  # noqa: E402  (reuse its diagrams + img + ASSETS)
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from flickr_results_data import (  # noqa: E402
    AVSB_NYC, CHEN_PAIRS_F1, CITIES, DATASET_STATS, OURS_PAIRS_F1, PHASE1_METRICS,
    PHASE1_TIER, POINTER_PAIRS_F1, PUB_CITIES, PUBLISHED_PAIRS_F1,
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

OUTPUT = os.path.join(HERE, "thesis_explained.pdf")


def callout(html, fill=BG_S, stroke=TEAL):
    p = Paragraph(html, S["Body"])
    t = Table([[p]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill), ("BOX", (0, 0), (-1, -1), 1.2, stroke),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


class TwoStrategies(mr.Diagram):
    """The engine, used two ways: Strategy A (decoupled) and B (integrated)."""

    def __init__(self):
        super().__init__(16.4 * cm, 5.6 * cm)

    def draw(self):
        c = self.canv
        self._box(c, 5.7 * cm, 4.1 * cm, 5.0 * cm, 1.2 * cm, BG_G, stroke=INDIGO)
        self._label(c, "the next-POI engine (Phase 1)", 8.2 * cm, 4.85 * cm, size=11, color=INDIGO)
        self._label(c, "GCN + context + user", 8.2 * cm, 4.4 * cm, size=8.5, font="Helvetica", color=GREY_DARK)
        self._arrow(c, 7.0 * cm, 4.1 * cm, 4.5 * cm, 3.2 * cm, color=NAVY)
        self._arrow(c, 9.4 * cm, 4.1 * cm, 11.9 * cm, 3.2 * cm, color=NAVY)
        self._box(c, 0.3 * cm, 0.6 * cm, 7.4 * cm, 2.6 * cm, BG_H, stroke=ORANGE)
        self._label(c, "Strategy A — DECOUPLED", 4.0 * cm, 2.7 * cm, size=11, color=ORANGE)
        for i, t in enumerate(["decode the FROZEN engine into a route:",
                               "mask visited POIs, reserve the end,",
                               "append the best next POI, repeat.",
                               "No itinerary training."]):
            self._label(c, t, 4.0 * cm, 2.2 * cm - i * 0.45 * cm, size=9, font="Helvetica", color=GREY_DARK)
        self._box(c, 8.7 * cm, 0.6 * cm, 7.4 * cm, 2.6 * cm, BG_S, stroke=TEAL)
        self._label(c, "Strategy B — INTEGRATED", 12.4 * cm, 2.7 * cm, size=11, color=TEAL)
        for i, t in enumerate(["TRAIN a dedicated pointer model",
                               "end-to-end on whole trajectories",
                               "(GCN + GRU + pointer).",
                               "v1: no context · v2: + context."]):
            self._label(c, t, 12.4 * cm, 2.2 * cm - i * 0.45 * cm, size=9, font="Helvetica", color=GREY_DARK)


def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setStrokeColor(GREY_LIGHT); c.setLineWidth(0.6)
        c.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        c.setFont(mr._UNICODE_FONT_NAME, 8); c.setFillColor(GREY_MED)
        c.drawString(2 * cm, A4[1] - 1.1 * cm, "Smart Visit Module — thesis explained")
        c.drawRightString(A4[0] - 2 * cm, A4[1] - 1.1 * cm, "Personalized itinerary recommendation")
        c.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
        c.drawCentredString(A4[0] / 2, 1.0 * cm, f"— {doc.page} —")
    c.restoreState()


def avsb_rows():
    rows = [["Method", "pairs-F1", "kind"]]
    for m in ["A — frozen rollout (greedy)", "A — frozen rollout (beam 3)",
              "B-v1 — pointer (no context)", "B-v2 — pointer (+ context)"]:
        pf1, _s, _e, kind = AVSB_NYC[m]
        rows.append([m, f"{pf1:.3f}", kind])
    return rows


def build_story():
    s = []
    # cover
    s.append(Spacer(1, 3.3 * cm))
    s.append(Paragraph("Personalized Tourist Itinerary Recommendation", S["Title"]))
    s.append(Paragraph("The Smart Visit Module — the whole thesis, explained", S["Subtitle"]))
    s.append(Spacer(1, 0.6 * cm))
    s.append(Paragraph("From a personalized, context-aware next-POI engine to literature-comparable "
                       "itineraries — testing the field's integration hypothesis", S["Subtitle"]))
    s.append(Spacer(1, 2.4 * cm))
    s.append(Paragraph("Companion document to the defence deck", S["Author"]))
    s.append(Paragraph("Repository: github.com/6ym6n/PFE_IMPLEMTATION", S["Author"]))
    s.append(PageBreak())

    # executive summary
    s.append(section("Executive summary — the whole idea", 1))
    s.append(body(
        "This thesis builds a <b>Smart Visit Module</b> that recommends <b>personalized, context-aware "
        "tourist itineraries</b> — <i>ordered routes</i> of places to visit, tailored to the user. Its "
        "core is a <b>next-POI engine</b>: a graph neural network + sequence model that uses <b>who the "
        "user is</b> (a user embedding — the title's <i>user preferences</i>) and <b>context</b> (distance "
        "Δd and time Δt between places — the title's <i>contextual data</i>) to score the next place to "
        "visit."
    ))
    s.append(body(
        "The thesis then confronts the field's central question — raised by Halder's 2024 survey and the "
        "DLIR model (2025): should <b>choosing good POIs</b> and <b>ordering them into a route</b> be "
        "<b>separate</b> steps, or <b>integrated</b>? It builds <b>both</b> and compares them: <b>Strategy "
        "A</b> decodes the frozen engine into a route (decoupled), and <b>Strategy B</b> trains a dedicated "
        "itinerary model end-to-end (integrated). The <b>honest finding</b>: on Foursquare NYC the "
        "integrated model did <i>not</i> beat the decoupled one — because the engine was trained on far more "
        "examples. Integration is not automatically better; <b>supervision density matters</b>."
    ))
    s.append(body(
        "Finally, because the NYC itinerary scores are <b>not comparable</b> to the published literature "
        "(a different benchmark), the thesis re-runs everything on the standard <b>Flickr</b> datasets under "
        "the <b>exact published protocol</b> (<b>Strategy D</b>). There a <b>Random baseline reproduces "
        "Chen 2016</b> (proving the setup is fair), and the methods land on the published <b>0.3–0.85 "
        "pairs-F1 scale</b>. The contribution: a personalized, context-aware engine + an honest "
        "decoupled-vs-integrated study + a validated literature comparison — testing, and nuancing, the "
        "field's integration hypothesis, with a clear path to a full scheduling-aware module."
    ))
    s.append(Spacer(1, 0.2 * cm))
    s.append(img(ASSETS["scale"], 15.0))
    s.append(mr.caption("Figure 1 — Why Strategy D exists: the same task on a different benchmark sits on a "
                        "different scale. Our Flickr work puts us on the literature's scale."))

    # 1. goal
    s.append(section("1. The goal — a Smart Visit Module", 1))
    s.append(body(
        "A tourist with limited time faces hundreds of places and wants a <b>personalized day plan</b>: an "
        "ordered route of POIs (Points Of Interest — places worth visiting). A good plan depends on the "
        "<b>user</b> (their preferences) and the <b>context</b> (where they are, how far apart places are, "
        "the time). This is <b>itinerary recommendation</b> — producing the whole ordered route — which is "
        "a different problem from <b>next-POI prediction</b> (guessing only the single next place)."
    ))
    s.append(mpdf.PhaseMap())
    s.append(mr.caption("Figure 2 — Where the pieces sit. Phase 1 (next-POI) is the engine; Phase 2 turns it "
                        "into itineraries on Foursquare NYC; Strategy D validates on the Flickr benchmark."))

    # 2. background
    s.append(section("2. Background — places, trips, queries, and the score", 1))
    s.append(body(
        "A <b>trajectory</b> (a trip) is the ordered list of POIs one tourist visited. An itinerary "
        "<b>query</b> gives the <b>start POI</b>, the <b>end POI</b>, and the <b>length K</b>; the model "
        "must recover the ordered POIs in between."
    ))
    s.append(mpdf.QueryRoute())
    s.append(mr.caption("Figure 3 — The query gives the endpoints and the length; the model fills the "
                        "ordered middle."))
    s.append(body(
        "Quality is measured by two scores in [0, 1]. <b>point-F1</b> asks <i>did we pick the right "
        "places?</i> (ignoring order). <b>pairs-F1</b> (Chen 2016, the main metric) asks <i>did we get the "
        "order right too?</i> — it is the F1 over ordered pairs of POIs, where a pair (a, b) counts only if "
        "a comes before b in <i>both</i> the prediction and the truth."
    ))
    s.append(mpdf.PairsF1())
    s.append(mr.caption("Figure 4 — A worked pairs-F1: swapping two middle POIs breaks one ordered pair → "
                        "5 of 6 survive → 0.83. (In this task the route length K is fixed, so precision = "
                        "recall = F1.)"))

    # 3. Phase 1 engine
    s.append(section("3. Phase 1 — the personalized, context-aware engine", 1))
    s.append(body(
        "The engine is a <b>next-POI model</b> that fuses three signals into a score over all POIs: a "
        "<b>graph</b> of POIs (geographic kNN ∪ co-visit) encoded by a 2-layer <b>GCN</b>; per-step "
        "<b>context</b> (Δd, Δt) encoded by small MLPs; and a <b>user embedding</b>. A GRU reads the "
        "sequence so far, and an MLP head scores the next POI. The user embedding is the title's "
        "<i>user preferences</i>; the Δd/Δt features are the <i>contextual data</i>."
    ))
    s.append(mr.ArchitectureDiagram())
    s.append(mr.caption("Figure 5 — The next-POI engine (Phase 1): graph + context + user → GRU → score "
                        "over all POIs. This single model is reused by every itinerary strategy."))
    s.append(body(
        "<b>Phase-1 result.</b> On Foursquare NYC (full vocabulary) the engine reaches "
        f"<b>HR@1 = {PHASE1_METRICS['HR@1']:.3f}</b> (HR@10 = {PHASE1_METRICS['HR@10']:.3f}, "
        f"MRR = {PHASE1_METRICS['MRR']:.3f}) — the LSTM/STGCN tier of the LLM4POI benchmark. An honest, "
        "leakage-free baseline, below the transformer/LLM state of the art but a strong engine."
    ))
    tier = [["Model", "HR@1"]] + [[m, f"{v:.3f}"] for m, v in PHASE1_TIER]
    s.append(mr.make_table(tier, col_widths=[8 * cm, 3 * cm]))
    s.append(mr.caption("Table 1 — HR@1 tier (LLM4POI table). Ours sits between STGCN and STAN."))

    # 4. two strategies
    s.append(section("4. From engine to itinerary — two strategies", 1))
    s.append(body(
        "There are two ways to turn the engine into a full itinerary — and they are exactly the "
        "<b>decoupled</b> vs <b>integrated</b> options the field debates. <b>Strategy A</b> decodes the "
        "<i>frozen</i> engine: start at the start POI, repeatedly append the best next POI (masking visited "
        "ones, reserving the end), with no extra training. <b>Strategy B</b> trains a <i>dedicated</i> "
        "pointer model end-to-end on whole trajectories."
    ))
    s.append(TwoStrategies())
    s.append(mr.caption("Figure 6 — One engine, two strategies. A = decoupled decoding (the “separate "
                        "problems” approach); B = an integrated end-to-end itinerary model."))

    # 5. integration experiment
    s.append(section("5. The integration experiment — A vs B (NYC)", 1))
    s.append(body(
        "Comparing A and B directly <b>tests the field's integration hypothesis</b>. The result is honest "
        "and instructive — on Foursquare NYC (length ≥ 3 test trajectories):"
    ))
    s.append(mr.make_table(avsb_rows(), col_widths=[8.5 * cm, 3.5 * cm, 4 * cm]))
    s.append(mr.caption("Table 2 — Strategy A (decoupled) vs B (integrated), pairs-F1."))
    s.append(callout(
        "<b>The integrated model did NOT beat the decoupled one.</b> The trained pointer (B) trailed the "
        "frozen rollout (A) by ~0.03 pairs-F1. Why: the next-POI engine was trained on ~75,000 prefix→next "
        "examples, whereas the integrated pointer saw only ~10,000 whole trajectories. <b>Lesson: "
        "integration does not automatically win — supervision density matters.</b> This <i>nuances</i> "
        "Halder/DLIR rather than contradicting it.",
        fill=BG_H, stroke=RED))

    # 6. comparability + Strategy D
    s.append(section("6. The comparability problem, and Strategy D", 1))
    s.append(body(
        "Our NYC itinerary pairs-F1 (~0.29) looks far below the literature's ~0.6–0.8 — but that is a "
        "<b>comparability gap</b>, not a quality gap. Two scores compare only if the <b>dataset</b>, "
        "<b>protocol</b>, and <b>metric</b> all match. The literature uses the small <b>Flickr</b> "
        "photo-trajectory datasets (27–88 POIs) with leave-one-trajectory-out CV; NYC has ~5,000 POIs and a "
        "fixed split. So <b>Strategy D</b> re-runs the itinerary task on the Flickr datasets, under the "
        "exact published protocol, using Chen 2016's own trajectory files (so our trips are <i>identical</i> "
        "to the literature's)."
    ))
    s.append(mpdf.LOOCV())
    s.append(mr.caption("Figure 7 — Leave-one-trajectory-out CV: hold out one trip, train on the rest, "
                        "predict it, repeat for all, average. The protocol the Flickr papers use."))
    stats = [["City", "#POIs", "#users", "#trips", "#trips≥3 (evaluated)"]]
    for c in CITIES:
        p, u, t, e = DATASET_STATS[c]
        stats.append([c, str(p), f"{u:,}", f"{t:,}", str(e)])
    s.append(mr.make_table(stats, col_widths=[4 * cm, 2.6 * cm, 2.8 * cm, 2.8 * cm, 4 * cm]))
    s.append(mr.caption("Table 3 — The five Flickr cities. The length≥3 column is the evaluated set."))

    # 7. validation
    s.append(section("7. Validation — we reproduce Chen 2016", 1))
    s.append(body(
        "The decisive fairness check: a <b>Random</b> baseline has no modelling choices, so it can only "
        "match Chen 2016's Random if our data, protocol, and metric are identical to the paper's. They "
        "match within noise on every city — proving the comparison is valid."
    ))
    faith = [["pairs-F1", "Toronto", "Osaka", "Glasgow", "Edinburgh", "Melbourne"]]
    for m in ("Random", "PoiPopularity"):
        faith.append([f"{m} — ours"] + [f"{OURS_PAIRS_F1[m][c]:.3f}" for c in CITIES])
        faith.append([f"{m} — Chen 2016"] + [f"{CHEN_PAIRS_F1[m][c]:.3f}" for c in CITIES])
    s.append(mr.make_table(faith, col_widths=[4.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm]))
    s.append(Spacer(1, 0.2 * cm))
    s.append(img(ASSETS["validation"], 10.0))
    s.append(mr.caption("Figure 8 — Our pairs-F1 vs Chen 2016's; Random sits on the y = x line."))

    # 8. results
    s.append(section("8. Strategy D — results on the literature's scale", 1))
    s.append(body(
        "Our measured pairs-F1 (leave-one-out, length ≥ 3). The classical Markov baseline is the strongest "
        "of ours; the learned GCN+pointer reaches the published scale but trails Markov and the neural SOTA "
        "— the same lesson as the NYC integration experiment."
    ))
    res = [["pairs-F1", "Toronto", "Osaka", "Glasgow", "Edinburgh", "Melbourne"]]
    for m in ("Random", "PoiPopularity", "Markov", "MarkovPath"):
        res.append([m] + [f"{OURS_PAIRS_F1[m][c]:.3f}" for c in CITIES])
    res.append(["Pointer (learned)"] + [f"{POINTER_PAIRS_F1['Pointer (beam)'][c]:.3f}" for c in CITIES])
    s.append(mr.make_table(res, col_widths=[4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm]))
    s.append(Spacer(1, 0.2 * cm))
    s.append(img(ASSETS["results"], 15.0))
    s.append(mr.caption("Figure 9 — Our methods beside the published SOTA (AR-Trip). Everything is on the "
                        "literature's 0.3–0.85 scale; the red band is where our NYC numbers sat."))
    pub = [["pairs-F1 (published)"] + PUB_CITIES]
    for m, (vals, _f, _y) in PUBLISHED_PAIRS_F1.items():
        pub.append([m] + [f"{v:.3f}" for v in vals])
    s.append(mr.make_table(pub, col_widths=[6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    s.append(mr.caption("Table 4 — Published pairs-F1 (comparable subset). No single SOTA across cities: "
                        "AR-Trip leads on 3 cities, SelfTrip on Osaka."))

    # 9. positioning
    s.append(section("9. Positioning vs Halder / DLIR", 1))
    s.append(body(
        "Halder's 2024 survey defines the personalized-itinerary field; <b>DLIR (2025)</b> argues that "
        "treating POI recommendation and itinerary construction as <i>separate</i> problems is sub-optimal, "
        "and proposes an <b>integrated</b> model that also adds <b>dynamic temporal interest</b>, "
        "<b>queuing time</b>, and <b>time-budget scheduling</b>."
    ))
    s.append(bullet("<b>We test Halder's central claim directly</b> (Strategy A vs B) and report a nuanced "
                    "result: integration did not automatically win on our data."))
    s.append(bullet("<b>We model</b> user preference (embedding) and spatial-temporal context (Δd, Δt) — "
                    "the title's two pillars."))
    s.append(bullet("<b>We deliberately scope smaller</b> than DLIR: we do not yet model queuing, explicit "
                    "time budgets, or time-of-day dynamics. These are the path to a full module (Section 11)."))

    # 10. limitations
    s.append(section("10. Honest limitations", 1))
    s.append(bullet("<b>Integration didn't beat decoupling on NYC</b> — a supervision-density effect, not a "
                    "refutation of the idea."))
    s.append(bullet("<b>Personalization is uneven</b> — strong on Foursquare (user embedding), light in the "
                    "comparable Flickr results."))
    s.append(bullet("<b>Context is light</b> — Δd/Δt only; no time-of-day dynamics, queuing, or budgets."))
    s.append(bullet("<b>Markov is a hard baseline</b> — on tiny data, a simple transition model beats the "
                    "neural pointer."))
    s.append(bullet("<b>Single seed; NYC only for Phase 1</b> — multi-seed and TKY runs are planned."))

    # 11. future work
    s.append(section("11. Future work — toward the full Smart Visit Module", 1))
    s.append(bullet("<b>Integrate properly</b> — warm-start / share the engine with the pointer; emit prefix "
                    "sub-trajectories to match supervision density."))
    s.append(bullet("<b>Personalize the itinerary</b> — turn on the pointer's user-embedding lever; report "
                    "personalized-vs-not pairs-F1 on Flickr."))
    s.append(bullet("<b>Richer context</b> — time-of-day / day-of-week dynamic interest, opening hours."))
    s.append(bullet("<b>Add scheduling</b> — explicit time budget + queuing time (the DLIR direction) for "
                    "realistic day plans."))
    s.append(bullet("<b>Robustness</b> — multi-seed, more cities, component ablations."))

    s.append(section("12. In one sentence", 1))
    s.append(callout(
        "<b>A personalized, context-aware next-POI engine, turned into itineraries two ways "
        "(decoupled vs integrated) and honestly benchmarked against the literature on the standard "
        "Flickr datasets — testing, and nuancing, the field's integration hypothesis, with a clear path "
        "toward a full scheduling-aware Smart Visit Module.</b>",
        fill=BG_C, stroke=TEAL))
    return s


def main():
    doc = BaseDocTemplate(OUTPUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2 * cm, bottomMargin=2 * cm,
                          title="Smart Visit Module — the whole thesis, explained")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="default", frames=frame, onPage=on_page)])
    doc.build(build_story())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
