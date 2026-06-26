"""A plain-English primer to read BEFORE strategy_d_explained.pdf.

Explains, with everyday analogies and tiny worked examples, the handful of ideas
you need before the detailed document: POIs/trajectories, the recommendation
task, precision/recall/F1, pairs-F1, leave-one-out testing, and the
"comparable = same exam" idea — plus a mini-glossary. Run:

    py -3.11 presentation/strategy_d/make_primer.py

Output: presentation/strategy_d/strategy_d_primer.pdf
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "report"))

import make_report as mr  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

S = mr.STYLES
NAVY, INDIGO, TEAL, ORANGE = mr.NAVY, mr.INDIGO, mr.TEAL, mr.ORANGE
GREY_DARK, GREY_MED, GREY_LIGHT, WHITE = mr.GREY_DARK, mr.GREY_MED, mr.GREY_LIGHT, mr.WHITE
BG_G, BG_C, BG_U, BG_S, BG_H = (mr.BG_BOX_GRAPH, mr.BG_BOX_CTX, mr.BG_BOX_USER,
                                mr.BG_BOX_SEQ, mr.BG_BOX_HEAD)
RED = mr.colors.HexColor("#C0392B")

OUTPUT = os.path.join(HERE, "strategy_d_primer.pdf")


def body(t):
    return mr.body(t)


def bullet(t):
    return mr.bullet(t)


def section(t, lvl=1):
    return mr.section(t, lvl)


def callout(html, fill=BG_S, stroke=TEAL, style="Body"):
    """A tinted, bordered box (for the 30-second summary, analogies, examples)."""
    p = Paragraph(html, S[style])
    t = Table([[p]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 1.2, stroke),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# ===========================================================================
# Simple diagrams
# ===========================================================================
class TripStrip(mr.Diagram):
    """A tourist's day: an ordered strip of places."""

    def __init__(self):
        super().__init__(16.4 * cm, 3.0 * cm)

    def draw(self):
        c = self.canv
        places = ["Castle", "Royal Mile", "Holyrood", "Arthur's Seat", "Museum"]
        bw = 2.7 * cm
        gap = 0.5 * cm
        y = 1.0 * cm
        x = 0.2 * cm
        for i, p in enumerate(places):
            fill = BG_U if i == 0 or i == len(places) - 1 else BG_S
            stroke = INDIGO if i == 0 or i == len(places) - 1 else TEAL
            self._box(c, x, y, bw, 1.1 * cm, fill, stroke=stroke, radius=8)
            self._label(c, p, x + bw / 2, y + 0.45 * cm, size=10, color=stroke)
            if i == 0:
                self._label(c, "start", x + bw / 2, y + 1.3 * cm, size=8.5,
                            font="Helvetica-Oblique", color=GREY_MED)
            if i == len(places) - 1:
                self._label(c, "end", x + bw / 2, y + 1.3 * cm, size=8.5,
                            font="Helvetica-Oblique", color=GREY_MED)
            if i < len(places) - 1:
                self._arrow(c, x + bw, y + 0.55 * cm, x + bw + gap - 2, y + 0.55 * cm, color=NAVY)
            x += bw + gap


class PRBoxes(mr.Diagram):
    """Precision/recall as 'what you recommended' vs 'what they visited'."""

    def __init__(self):
        super().__init__(16.4 * cm, 5.0 * cm)

    def draw(self):
        c = self.canv
        # two lists
        self._box(c, 0.2 * cm, 2.7 * cm, 6.6 * cm, 1.7 * cm, BG_H, stroke=ORANGE)
        self._label(c, "You recommended (3):", 3.5 * cm, 3.9 * cm, size=10, color=ORANGE)
        self._label(c, "Castle · Park · Museum", 3.5 * cm, 3.3 * cm, size=10.5, font="Helvetica",
                    color=GREY_DARK)
        self._box(c, 9.6 * cm, 2.7 * cm, 6.6 * cm, 1.7 * cm, BG_S, stroke=TEAL)
        self._label(c, "They actually visited (3):", 12.9 * cm, 3.9 * cm, size=10, color=TEAL)
        self._label(c, "Castle · Museum · Garden", 12.9 * cm, 3.3 * cm, size=10.5, font="Helvetica",
                    color=GREY_DARK)
        # overlap
        self._box(c, 5.7 * cm, 1.2 * cm, 5.0 * cm, 1.1 * cm, BG_G, stroke=NAVY)
        self._label(c, "in BOTH = correct:  Castle, Museum  → 2", 8.2 * cm, 1.6 * cm, size=10,
                    color=NAVY)
        self._arrow(c, 4.0 * cm, 2.7 * cm, 6.5 * cm, 2.3 * cm, color=GREY_MED, line_width=0.8)
        self._arrow(c, 12.4 * cm, 2.7 * cm, 9.9 * cm, 2.3 * cm, color=GREY_MED, line_width=0.8)
        # math
        self._label(c, "precision = 2 of 3 you guessed = 0.67     "
                       "recall = 2 of 3 they visited = 0.67     F1 = 0.67",
                    8.2 * cm, 0.5 * cm, size=10.5, color=NAVY)


class OrderSwap(mr.Diagram):
    """Same places, swapped order: point-F1 = 1.0 but pairs-F1 < 1."""

    def __init__(self):
        super().__init__(16.4 * cm, 4.2 * cm)

    def draw(self):
        c = self.canv

        def row(label, seq, y, color, fill, swap_idx=()):
            self._label(c, label, 0.2 * cm, y + 0.3 * cm, size=10, color=color, anchor="left",
                        font="Helvetica-Bold")
            for i, p in enumerate(seq):
                x = 4.2 * cm + i * 1.5 * cm
                hl = i in swap_idx
                self._box(c, x, y, 1.2 * cm, 0.8 * cm, BG_C if hl else fill,
                          stroke=ORANGE if hl else color, radius=5)
                self._label(c, p, x + 0.6 * cm, y + 0.28 * cm, size=11, color=ORANGE if hl else color)
                if i < len(seq) - 1:
                    self._arrow(c, x + 1.2 * cm, y + 0.4 * cm, x + 1.5 * cm - 2, y + 0.4 * cm,
                                color=GREY_MED, line_width=0.8)

        row("the real trip:", ["A", "B", "C", "D"], 3.1 * cm, TEAL, BG_S)
        row("our guess:", ["A", "C", "B", "D"], 1.9 * cm, INDIGO, BG_U, swap_idx=(1, 2))
        self._label(c, "Same 4 places → point-F1 = 1.0 (we picked them all).",
                    0.2 * cm, 1.0 * cm, size=10, color=GREY_DARK, anchor="left", font="Helvetica")
        self._label(c, "But B and C are swapped → the order is wrong → pairs-F1 = 0.83 (not 1.0).",
                    0.2 * cm, 0.45 * cm, size=10, color=NAVY, anchor="left", font="Helvetica-Bold")


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
        c.drawString(2 * cm, A4[1] - 1.1 * cm, "Read me first — a plain-English primer")
        c.drawRightString(A4[0] - 2 * cm, A4[1] - 1.1 * cm, "Strategy D")
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
    s.append(Paragraph("Read Me First", S["Title"]))
    s.append(Paragraph("A plain-English primer for the Strategy D document", S["Subtitle"]))
    s.append(Spacer(1, 0.6 * cm))
    s.append(Paragraph("The handful of ideas — with simple examples — you need before reading "
                       "<i>strategy_d_explained.pdf</i>", S["Subtitle"]))
    s.append(Spacer(1, 2.4 * cm))
    s.append(callout(
        "<b>The 30-second version.</b> We build a system that suggests a <b>tour route</b> "
        "through a city's attractions. We measure how good a route is with a score called "
        "<b>pairs-F1</b>. Our score on one dataset looked low next to the research papers — but "
        "that was like comparing exam marks from <i>different exams</i>. We re-ran our system on "
        "the <i>same dataset and rules</i> the papers use, so now the scores are finally "
        "comparable (and they look fine). This primer explains every word in that sentence.",
        fill=BG_G, stroke=NAVY))
    s.append(PageBreak())

    # 1
    s.append(section("1. Places and trips", 1))
    s.append(body(
        "A <b>POI</b> (Point Of Interest) is just <b>a place worth visiting</b> — a castle, a "
        "museum, a park. A <b>trajectory</b> (we also say a <b>trip</b> or <b>tour</b>) is the "
        "<b>ordered list of places one tourist actually visited</b>, in the order they visited "
        "them. For example, a real day in Edinburgh:"
    ))
    s.append(TripStrip())
    s.append(body(
        "These trips are reconstructed from people's <b>geo-tagged photos</b>: if someone "
        "photographs the Castle at 10am and the Museum at 2pm, we know they went Castle → "
        "Museum. Each place is visited once (no loops). That is all a &quot;dataset&quot; of "
        "trajectories is: thousands of these little ordered lists, one per tourist per day."
    ))

    # 2
    s.append(section("2. The task: recommend a route", 1))
    s.append(body(
        "The job is: <b>given where a tourist starts, where they want to end, and how many stops "
        "they want, suggest the places in between — in the right order.</b> For example:"
    ))
    s.append(callout(
        "<b>Input (the &quot;query&quot;):</b> start = Castle, end = Museum, I want 4 stops total.<br/>"
        "<b>Output (our recommendation):</b> Castle → ? → ? → Museum, where we fill in the two "
        "middle places, in order.<br/>"
        "<b>How we check it:</b> we compare our route to what the tourist <i>actually</i> did.",
        fill=BG_U, stroke=INDIGO))
    s.append(body(
        "Think of it like Google Maps, but instead of choosing which <i>roads</i> to take, we "
        "choose which <i>attractions</i> to visit and in what order. The start and end are given "
        "to us, so the only real challenge is the <b>middle, and its order</b>."
    ))

    # 3
    s.append(section("3. Two tasks — don't mix them up", 1))
    s.append(body("The thesis has two related but different jobs. It matters which one we are talking about:"))
    s.append(bullet("<b>Next-POI prediction</b> — predict only the <b>single next place</b> "
                    "someone will go. (Like autocomplete: you've been to the Castle, what's next?) "
                    "This is Phase 1 of the thesis."))
    s.append(bullet("<b>Itinerary recommendation</b> — predict the <b>whole ordered route</b> at "
                    "once. (The full day plan.) <b>This is the work the document is about.</b>"))
    s.append(body(
        "They use different datasets and different scores, and that is normal and correct — the "
        "research field treats them as two separate problems. The only mistake would be to "
        "compare a score from one against a score from the other."
    ))

    # 4
    s.append(section("4. Scoring a guess: precision, recall, F1", 1))
    s.append(body(
        "Before order, the simplest question is: <b>did we pick the right places?</b> Two ideas "
        "answer this. Suppose we recommend 3 places and the tourist actually visited 3 places, "
        "with 2 in common:"
    ))
    s.append(PRBoxes())
    s.append(bullet("<b>Precision</b> = of the places <i>we recommended</i>, how many were right? "
                    "Here 2 of 3 = 0.67."))
    s.append(bullet("<b>Recall</b> = of the places they <i>actually visited</i>, how many did we "
                    "find? Here 2 of 3 = 0.67."))
    s.append(bullet("<b>F1</b> = a single number that balances precision and recall (their "
                    "&quot;harmonic mean&quot;). Here F1 = 0.67. F1 = 1.0 means a perfect set of "
                    "places; 0.0 means none right."))
    s.append(body(
        "In the document this is called <b>point-F1</b> or <b>set-F1</b>. The key limitation: "
        "<b>it ignores order completely</b> — which is a problem, because a tour is all about order."
    ))

    # 5
    s.append(section("5. Why order matters: pairs-F1", 1))
    s.append(body(
        "A route that visits the right places <i>in the wrong order</i> is still a bad route. So "
        "the main score, <b>pairs-F1</b> (from the Chen 2016 paper), checks the <b>order</b>. The "
        "trick: for every pair of places, ask &quot;did X come before Y in both routes?&quot; "
        "Here is the same example with letters (A = Castle, B = Park, C = Museum, D = Garden):"
    ))
    s.append(OrderSwap())
    s.append(body(
        "Both routes contain the same four places, so <b>point-F1 = 1.0</b>. But because B and C "
        "are swapped, one ordered pair is wrong, so <b>pairs-F1 = 0.83</b>. That is the whole "
        "point of pairs-F1: it rewards getting the <b>order</b> right, not just the places. "
        "pairs-F1 also ranges from 0 (everything backwards) to 1 (perfect)."
    ))
    s.append(callout(
        "<b>Remember just this:</b> &nbsp; point-F1 = &quot;right places?&quot; &nbsp;·&nbsp; "
        "pairs-F1 = &quot;right places <i>and</i> right order?&quot; &nbsp; pairs-F1 is the number "
        "everyone reports and compares.",
        fill=BG_C, stroke=ORANGE))

    # 6
    s.append(section("6. Testing fairly: leave-one-out", 1))
    s.append(body(
        "These datasets are small (a city may have only ~50–600 example trips). To test honestly "
        "we use <b>leave-one-out cross-validation</b>: to test on one trip, we let the model learn "
        "from <b>all the other trips</b>, then ask it to reconstruct the one we held back — which "
        "it never saw during learning. We do this once for <i>every</i> trip and average the scores."
    ))
    s.append(callout(
        "<b>Analogy.</b> It is like quizzing yourself with flashcards: to test card #1 fairly, you "
        "study all the <i>other</i> cards first, then try card #1. Then you repeat for card #2, and "
        "so on. You never &quot;study the answer&quot; to the card you are about to be tested on — "
        "that would be cheating (in machine learning we call it <i>leakage</i>).",
        fill=BG_S, stroke=TEAL))

    # 7
    s.append(section("7. The big idea: comparable means &quot;same exam&quot;", 1))
    s.append(body(
        "This is the heart of the whole document. <b>You can only compare two scores if they were "
        "measured the same way.</b> Two students score 60% and 80% — is the second one better? "
        "Only if they sat the <b>same exam</b>. If 60% was on a brutally hard exam and 80% on an "
        "easy one, the comparison is meaningless."
    ))
    s.append(body(
        "That is exactly what happened to us. Our itinerary system was first measured on a "
        "<b>Foursquare-New-York</b> dataset — a huge, hard &quot;exam&quot; with about "
        "<b>5,000 places</b> — and scored pairs-F1 ≈ 0.27. The research papers measured on the "
        "small <b>Flickr</b> city datasets — an easier &quot;exam&quot; with about <b>30 "
        "places</b> — and scored ≈ 0.7. <b>Different exams, so the 0.27 and the 0.7 cannot be "
        "compared.</b>"
    ))
    s.append(callout(
        "<b>What &quot;Strategy D&quot; does:</b> it re-runs our system on the <b>exact same "
        "datasets and exact same rules</b> the papers use (the Flickr cities, leave-one-out, "
        "endpoints given, pairs-F1). Now our scores and theirs are on the <b>same exam</b> — "
        "finally comparable. (And as a sanity check, our simplest method reproduces the paper's "
        "number almost exactly, which proves we are really taking the same exam.)",
        fill=BG_G, stroke=NAVY))

    # 8 glossary
    s.append(section("8. Mini-glossary (every term you'll meet)", 1))
    glos = [
        ["Term", "In plain words"],
        ["POI", "a place worth visiting (museum, park, castle)"],
        ["Trajectory / trip / tour", "the ordered places one tourist visited"],
        ["Itinerary recommendation", "suggest the whole ordered route (this work)"],
        ["Next-POI prediction", "predict only the single next place (Phase 1)"],
        ["point-F1 / set-F1", "did we pick the right places? (ignores order)"],
        ["pairs-F1", "did we get the order right too? (the main score)"],
        ["Leave-one-out CV", "test on one trip, learn from all the others, repeat"],
        ["Leakage", "accidentally letting the test answer into training (cheating)"],
        ["Baseline", "a simple method we compare against"],
        ["Random", "the simplest baseline: order the places at random (the floor)"],
        ["PoiPopularity", "just recommend the most-visited places"],
        ["Markov", "use “which place usually comes after which”"],
        ["GCN", "a neural network that learns from the map/graph of places"],
        ["Pointer network", "a neural network that outputs an ordered sequence"],
        ["SOTA", "“state of the art” — the current best published method"],
        ["Foursquare (NYC)", "big check-in dataset (~5,000 places) — Phase 1 / 2A / 2B"],
        ["Flickr cities", "small photo-trip datasets (~30 places) — the itinerary benchmark"],
        ["Chen 2016, DeepTrip, SelfTrip, AR-Trip", "published methods we compare our numbers to"],
    ]
    s.append(mr.make_table(glos, col_widths=[5.6 * cm, 10.8 * cm]))

    # 9 ready
    s.append(section("9. You're ready — what the detailed document covers", 1))
    s.append(body(
        "With the above, <i>strategy_d_explained.pdf</i> will read smoothly. Here is its map:"
    ))
    s.append(bullet("<b>§1–2 Context &amp; the comparability problem</b> — the &quot;different "
                    "exam&quot; idea, in full."))
    s.append(bullet("<b>§3–4 Datasets &amp; the task</b> — exactly which files and queries we use."))
    s.append(bullet("<b>§5–6 Metric &amp; protocol</b> — pairs-F1 and leave-one-out, formally."))
    s.append(bullet("<b>§7 Methods</b> — the simple baselines and the neural GCN+pointer model."))
    s.append(bullet("<b>§8 Validation</b> — how we proved we &quot;sat the same exam&quot; "
                    "(reproducing Chen 2016)."))
    s.append(bullet("<b>§9–10 Results</b> — our numbers next to the published ones, and an honest "
                    "look at where the neural model still falls short."))
    s.append(bullet("<b>§11–12 Limitations &amp; how to run</b> — and an appendix with the code map."))
    s.append(Spacer(1, 0.3 * cm))
    s.append(callout(
        "<b>One sentence to keep in your head:</b> &quot;We measured the same tour-recommendation "
        "task on the field's standard small datasets with the field's exact rules, so our pairs-F1 "
        "is now on the same scale as the published methods — and our simplest baseline reproduces "
        "the reference paper, proving the comparison is fair.&quot;",
        fill=BG_C, stroke=TEAL))
    return s


def main():
    doc = BaseDocTemplate(OUTPUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2 * cm, bottomMargin=2 * cm,
                          title="Read me first — Strategy D primer")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="default", frames=frame, onPage=on_page)])
    doc.build(build_story())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
