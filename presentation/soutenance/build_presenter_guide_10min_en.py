"""Generate the English 10-minute presenter guide from deck speaker-note JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#0B1F33")
NAVY2 = colors.HexColor("#153B5B")
TEAL = colors.HexColor("#1FA6A0")
TEAL_DARK = colors.HexColor("#137B77")
TEAL_PALE = colors.HexColor("#DDF3F1")
ORANGE = colors.HexColor("#F28C45")
ORANGE_PALE = colors.HexColor("#FCE8D9")
INK = colors.HexColor("#142632")
MUTED = colors.HexColor("#5A6C79")
LIGHT = colors.HexColor("#DCE4E9")
PALE = colors.HexColor("#F4F6F8")


QA = [
    (
        "What is the main contribution of this thesis?",
        "A reproducible context-aware next-POI engine, Frozen Rollout for converting its scores into loop-free routes without retraining, and a Flickr Benchmark that places the itinerary results on a literature-comparable scale. The contribution is the system, the integration experiment, and the rigorous evaluation - not a new state-of-the-art architecture.",
    ),
    (
        "Why treat next-POI prediction and itinerary generation as separate tasks?",
        "Next-POI prediction ranks one future place, while itinerary generation must select and order an entire route. They therefore use different outputs and metrics: HR@k, NDCG, and MRR for next-POI; set-F1 and pairs-F1 for itineraries.",
    ),
    (
        "What does HR@1 = 0.187 actually mean, and why use a GRU?",
        "It means the correct next venue is the top prediction in 18.7 percent of cases among about 5,100 candidates; it is not an itinerary-success rate. The GRU provides a lean, reproducible sequential backbone and produces an honest LSTM/STGCN-tier result.",
    ),
    (
        "How does Frozen Rollout generate an itinerary?",
        "It repeatedly applies the frozen next-POI scorer to the current partial route. Already visited POIs are masked, the requested end POI is reserved for the final step, and greedy or beam selection continues until the requested route length is reached.",
    ),
    (
        "Why did Frozen Rollout outperform the Trained Pointer in this setup?",
        "The strongest explanation is supervision density: the engine learns from roughly 10^5 prefix-target pairs, whereas the Trained Pointer receives one whole-trajectory example per session. Frozen Rollout reuses a converged engine, while the Trained Pointer trains its encoder from scratch. Adding context improves the Trained Pointer only from 0.259 to 0.261.",
    ),
    (
        "How do you know the Trained Pointer result is not a training bug?",
        "Validation pairs-F1 rose to a genuine peak before early stopping, and the context-enhanced variant was tested. The same data-size pattern appears on the Flickr Benchmark, where the Trained Pointer trails Markov. This supports a real negative result for the tested setup, not a universal claim about every itinerary model.",
    ),
    (
        "Why does beam search barely improve Frozen Rollout?",
        "Beam search changes pairs-F1 only from 0.289 to 0.290. The much higher set-F1 of 0.609 shows that many correct places are found but their order remains imperfect. Searching harder over the same local scores cannot introduce information the scorer never learned.",
    ),
    (
        "Why can the NYC and Flickr results not be compared directly?",
        "NYC contains about 5,100 POIs, while the Flickr cities contain only 27 to 88. Chance level and task difficulty differ dramatically. Valid comparison requires the same dataset, protocol, and metric.",
    ),
    (
        "How did you validate the Flickr evaluation harness?",
        "I followed Chen 2016: leave-one-trajectory-out evaluation, first and last POIs plus route length given, routes of length at least three, loop-free output, and pairs-F1. Our Random results differ from Chen by at most 0.021 across five cities.",
    ),
    (
        "Does personalization actually help?",
        "Conditionally. The user embedding improves Glasgow from 0.451 to 0.489, a +0.038 gain, but changes Osaka by -0.007 and Toronto by -0.017. Under leave-one-out, many users have too little remaining history to estimate an identity embedding reliably.",
    ),
    (
        "What are the main limitations?",
        "The engine remains below neural and LLM state of the art; decoding cannot correct the scorer's myopia; context is limited to distance and elapsed-time gaps; decode-time elapsed time is assumed; and the next-POI experiment covers one city with offline evaluation only - no deployment or user study.",
    ),
    (
        "What is future work, and what was not implemented?",
        "Future work includes weather and time-of-day context, persona or content-based cold-start representations, schedule-aware decoding with travel cost and opening hours, stronger pre-training, multi-city validation, and user studies. The orienteering or iterated-local-search solver was scoped but not implemented.",
    ),
]


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=18 * mm,
            bottomMargin=16 * mm,
            title="10-Minute English PFE Defense - Presenter Guide",
            author="Ayman Naaimi",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates(PageTemplate(id="guide", frames=[frame], onPage=self._draw_page))

    def _draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 10 * mm, A4[0], 10 * mm, fill=1, stroke=0)
        canvas.setFillColor(ORANGE)
        canvas.rect(0, A4[1] - 10 * mm, 34 * mm, 2 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 9 * mm, "Smart Visit · English PFE Defense · Ayman Naaimi")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GuideTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24,
            leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11,
            leading=15, textColor=MUTED, spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18,
            leading=22, textColor=NAVY, spaceBefore=6, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            leading=16, textColor=TEAL_DARK, spaceBefore=5, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
            leading=14, textColor=INK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5,
            leading=11, textColor=INK,
        ),
        "memory": ParagraphStyle(
            "Memory", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10,
            leading=14, textColor=NAVY,
        ),
        "cue": ParagraphStyle(
            "Cue", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=9,
            leading=12, textColor=MUTED,
        ),
        "anchor": ParagraphStyle(
            "Anchor", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=17,
            leading=20, textColor=colors.white, alignment=TA_CENTER,
        ),
        "anchor_desc": ParagraphStyle(
            "AnchorDesc", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5,
            leading=11, textColor=colors.white, alignment=TA_CENTER,
        ),
    }


def memory_page(story, st, notes):
    story.append(Paragraph("10-Minute English PFE Defense", st["title"]))
    story.append(Paragraph("Presenter memory sheet · target delivery 9:15 · 45-second safety margin", st["subtitle"]))

    thesis_box = Table([[Paragraph(
        "<b>One-sentence thesis</b><br/>I built a personalized, context-aware next-POI engine, decoded it without retraining into loop-free itineraries, and evaluated each task on its own fair benchmark.",
        st["body"],
    )]], colWidths=[178 * mm])
    thesis_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL_PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([thesis_box, Spacer(1, 8)])

    anchors = Table([
        [Paragraph("0.187", st["anchor"]), Paragraph("0.289 &gt; 0.259", st["anchor"]), Paragraph("0.021", st["anchor"])],
        [
            Paragraph("HR@1 among ~5,100 NYC candidates", st["anchor_desc"]),
            Paragraph("Frozen Rollout beats Trained Pointer", st["anchor_desc"]),
            Paragraph("maximum Random gap vs Chen 2016", st["anchor_desc"]),
        ],
    ], colWidths=[59.3 * mm] * 3, rowHeights=[13 * mm, 15 * mm])
    anchors.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), TEAL_DARK),
        ("BACKGROUND", (1, 0), (1, -1), ORANGE),
        ("BACKGROUND", (2, 0), (2, -1), NAVY2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.white),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
    ]))
    story.extend([anchors, Spacer(1, 8)])

    story.append(Paragraph("The rule to memorize", st["h2"]))
    rule = Table([[Paragraph(
        "<b>Compare results only when the dataset, protocol, and metric are identical.</b><br/>Never compare next-POI HR@k with itinerary pairs-F1, and never compare NYC itinerary scores directly with Flickr literature scores.",
        st["body"],
    )]], colWidths=[178 * mm])
    rule.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ORANGE_PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, ORANGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([rule, Spacer(1, 8)])

    flow = [
        [str(index), item["time"], item["memory"]]
        for index, item in enumerate(notes, start=1)
    ]
    data = [["Slide", "Time", "Memory cue"]] + flow
    table = Table(data, colWidths=[18 * mm, 22 * mm, 138 * mm], rowHeights=[7 * mm] + [6.2 * mm] * 10)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.4, LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 7)])
    story.append(Paragraph(
        "<b>If running late:</b> use each slide's 'Skip if late' line; if needed, compress slides 5 and 9, but keep the state-of-the-art gap and the comparability rule. Safe closing: <i>In this tested setting, decoding a densely supervised next-POI engine outperformed a dedicated itinerary model.</i>",
        st["small"],
    ))
    story.append(PageBreak())


def script_section(story, st, notes):
    story.append(Paragraph("Full 9:15 Talk Track", st["title"]))
    story.append(Paragraph(
        "Pronunciation: say 'HR at one', 'HR at ten', and 'pairs F-one'. Each slide includes one sentence to memorize, a transition, and a safe sentence to skip.",
        st["subtitle"],
    ))
    for index, item in enumerate(notes, start=1):
        block = [
            Paragraph(f"Slide {index} · {esc(item['time'])} · {esc(item['title'])}", st["h1"]),
            Table([[Paragraph(f"<b>MEMORY:</b> {esc(item['memory'])}", st["memory"]) ]], colWidths=[178 * mm], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), TEAL_PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])),
            Spacer(1, 5),
            Paragraph(esc(item["script"]), st["body"]),
            Paragraph(f"<b>Transition:</b> {esc(item['transition'])}", st["cue"]),
            Paragraph(f"<b>Skip if late:</b> {esc(item['skip'])}", st["cue"]),
            Spacer(1, 7),
        ]
        story.append(KeepTogether(block))
    story.append(PageBreak())


def qa_section(story, st):
    story.append(Paragraph("Likely Jury Questions", st["title"]))
    story.append(Paragraph("Answer directly, give one piece of evidence, and stop. Return to the closing slide after the answer.", st["subtitle"]))
    for index, (question, answer) in enumerate(QA, start=1):
        story.append(KeepTogether([
            Paragraph(f"{index}. {esc(question)}", st["h2"]),
            Paragraph(esc(answer), st["body"]),
        ]))
    story.append(Spacer(1, 8))
    warning = Table([[Paragraph(
        "<b>Never claim:</b> state of the art; that HR@1 and pairs-F1 are comparable; that NYC and Flickr have the same difficulty; that personalization always helps; or that the orienteering / ILS method was implemented.",
        st["body"],
    )]], colWidths=[178 * mm])
    warning.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ORANGE_PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, ORANGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(warning)


def build_guide(notes_path: Path, output_path: Path) -> None:
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    if len(notes) != 10:
        raise ValueError(f"Expected 10 main-slide notes, found {len(notes)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = GuideDoc(str(output_path))
    st = styles()
    story = []
    memory_page(story, st, notes)
    script_section(story, st, notes)
    qa_section(story, st)
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_guide(args.notes.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
