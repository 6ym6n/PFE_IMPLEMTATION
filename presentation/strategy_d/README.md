# Strategy D — supervisor presentation

Deliverables for the Flickr itinerary-recommendation work (Strategy D).

**Suggested reading order:** `strategy_d_primer.pdf` → `strategy_d_explained.pdf` → `strategy_d_slides.pptx`.

| File | What it is |
|---|---|
| **`strategy_d_primer.pdf`** | **Read first.** A plain-English primer (5 pages) — the few ideas you need before the detailed doc (POIs/trips, the task, precision/recall/F1, pairs-F1, leave-one-out, the "same exam" idea) with everyday analogies, worked examples and a mini-glossary. |
| **`strategy_d_explained.pdf`** | The detailed companion document (9 pages) — data, task, metric, protocol, methods, validation, results, honest limitations, with worked examples and the full published comparison. |
| **`strategy_d_slides.pptx`** | The Strategy-D talk — 17 widescreen slides, editable in PowerPoint/Google Slides. Native shapes for the diagrams; the three analytical charts are images. |
| **`thesis_defense_slides.pptx`** | The **full-thesis** defence deck (18 slides) — the whole arc (motivation → Phase-1 engine → Strategy A/B/D → Halder/DLIR positioning → results → future work). Editable, native diagrams. |
| **`thesis_explained.pdf`** | The **full-thesis** companion document (9 pages) — the whole project explained in plain language with diagrams and worked examples (page 2 is a one-page "whole idea" summary). |
| **`model_slides.pptx`** | The **model-centric** deck (14 slides) — leads with the next-POI engine as "the model" and Strategy A as the itinerary method; B/D appear only as supporting experiment/validation. The recommended deck to present to the jury. |

The thesis 1-page positioning statement lives at repo root: **`../../thesis_positioning.md`**
(title → research questions → how A/B/D map → how it engages Halder/DLIR). Drop it into the thesis intro.

All are generated from one source of truth, so they never disagree.

## Rebuild

```bash
pip install python-pptx matplotlib reportlab        # build-only deps
py -3.11 presentation/strategy_d/make_primer.py      # -> strategy_d_primer.pdf
py -3.11 presentation/strategy_d/make_pdf.py         # -> strategy_d_explained.pdf
py -3.11 presentation/strategy_d/make_pptx.py        # -> strategy_d_slides.pptx
py -3.11 presentation/strategy_d/make_thesis_pptx.py # -> thesis_defense_slides.pptx
```

`make_pdf.py` reuses the thesis report's styling (`report/make_report.py`); `make_pptx.py`
builds editable native-shape diagrams. Both call `charts.make_charts()` to (re)generate the
PNGs in `assets/`.

## Source files

- `flickr_results_data.py` — all numbers (dataset stats, our results, published comparison, the worked pairs-F1 example). Edit here to update both deliverables.
- `charts.py` — the three matplotlib charts (scale, results, validation).
- `make_pdf.py` — the detailed PDF (reportlab + custom vector diagrams).
- `make_pptx.py` — the slide deck (python-pptx).
- `assets/` — generated chart PNGs.

All numbers trace back to `src/flickr/` (our measured results) and `src/flickr/published.py`
(the curated literature). The measured pointer numbers are from the Colab GPU run; rerun
`colab_flickr.ipynb` and update `flickr_results_data.py` to refresh them.
