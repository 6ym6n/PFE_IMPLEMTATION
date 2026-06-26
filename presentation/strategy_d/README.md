# Strategy D — supervisor presentation

Deliverables for presenting the Flickr itinerary-recommendation work (Strategy D):

| File | What it is |
|---|---|
| **`strategy_d_slides.pptx`** | The talk — 17 widescreen slides, editable in PowerPoint/Google Slides. Native shapes for the diagrams; the three analytical charts are images. |
| **`strategy_d_explained.pdf`** | The detailed companion document (9 pages) — explains data, task, metric, protocol, methods, validation, results, honest limitations, with worked examples. |

Both are generated from one source of truth, so the deck and the document never disagree.

## Rebuild

```bash
pip install python-pptx matplotlib reportlab        # build-only deps
py -3.11 presentation/strategy_d/make_pdf.py         # -> strategy_d_explained.pdf
py -3.11 presentation/strategy_d/make_pptx.py        # -> strategy_d_slides.pptx
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
