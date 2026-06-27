# Thesis Kit — everything you need to write the full thesis

**Thesis:** *Smart Visit Module for Personalized Tourist Itinerary Recommendation Based on
User Preferences and Contextual Data.*

This folder gathers **only** the materials useful for *writing* the thesis (design notes,
draftable prose, results, figures, bibliography, decks). The implementation itself stays in
the repo (`src/`, `tests/`, the `presentation/strategy_d/make_*.py` builders) — see
"Regenerating things" below.

> **Everything here is a *copy*.** The live sources are in the repo; if numbers/figures change,
> regenerate (bottom of this file) and re-copy.

---

## How to use this kit
1. Read **`01_framing/thesis_positioning.md`** first — it's the 1-page spine (title → RQs → contributions).
2. Skim **`03_written_documents/thesis_explained.pdf`** (page 2 is a one-page summary of the whole project).
3. Write chapter by chapter using the **outline map** below; pull numbers from the **cheat-sheet**; drop in the **figures** from `05_figures/`.

## Folder structure
| Folder | What's in it |
|---|---|
| `01_framing/` | positioning (RQs/contributions), the load-bearing evaluation methodology, the roadmap |
| `02_design_and_methods/` | the Phase-1 spec + the Phase-2 design document (architectures, decisions, results) |
| `03_written_documents/` | draftable prose PDFs (the report + the plain-language explainers) |
| `04_results_numbers/` | the measured + published numbers (JSON) and the single source of truth (`flickr_results_data.py`) |
| `05_figures/` | ready-to-insert figures (3 charts) |
| `06_bibliography/` | the annotated bibliography (~81 papers) |
| `07_presentation/` | the defence decks (`model_slides_lean.pptx` = recommended) |

---

## ⚠️ Naming — read this before you write
The internal design IDs **Strategy A/B/C/D** were renamed to descriptive names in the
*model-centric* deliverables. **Use the descriptive names in the thesis.** Mapping:

| Internal | Descriptive name (use this) | What it is |
|---|---|---|
| Strategy A | **Frozen Rollout** | decode the frozen next-POI engine (decoupled) — *the headline method* |
| Strategy B | **Trained Pointer** | GCN+pointer trained end-to-end (integrated); v1 = no context, v2 = + context. **Did not beat A.** |
| Strategy C | *(none — never implemented)* | orienteering/ILS reference, scoped only → at most a future-work line |
| Strategy D | **Flickr Benchmark** | re-run on the Flickr datasets, literature-comparable validation |

**Consistency caveat:** `model_explained.pdf` and `model_slides_lean.pptx` already use the
descriptive names. The other files here (`itinerary_plan.md`, `evaluation_methodology.md`,
`poi_baseline_report.pdf`, `thesis_explained.pdf`, `strategy_d_*.pdf`, `thesis_defense_slides.pptx`)
still say "Strategy A/B/D" — translate with the table above as you write.

---

## Suggested thesis outline → which files feed each chapter

| Chapter | Write from | Figures / numbers |
|---|---|---|
| **1. Introduction** (motivation, problem, RQs, contributions) | `01_framing/thesis_positioning.md`; `thesis_explained.pdf` §exec-summary, §1 | — |
| **2. Background & Related Work** (next-POI vs itinerary; OR → deep learning → LLM era; Halder survey & DLIR) | `06_bibliography/bibliography.md`; `strategy_d_primer.pdf`; `thesis_explained.pdf` §2 | — |
| **3. Problem Formulation & Evaluation** (query, F1/pairs-F1, leave-one-out, the comparability rule) | `01_framing/evaluation_methodology.md`; `model_explained.pdf` §2; `strategy_d_explained.pdf` §4–6 | `chart_scale.png` (comparability) |
| **4. Phase 1 — The Next-POI Engine** (GCN + context + user + GRU + head; training; results) | `02_design_and_methods/implementation_guide.md`; `poi_baseline_report.pdf` §1–8 | Phase-1 metrics + tier (cheat-sheet) |
| **5. Phase 2 — Itinerary Recommendation** (Frozen Rollout; Trained Pointer; the integration experiment) | `02_design_and_methods/itinerary_plan.md`; `model_explained.pdf` §7–9; `poi_baseline_report.pdf` §9 | A-vs-B table (cheat-sheet) |
| **6. The Flickr Benchmark** (datasets, Chen-2016 protocol, reproduce Chen, results vs literature) | `strategy_d_explained.pdf`; `itinerary_plan.md` §13; `poi_baseline_report.pdf` §10; `04_results_numbers/` | `chart_results.png`, `chart_validation.png` |
| **7. Discussion** (integration didn't win; personalization; positioning vs Halder/DLIR; what simple baselines tell us) | `thesis_positioning.md`; `model_explained.pdf` §9–11; `thesis_explained.pdf` §5,10 | personalization (cheat-sheet) |
| **8. Limitations & Future Work** | future-work sections of `model_explained.pdf`, `thesis_explained.pdf` | — |
| **9. Conclusion** | `thesis_positioning.md`; `model_explained.pdf` §13 | — |
| **Appendices** (full result tables, reproducibility, code map) | `04_results_numbers/*.json`; repo `CLAUDE.md` | all of `05_figures/` |

---

## Numbers cheat-sheet (authoritative — also in `04_results_numbers/flickr_results_data.py`)

**Phase 1 — next-POI (Foursquare NYC, full vocabulary):**
HR@1 = 0.187 · HR@5 = 0.476 · HR@10 = 0.588 · NDCG@10 = 0.375 · MRR = 0.316.
Tier (HR@1): LSTM 0.130 · STGCN 0.180 · **Ours 0.187** · STAN 0.220 · GETNext 0.240 · STHGCN 0.270 · LLM4POI 0.340.

**Phase 2 — itinerary on NYC (length≥3 test, n = 2,880), pairs-F1:**
Frozen Rollout greedy **0.289** (set-F1 0.609) · beam 0.290 (0.610) · Trained Pointer v1 0.259 · v2 0.261.
→ *Frozen Rollout (decoupled) beats the Trained Pointer (integrated).*

**Flickr datasets (length≥3 = evaluated set):** Toronto 335 · Osaka 47 · Glasgow 112 · Edinburgh 634 · Melbourne 442. (POIs: 29/27/27/28/88.)

**Flickr — our pairs-F1** (Toronto / Osaka / Glasgow / Edinburgh / Melbourne):
Random 0.298 / 0.301 / 0.301 / 0.270 / 0.227 · PoiPopularity 0.443 / 0.413 / 0.510 / 0.439 / 0.320 ·
Markov 0.504 / 0.421 / 0.587 / 0.449 / 0.333 · MarkovPath 0.528 / 0.398 / 0.543 / 0.452 / 0.346 ·
Pointer 0.431 / 0.399 / 0.489 / 0.414 / 0.312.

**Flickr — published pairs-F1** (Toronto / Osaka / Glasgow / Edinburgh):
PoiRank 0.518 / 0.511 / 0.548 / 0.432 · Rank+Markov 0.512 / 0.486 / 0.545 / 0.444 ·
DeepTrip 0.748 / 0.755 / 0.782 / 0.660 · CTLTR 0.748 / 0.719 / 0.763 / 0.681 ·
SelfTrip 0.835 / 0.851 / 0.818 / 0.779 · **AR-Trip 0.839 / 0.828 / 0.820 / 0.808**.

**Validation (faithfulness):** our **Random reproduces Chen 2016** within noise (e.g. Edinburgh 0.270 vs 0.261) → the harness is protocol-faithful.

**Personalization (user-embedding on→off, Flickr pairs-F1):** Osaka 0.442→0.435 (−0.007) · Glasgow 0.451→0.489 (**+0.038**) · Toronto 0.440→0.423 (−0.017). Edinburgh/Melbourne pending the GPU run. → *mixed; helps where users recur, cold-start otherwise.*

---

## Figures in `05_figures/`
- `chart_results.png` — our methods (Random/Markov/Pointer) vs the published SOTA (AR-Trip), per city. *(Ch. 6)*
- `chart_scale.png` — the comparability picture: Flickr literature vs our Flickr vs our NYC scale. *(Ch. 3)*
- `chart_validation.png` — our pairs-F1 vs Chen 2016 (Random on the y = x line). *(Ch. 6)*

## Bibliography
`06_bibliography/bibliography.md` — ~81 annotated papers (✅ = PDF available in the repo's `ARTICLES/`).
Key citations: **Chen 2016** (pairs-F1 + the data), **DeepTrip 2019**, **SelfTrip 2022**, **CTLTR 2021/22**,
**AR-Trip 2024**, **Halder survey 2024**, **DLIR 2025**, **GETNext 2022**, **LLM4POI 2024**.

## Regenerating things (live sources in the repo)
- **Figures:** `py -3.11 presentation/strategy_d/charts.py`
- **Decks / explainer PDFs:** `py -3.11 presentation/strategy_d/make_model_lean_pptx.py` (and the other `make_*.py`)
- **The report:** `py -3.11 report/make_report.py`
- **Flickr numbers (classical, local):** `py -3.11 -m src.flickr.run_flickr --data_dir data/flickr`
- **Neural / personalization (GPU):** `colab_flickr.ipynb`
- All numbers come from `presentation/strategy_d/flickr_results_data.py` (single source of truth).
- Project context for any tooling/AI assistant: repo-root `CLAUDE.md`.

## Writing checklist
- [ ] Use the **descriptive names** (Frozen Rollout / Trained Pointer / Flickr Benchmark) everywhere; translate older docs via the mapping table.
- [ ] Keep next-POI metrics (HR@k) and itinerary metrics (pairs-F1) in **separate tables** — never one leaderboard.
- [ ] State once that the two tasks use different benchmarks (Foursquare vs Flickr) — correct and standard.
- [ ] Lead the contribution with the **engine + Frozen Rollout**; present Trained Pointer + Flickr Benchmark as experiment + validation.
- [ ] Cite Chen 2016 for pairs-F1 and the data; cite Halder/DLIR for the integration debate you engage with.
- [ ] Note honest findings: integration didn't win on NYC; Markov is a strong Flickr baseline; personalization is mixed.
