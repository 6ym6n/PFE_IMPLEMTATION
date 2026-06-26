# Itinerary Recommendation — Design Document

**Phase 2 of the thesis: from next-POI prediction to itinerary recommendation.**

This document specifies how the existing next-POI baseline (GCN + context + user + GRU + MLP head,
trained on Foursquare NYC) is extended to *itinerary recommendation* — producing an **ordered route of
multiple POIs** given a query (start POI, optional end POI, and a budget). It follows the same
honest-baseline ethos as the next-POI work: maximize reuse, minimize new complexity, surface every
assumption, and report results comparable to the literature.

The design was selected from a 4-strategy design panel (see commit history). The chosen path is:

> **Strategy A (frozen rollout) first → Strategy B (learned pointer decoder) second.**

---

## 1. Problem statement

**Next-POI (Phase 1):** given a session prefix and a user, predict the single next POI.

**Itinerary (Phase 2):** given a *query* `q = (user u, start POI s, end POI e, budget)`, produce an
**ordered sequence** of POIs `Ŷ = (s, p₂, p₃, …, e)` that (a) starts at `s`, (b) ends at `e` (if given),
(c) contains no repeats (loop-free), and (d) respects the budget. Quality is measured by how well `Ŷ`
recovers the user's ground-truth visited sequence `Y`, using order-aware metrics (**pairs-F1**).

This mirrors the standard PersTour / Chen-2016 evaluation convention: *recover the visited sequence given
its start, end, and length/time budget.*

---

## 2. Core idea — an itinerary is a decoded next-POI model

The Phase-1 model already scores, for any partial route, a probability distribution over the next POI.
An itinerary is what you get by **rolling that scorer out** under two constraints the model never saw at
training time:

1. **No-revisit mask** — set the logit of every already-visited POI to `-∞` (loop-free; Menon 2017,
   AR-Trip 2024).
2. **Budget stop** — stop when the budget is exhausted; reserve the final hop for the fixed end POI.

```
query (u, s, e, K)
   │
   ▼
route = [s] ──┐
   │          │  repeat until len(route) == K:
   │          │     logits = model(route, u)            # next-POI scorer (frozen)
   │          │     logits[visited] = -inf              # loop-free
   │          │     next = e        if last step        # reserve end
   │          │            argmax(logits) otherwise     # greedy  (or beam)
   │          │     route.append(next)
   └──────────┘
   ▼
Ŷ = ordered route   ──► pairs-F1 vs ground-truth session
```

**Nothing in the trained model changes.** The same `best.pt` from Phase 1 serves both tasks. The whole
itinerary capability is an inference-only decoder plus an evaluator.

---

## 3. Budget modes

| Mode | Stop rule | Constants needed | Status |
|---|---|---|---|
| **Length `K`** (default) | stop when route has `K` stops | **none** | headline |
| Time `B` (secondary) | stop when accumulated travel+visit time would exceed `B` | travel speed + per-POI dwell | sensitivity only |

**Length budget is the default and the headline.** `K` = the ground-truth session length, read directly
from `test.parquet`. It needs no invented dwell/speed constants, so the headline numbers are fully
reproducible on the locked Foursquare-NYC data. The time budget needs a travel-speed and a per-POI dwell
constant (TSMC2014 has neither opening hours nor service times), so it is reported only as a documented
secondary sensitivity mode, never as the headline.

### Decode-time model inputs

The model consumes `Δd` (km) and `Δt` (h) per step. At decode time:

- **`Δd`** = real haversine distance between consecutive route POIs (from `poi_coords.npy`), clipped to
  `[0, 100]` km exactly as `preprocess.build_sessions` does. This is real geometry — no assumption.
- **`Δt`** = a single documented constant `assumed_dt_hours` (default `1.0`, clipped to `[0, 24]`). The
  context encoder is a minor input (`d_c = 32` vs `d_p = 128`), and in **length-budget mode this constant
  does not affect the stop rule** (which counts stops), only the context features. It is configurable and
  ablatable.

---

## 4. Fixed-end handling (reserve-the-return-leg)

When the query fixes an end POI `e`, we guarantee it lands last:

- **Length mode:** at the final step (`step == K-1`) force `next = e` (the discrete analogue of the
  orienteering start/end constraint in PersTour). If `e` was already visited (degenerate), fall back to
  the best unvisited POI.
- This is a **heuristic**: it guarantees `e` is last but can produce an abrupt final jump and does not
  guarantee a globally optimal route. We report the abrupt-jump / feasibility rate rather than claiming
  clean termination.

---

## 5. Decoding strategies

- **Greedy** (default baseline): pick `argmax` of the masked logits each step. Honest, deterministic,
  fast. This is the floor.
- **Beam(b):** keep the top-`b` partial routes ranked by summed transition log-probability; return the
  highest-scoring complete route. Less myopic, still touches nothing in the model. `beam=1` is exactly
  greedy (used as a correctness test).

---

## 6. Evaluation

Itinerary metrics are reported in a **separate table** from the strict full-vocab ranking metrics
(HR@k / NDCG@k / MRR). The two protocols are never mixed — the Phase-1 reproducibility depends on it.

### pairs-F1 (Chen 2016, primary metric)

For a route `T`, its ordered-pair set is `P(T) = { (Tᵢ, Tⱼ) : i < j }`. Then:

```
pairs-precision = |P(Ŷ) ∩ P(Y)| / |P(Ŷ)|
pairs-recall    = |P(Ŷ) ∩ P(Y)| / |P(Y)|
pairs-F1        = 2·prec·rec / (prec + rec)
```

A pair `(a, b)` is correct iff both POIs appear in both routes **and** in the same order. In length-budget
mode `|Ŷ| = |Y| = K`, so the denominators are equal and pairs-F1 = the fraction of correctly-ordered
pairs. Start and end are *given*, so a perfect recovery scores 1.0 and a reversed middle scores `< 1.0`.

### Supporting metrics

- **set-F1** (order-agnostic): F1 over the *set* of visited POIs — measures "did we pick the right places"
  ignoring order.
- **exact-match rate**: fraction of routes identical to ground truth.
- **feasibility rate**: fraction of routes that are loop-free and within budget (1.0 by construction in
  length mode; reported as a guard).

### The length≥3 caveat (critical)

NYC sessions average ~5.2 check-ins but **many are length 2** → trivial `start→end` queries with zero
ordering signal (the decoder is forced to output exactly `[s, e]` = ground truth, scoring 1.0 for free).
**Pairs-F1 only has dynamic range on the length≥3 subset.** Every result is therefore reported twice:
over **all** sessions and over the **length≥3** subset. The length≥3 numbers are the meaningful ones.

### pairs-F1 must be pinned with unit tests

Because all four strategies (and the cited papers) report pairs-F1, a subtly different definition makes
every comparison invalid. The implementation is locked with hand-checked tests:
`perfect = 1.0`, `reversed-middle < 1.0`, disjoint-middle lower, length-1 edge case defined.

---

## 7. Honest limitations (write these in the thesis — they justify Phase-2 Strategy B)

- **Myopia / exposure bias.** The model was trained one-step with teacher forcing; at rollout it
  conditions on its own (possibly wrong) prefix and never saw the budget, the fixed end, or the
  no-revisit constraint. This is exactly what an honest *floor* should expose.
- **Decode-time `Δt` is assumed**, not observed (length mode bounds its impact to the context features).
- **Fixed-end heuristic** can cause an abrupt final hop.
- **Thin ordering signal on NYC** (length-2-dominated) — quantified by this baseline, and the deciding
  factor for whether Strategy B's training run is worthwhile.

---

## 8. Module design (`src/itinerary/`)

Nothing in the existing Phase-1 code changes. New, inference-only package:

```
src/itinerary/
├── __init__.py             # exports
├── query.py                # ItineraryQuery + build_eval_queries(test_df)
├── decode.py               # rollout_greedy(), rollout_beam(), score helper (GCN cached once)
├── eval_itinerary.py       # pairs_f1, set_f1, evaluate_itinerary, fmt_itinerary_metrics
└── run_itinerary.py        # load best.pt → build queries → decode → write results JSON
tests/test_itinerary.py     # pairs-F1 + decode-invariant tests
```

**Performance note (taken from the start):** the GCN output is input-independent (see `notes.md`), so
`poi_features = model.gcn(edge_index)` is computed **once** and reused across all decode steps and all
queries, instead of calling `model.forward` verbatim each step. Identical outputs, `O(route_len × n_queries)`
fewer GCN passes.

### Key signatures

```python
@dataclass
class ItineraryQuery:
    user_idx: int
    start_poi: int
    end_poi: int | None
    K: int                      # length budget = ground-truth session length
    ground_truth: list[int]     # the actual visited session, for scoring

def build_eval_queries(test_df, *, fixed_end=True) -> list[ItineraryQuery]: ...

def rollout_greedy(model, query, edge_index, poi_coords, device,
                   assumed_dt_hours=1.0, poi_features=None) -> list[int]: ...
def rollout_beam(model, query, edge_index, poi_coords, device, beam=3,
                 assumed_dt_hours=1.0, poi_features=None) -> list[int]: ...

def pairs_f1(pred: list[int], truth: list[int]) -> float: ...
def set_f1(pred: list[int], truth: list[int]) -> float: ...
def evaluate_itinerary(model, queries, edge_index, poi_coords, device,
                       decoder="greedy", beam=3, assumed_dt_hours=1.0,
                       min_len=1) -> dict: ...
```

---

## 9. Roadmap after this baseline

- **Strategy B (next rung):** replace the MLP head with a pointer/attention decoder, trained end-to-end
  on observed trajectories with sequence cross-entropy. Reuses this baseline's `eval_itinerary.py`,
  visited-mask/decode logic, and query builder. Its main risk (training-data shrinkage to one example per
  length≥3 session) is precisely what this Strategy-A floor measures first.
- **Strategy C (OR reference):** feed the learned scores as profits into an Orienteering-Problem / ILS
  solver — a complementary upper reference, not part of the neural line.
- **Strategy D (Flickr chapter):** Chen-style structured prediction on the Flickr photo-trajectory
  datasets (Toronto/Osaka/Glasgow/Edinburgh) — a separate dataset, a later research chapter.

---

## 10. What to run (Colab)

Prerequisite: Phase-1 already produced `checkpoints/NYC/best.pt` and `data/processed/NYC/` on Drive.
Then (Section 10 of `train_poi.ipynb`):

```python
from src.itinerary.run_itinerary import run_itinerary
res = run_itinerary("NYC", project_root=PROJECT_ROOT, device=DEVICE)   # greedy + beam, length budget
```

It loads the frozen `best.pt`, builds one query per test session, decodes greedy and beam routes, and
writes `results/NYC_itinerary_{greedy,beam}.json` with pairs-F1 / set-F1 / exact-match over **all** and
**length≥3** sessions. No training, ~minutes on a T4.

### Strategy-A measured result (NYC, frozen rollout)

| Decoder | pairs-F1 (len≥3) | set-F1 | exact | note |
|---|---|---|---|---|
| greedy | 0.2887 | 0.6089 | 0.054 | the floor |
| beam(3) | 0.2902 | 0.6101 | 0.057 | +0.5% over greedy |

Beam barely beats greedy → the next-POI scorer is myopic; **decoding cannot fix it**. The set-F1≫pairs-F1
gap (right places, wrong order) is the headroom, and it motivates Strategy B.

---

## 11. Strategy B — pointer network (implemented scaffold)

A **trained** sequence decoder, not a decoded frozen model. Reuses the GCN encoder + user embedding;
replaces the MLP head with an **inner-product pointer** (the design panel's #1 priority for path B).

**Architecture** (`src/itinerary/pointer_model.py`, `PointerItineraryModel`):
- Encoder: `H = GCN(edge_index)` (POI features, |V|×d_p) + user embedding `e_u`.
- Initial decoder state: `h_0 = tanh(W_init [H_start ; H_end ; e_u])` — the query (start, end, user) seeds it.
- Decoder: a GRU rolled over the trajectory; input at step t is the previous POI's GCN feature.
- **Pointer scoring**: `logits_v = ⟨ W_o[h_t ; e_u], H_v ⟩` — inner product of the projected decoder state
  against every POI feature. Ties the output directly to the graph signal (vs. a free MLP-to-|V| head).
- Training: teacher-forced sequence cross-entropy over **whole** length≥3 trajectories
  (`ItinerarySeqDataset`: one example per session, not L−1 prefixes). No visited-mask in the loss
  (Foursquare sessions can revisit; masking the target would NaN); the mask is applied only at decode.
- Inference: greedy / beam rollout with the same loop-free + reserved-fixed-end discipline as Strategy A
  (`pointer_rollout_greedy` / `pointer_rollout_beam`).
- Early stopping on **val pairs-F1** (not loss).

**Modules**
```
src/itinerary/seq_dataset.py    ItinerarySeqDataset, seq_collate_fn   (whole-session examples)
src/itinerary/pointer_model.py  PointerItineraryModel, pointer_rollout_{greedy,beam}, evaluate_pointer
src/itinerary/train_pointer.py  make_seq_loaders, train_one_epoch_pointer, train_pointer_model
tests/test_pointer.py           dataset, forward+backward, decode invariants, overfit-sanity
```

**What to run (Colab, Section 11 of `train_poi.ipynb`)** — needs the same `data/processed/NYC/` as Phase-1;
trains a *new* model (does NOT touch `checkpoints/NYC/best.pt`):
```python
from src.itinerary.train_pointer import train_pointer_model
model, test_greedy, history = train_pointer_model(
    "NYC", project_root=PROJECT_ROOT, device=DEVICE, epochs=50, patience=8, beam=3)
```
Writes `checkpoints/NYC_pointer/{best,latest}.pt` and `results/NYC_pointer_test.json`
(test pairs-F1 greedy + beam, length≥3). Compare its pairs-F1 against the Strategy-A floor above.

**Data sizes (measured, NYC):** one whole-trajectory example per length≥3 session →
**10,281 train / 1,093 val / 2,880 test** sessions. (An earlier draft of this doc wrongly said the
training set was "2,880" — that 2,880 is the *test* count; training is ~5× larger.)

---

## 12. Measured results — A vs B (NYC, length≥3 test, n=2,880)

| Method | pairs-F1 | set-F1 | exact | vs floor |
|---|---|---|---|---|
| **Strategy A** — frozen next-POI rollout (greedy) | **0.2887** | 0.6089 | 0.054 | — (floor) |
| Strategy A — beam(3) | 0.2902 | 0.6101 | 0.057 | +0.0015 |
| **Strategy B-v1** — pointer, no context (greedy/beam3) | **0.2585** | 0.578 | 0.043 | **−0.0302** |

**The trained B-v1 pointer LOST to the frozen baseline by 0.030 pairs-F1 (−10.5%).** It trained cleanly
(loss 8.01→3.41 over 36 epochs; val pairs-F1 0.16→peak 0.271 at epoch 28; early-stop ep 36) — this is a
genuine negative result, not a bug.

**Why A beats a purpose-trained B-v1 (the lesson):**
1. **Supervision density.** A's engine (the next-POI model) was trained on ~75k prefix→next examples
   (every prefix, all lengths). B-v1 saw 10,281 whole trajectories — far fewer, sparser updates.
2. **Leaner inputs.** B-v1's decoder input is *only* the previous POI's GCN feature; it never sees the
   Δd/Δt context that the next-POI model consumes at every step.
3. **From-scratch vs. converged.** A reuses a fully-trained model; B-v1 trains GCN+GRU+pointer from random init.

Takeaway for the thesis: *naively training a dedicated itinerary model does not automatically beat
cleverly decoding a strong next-POI model* — the next-POI model is a hard baseline because of its much
denser supervision.

### Strategy B-v2 — context-aware pointer (implemented; closes capability gap #2)

`PointerItineraryModel(use_context=True)` adds the Δd/Δt context (the Phase-1 `ContextEncoder`) to the
decoder input at every step — the spatial/temporal signal B-v1 lacked. At decode time the context of the
current position is Δd from `poi_coords` (real geometry) + an assumed constant Δt (same convention as
Strategy A). Trains via `train_pointer_model(..., use_context=True)` → writes
`checkpoints/NYC_pointer_ctx/` and `results/NYC_pointer_ctx_test.json`. **Run it and compare to both the
floor and B-v1** to test whether the context features close the −0.03 gap.

Further levers if B-v2 still trails (ranked): warm-start B's GCN from the trained next-POI GCN; emit
prefix sub-trajectories to match A's data density; try an MLP-to-|V| head instead of pure inner product.

---

## 13. Strategy D — Flickr datasets (literature-comparable pairs-F1)

**Why.** Strategies A/B score pairs-F1 ≈ 0.26–0.29 on Foursquare NYC, but that is **not comparable** to
the published trip-recommendation numbers (~0.6–0.8), because the literature is evaluated on the small
Flickr photo-trajectory datasets under a *different* protocol. Strategy D re-runs the itinerary task on
those datasets, under the **exact** published protocol, so the numbers land on the **same scale** as the
papers. (The low NYC numbers were a property of that data + protocol — *not* a bug.)

**Data (`src/flickr/data.py`).** The canonical `traj-{City}.csv` (`userID,trajID,poiID,startTime,…`) +
`poi-{City}.csv` (`poiID,poiCat,poiLon,poiLat`) files — Chen 2016's own preprocessing output, mirrored in
the `computationalmedia/tour-cikm16` and `gcooq/DeepTrip` repos. A trajectory = the POIs of one `trajID`
ordered by `startTime` (consecutive duplicates collapsed). Using these files means **our trajectories are
identical to the literature's**, removing all preprocessing ambiguity. The loader also reads Lim's raw
`userVisits-{City}.csv` / `POI-{City}.csv` (grouping by `seqID`) for completeness. Every trajectory is
loop-free (`trajLen == #rows == #distinct POIs` for all rows), so the Chen pairs-F1 assumptions hold.

| City | #POIs | #users | #trajectories | **#traj (len ≥ 3) = eval set** |
|---|---|---|---|---|
| Toronto | 29 | 1,395 | 6,057 | **335** |
| Osaka | 27 | 450 | 1,115 | **47** |
| Glasgow | 27 | 601 | 2,227 | **112** |
| Edinburgh | 28 | 1,454 | 5,028 | **634** |
| Melbourne | 88 | 1,000 | 5,106 | **442** |

(Chen 2016's Table 1 reports the *total* trajectory counts — 6,057 / 1,115 / 2,227 / 5,028 / 5,106 — which
match ours exactly; evaluation is on the length ≥ 3 subset.)

**Protocol (`src/flickr/evaluate.py`), matched to the papers exactly:**
- **leave-one-trajectory-out** cross-validation: refit per fold on the *other length ≥ 3 trajectories*
  (`train_pool="ge3"`, the default — matching the papers' folds; the held-out trajectory never leaks).
  Training on *all* other trajectories incl. length < 2 (`train_pool="all"`) is supported but is a
  documented deviation (it adds extra popularity/transition signal the papers do not see);
- the query gives the **first and last POI** (origin + destination) and the desired length `K`;
- evaluate only **length ≥ 3**, **loop-free** trajectories with distinct endpoints (Chen's `calc_pairsF1`
  assumption; 0 dropped on the Flickr data, which is entirely loop-free);
- metrics: point-**F1** (`set_f1`) and order-aware **pairs-F1**, *reusing the unit-tested
  `src/itinerary/eval_itinerary.py`* — pinned against an independent re-implementation of Chen 2016's
  reference `calc_pairsF1` in `tests/test_flickr.py`.

**Protocol-faithfulness check (the honesty proof).** A `Random` baseline has no modelling choices, so it
must match Chen's `Random` if (and only if) data + protocol + metric are identical. They do, within noise:

| pairs-F1 | Toronto | Osaka | Glasgow | Edinburgh | Melbourne |
|---|---|---|---|---|---|
| **Random — ours** | 0.298 | 0.301 | 0.301 | 0.270 | 0.227 |
| Random — Chen 2016 | 0.310 | 0.304 | 0.320 | 0.261 | 0.248 |
| **PoiPopularity — ours** | 0.443 | 0.413 | 0.510 | 0.439 | 0.320 |
| PoiPopularity — Chen 2016 | 0.384 | 0.365 | 0.507 | 0.436 | 0.316 |

(PoiPopularity matches Chen near-exactly on Glasgow 0.510 vs 0.507, Edinburgh 0.439 vs 0.436, Melbourne
0.320 vs 0.316; the small Toronto/Osaka overshoot reflects our visit-count popularity vs Chen's exact rank.)

**Our measured results (classical baselines, leave-one-out, length ≥ 3, real local CPU run):**

| pairs-F1 | Toronto | Osaka | Glasgow | Edinburgh | Melbourne |
|---|---|---|---|---|---|
| Random | 0.298 | 0.301 | 0.301 | 0.270 | 0.227 |
| PoiPopularity | 0.443 | 0.413 | 0.510 | 0.439 | 0.320 |
| Markov (greedy) | 0.504 | 0.421 | 0.587 | 0.449 | 0.333 |
| MarkovPath (beam) | 0.528 | 0.398 | 0.543 | 0.452 | 0.346 |
| **F1 (point)** | | | | | |
| Random | 0.611 | 0.613 | 0.620 | 0.581 | 0.534 |
| PoiPopularity | 0.714 | 0.698 | 0.747 | 0.704 | 0.619 |
| Markov / MarkovPath | 0.737/0.758 | 0.685/0.681 | 0.786/0.762 | 0.702/0.705 | 0.619/0.628 |

The headline: **our pairs-F1 is 0.23–0.59 — on the published 0.26–0.85 scale**, versus 0.26–0.29 on NYC.
Goal achieved with classical baselines alone; the learned model targets the neural band above. (MarkovPath
beam-searches for the max-transition-likelihood path, which maximises *likelihood*, not pairs-F1 — so it
slightly trails greedy Markov on Osaka, an honest artefact of the surrogate objective.)

**Published comparison (directly-comparable subset, `src/flickr/published.py`).** Same protocol, 0–1 scale:

| pairs-F1 (published) | Toronto | Osaka | Glasgow | Edinburgh |
|---|---|---|---|---|
| PoiRank (Chen 2016) | 0.518 | 0.511 | 0.548 | 0.432 |
| Rank+Markov (Chen 2016) | 0.512 | 0.486 | 0.545 | 0.444 |
| DeepTrip (2019) | 0.748* | 0.755 | 0.782 | 0.660 |
| SelfTrip (2022) | 0.835 | 0.851 | 0.818 | 0.779 |
| CTLTR (via AR-Trip 2024) | 0.748 | 0.719 | 0.763 | 0.681 |
| AR-Trip (2024) | **0.839** | 0.828 | **0.820** | **0.808** |

(Bold = best pairs-F1 in that column. AR-Trip leads on Toronto/Glasgow/Edinburgh, but **SelfTrip is best on
Osaka (0.851 > AR-Trip's 0.828)** — there is no single SOTA across all cities on pairs-F1. *DeepTrip's own
paper reports only Edin/Glas/Osaka; the Toronto value is from SelfTrip's reproduction. Melbourne is reported
almost only by the classical line; best there is Rank+Markov 0.351.)

**Honest deviations / caveats (write these in the thesis):**
- **Markov ≠ Chen's Markov.** Ours is a raw empirical first-order transition matrix (Laplace-smoothed);
  Chen's is a *feature-factored* Markov. Ours therefore differs and on the data-rich cities (Toronto,
  Glasgow) *exceeds* Chen's published Markov — a modelling difference, not a protocol bug. `Random` and
  `PoiPopularity` are the clean reproductions.
- **PoiRank / Rank+Markov are cited, not re-implemented** (they need a per-query rankSVM); their numbers
  come straight from Chen 2016.
- **Not comparable, excluded from the headline:** POIBERT (80/20 chronological split, percentage-scale F1,
  no pairs-F1), DLIR 2025 (8-hour session split, different F1 ≈ 0.49), TourEmbedding (no pairs-F1,
  percentage-scale F1, query gives only the *first* POI not first+last, no leave-one-out). See
  `NON_COMPARABLE_NOTES` in `published.py`.

**Modules (`src/flickr/`).**
```
src/flickr/
├── data.py        FlickrCity, Trajectory, load_city (traj-/poi- and userVisits-/POI-), trajectories_min_len
├── evaluate.py    make_query, loocv_splits, evaluate_loocv  (F1 + pairs-F1, leave-one-out)
├── baselines.py   Random / PoiPopularity / Markov(greedy) / MarkovPath(beam) + factories
├── pointer.py     FlickrPointerNet (self-contained GCN + GRU pointer), per-fold trainer/decoder (Colab)
├── published.py   curated directly-comparable literature pairs-F1 / F1 + NON_COMPARABLE_NOTES
└── run_flickr.py  run_classical_baselines / run_neural / format_comparison + CLI
tests/test_flickr.py   loader, trajectory construction, pairs-F1 vs Chen reference, LOO splitter, baselines
```

**What to run.**
- *Locally (CPU, classical — already produces real numbers):*
  `py -3.11 -m src.flickr.run_flickr --data_dir data/flickr` (download CSVs first; see the notebook).
- *Colab (GPU, the learned pointer):* open **`colab_flickr.ipynb`** — it clones the repo, downloads the
  five cities, runs the classical baselines, then trains the GCN+pointer per leave-one-out fold and prints
  the combined comparison vs the published numbers. No `torch_geometric` needed (the GCN is self-contained).
