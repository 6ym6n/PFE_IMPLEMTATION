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

**Honest risk (measured by Strategy A first):** training examples shrink to one per length≥3 session —
**2,880 on NYC**. That is a small set for a sequence model, so watch for overfitting (the val-pairs-F1
early stop guards it). If B does not clearly beat the 0.29 floor, that thinness is the likely reason and
is itself a reportable finding.
