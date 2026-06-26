# Evaluation methodology — read before changing anything in the thesis

This is the **load-bearing decision record** for how the two tasks in this thesis are
evaluated and compared. If you change a dataset, a metric, a split, or a results table,
re-read this first so the comparisons stay valid (and defensible at the viva).

---

## TL;DR

This thesis has **two distinct tasks**, each evaluated on **its own field-standard benchmark**:

| Task | Dataset | Metrics | Compare against |
|---|---|---|---|
| **Next-POI recommendation** (Phase 1) | **Foursquare** NYC/TKY (+ Gowalla) | **Acc@k / HR@k, NDCG@k, MRR** (full vocab) | GETNext, STAN, STHGCN, LLM4POI |
| **Itinerary / tour recommendation** (Phase 2) | **Flickr** photo-trajectory cities (Toronto, Osaka, Glasgow, Edinburgh, Melbourne, …) | **set-F1, pairs-F1** | Chen 2016, DeepTrip, SelfTrip, CTLTR, DeepAltTrip |

**Using a different dataset per task is correct and standard — NOT a flaw.** The only
flaw would be **cross-comparing** the two (e.g. an itinerary pairs-F1 vs a next-POI
Acc@1, or scores across cities with different POI-vocabulary sizes).

---

## Why this is legitimate (the precedent — cite these)

Next-POI and itinerary are **formally distinct sub-tasks**, and the two field-defining
surveys say so explicitly:

- **Lim et al., KAIS 2019** ("Tour Recommendation and Trip Planning … A Survey") — its
  taxonomy puts *Next-POI Recommendation* and *Tour Itinerary Recommendation* as
  **separate** leaf nodes, and separates the data sources: LBSN check-ins
  (Foursquare/Gowalla) for next-POI vs. geo-tagged Flickr photo-trajectories for itineraries.
- **Halder et al., ASOC 2024** (itinerary survey) — same separation.
- **Chen et al., CIKM 2016** enumerates POI / next-location / trajectory recommendation
  as three distinct settings *in one paper*, then evaluates its trajectory task on Flickr.

Each task's leading works use exactly this split:
- **Next-POI on Foursquare/Gowalla:** GETNext (SIGIR 2022), STHGCN (SIGIR 2023),
  STAN (WWW 2021), LLM4POI (SIGIR 2024) — Acc@k / MRR.
- **Itinerary on Flickr:** Chen 2016 (origin of pairs-F1), DeepTrip (SIGSPATIAL 2019),
  SelfTrip (KBS 2022), CTLTR (TIST 2022), DeepAltTrip (TKDE 2021) — F1 / pairs-F1.

So a single work evaluating each task on its own benchmark is well-precedented.

---

## Rules to keep comparisons valid (do these, or a reviewer will object)

1. **Two separate results chapters/tables**, each with its own baselines. No merged
   leaderboard. State once that the two tasks' metrics are **not on a common scale**.
2. **Compare only within identical dataset + split + query + metric.**
   - Itinerary: replicate **Chen 2016** exactly — leave-one-trajectory-out CV,
     query = first POI + last POI + desired length, length ≥ 3 loop-free trajectories,
     report **set-F1 and pairs-F1** on the 0–1 scale.
   - Next-POI: **full-vocabulary** Acc@k, **no** sampled negatives / candidate
     restriction (matches GETNext/STHGCN/LLM4POI). Already done in Phase 1.
3. **Never cross-compare** pairs-F1 vs Acc@1/HR@k, and **never** compare absolute scores
   across cities/datasets with different POI-vocabulary sizes (different chance levels).
   Absolute gaps across datasets are data/protocol artifacts, not model superiority.
   → cite **Krichene & Rendle, KDD 2020** (sampled/cross-protocol metrics aren't consistent).
4. **Pin each metric with a unit test** and reproduce a Random/Popularity baseline within
   noise of Chen 2016's published numbers — proves the pairs-F1 implementation matches
   theirs. (`src/itinerary/eval_itinerary.py::pairs_f1` is already unit-tested.)
5. Keep an explicit **"excluded from headline comparison"** note listing any competitor
   dropped for a different split/vocab/metric/query (so the exclusion is principled, not
   cherry-picking).

---

## Paste-ready methodology sentence (for the thesis)

> Because next-POI recommendation and itinerary recommendation are formally distinct
> sub-tasks with distinct canonical benchmarks — the former evaluated on Foursquare/Gowalla
> check-ins with Acc@k/MRR and the latter on Flickr photo-trajectory city datasets with
> set-F1 and pairs-F1 — we evaluate each task on its own task-appropriate benchmark and
> protocol (following GETNext (SIGIR 2022) and Chen et al. (CIKM 2016) respectively), and
> report results strictly within each task; the two tasks' metrics are not on a common
> scale and are never cross-compared.

---

## Current measured results (snapshot — update as runs land)

**Phase 1 — next-POI (Foursquare NYC, full vocab):** HR@1 = **0.187** (HR@5 0.476,
HR@10 0.588, NDCG@10 0.375, MRR 0.316). Sits in the LSTM/STGCN tier of LLM4POI Table 3
(LSTM 0.13, STGCN 0.18, STAN 0.22, GETNext 0.24, STHGCN 0.27). ✅ directly comparable.

**Phase 2 — itinerary (Foursquare NYC, len≥3 test, n=2,880):**

| Method | pairs-F1 (greedy) | best |
|---|---|---|
| Strategy A — frozen rollout | 0.2887 | 0.2902 (beam3) |
| Strategy B-v1 — pointer, no context | 0.2585 | 0.2585 |
| Strategy B-v2 — pointer + context | 0.2579 | 0.2610 (beam3) |

⚠️ **These NYC itinerary numbers are NOT comparable to the itinerary literature** (which
reports pairs-F1 ≈ 0.6–0.8 on Flickr cities of ~25–90 POIs). The gap is a
dataset/vocabulary artifact (5,135 POIs vs ~30), not model quality. They are valid only
as an **internal A-vs-B study** on identical data.

**To get literature-comparable itinerary numbers → Strategy D:** run on the Flickr
datasets with Chen-2016 protocol (see `itinerary_plan.md` §11–12; data linked in
`articles/README.md` §9). Alternatively, to compare to a *Foursquare* itinerary paper,
replicate that one paper's exact POI-filtering + trajectory protocol (e.g. ESWA 2024
"interaction-based augmented data") — there is no shared Foursquare itinerary benchmark.

---

## Citations (most PDFs are in `articles/`)

- Lim, Chan, Karunasekera, Leckie. *Tour Recommendation and Trip Planning using LBSN: A Survey.* KAIS 2019.
- Halder, Lim, Chan, Zhang. *A Survey on Personalized Itinerary Recommendation.* Applied Soft Computing 2024.
- Chen, Ong, Xie. *Learning Points and Routes to Recommend Trajectories.* CIKM 2016 — defines set-F1 + pairs-F1.
- Yang, Liu, Zhao. *GETNext.* SIGIR 2022 — Foursquare NYC/TKY + Gowalla; Acc@k/MRR.
- Gao et al. *DeepTrip.* SIGSPATIAL 2019. · Gao et al. *SelfTrip.* KBS 2022. · Rashid et al. *DeepAltTrip.* TKDE 2021.
- Li et al. *LLM4POI.* SIGIR 2024 — Foursquare + Gowalla; Acc@1.
- Krichene, Rendle. *On Sampled Metrics for Item Recommendation.* KDD 2020 — grounds the no-cross-protocol rule.
- Bellogín et al. *POI Recommendation: Pitfalls and Viable Solutions.* 2025 — inconsistent splits/metrics make results incomparable.
