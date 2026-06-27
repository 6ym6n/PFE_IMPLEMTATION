# Thesis — Contribution & Positioning (1-page)

**Title:** *Smart Visit Module for Personalized Tourist Itinerary Recommendation Based on User Preferences and Contextual Data*

## Positioning (the one paragraph)
Tourists need a **personalized, context-aware itinerary** — an *ordered route* of POIs — not just the
single next place. The field's survey (Halder et al., *ASOC 2024*) treats itinerary recommendation as
distinct from next-POI prediction, and the recent DLIR model (Halder et al., *ACM TORS 2025*) argues that
handling **POI scoring and itinerary construction as *separate* problems is sub-optimal**, in favour of an
**integrated** model. This thesis builds a personalized, context-aware itinerary recommender whose
**engine** is a next-POI scoring model (graph + spatio-temporal context + user embedding), and then **tests
the integration question directly**: it compares *decoding the frozen engine* (decoupled) against *training
a dedicated end-to-end itinerary model* (integrated), and validates both against the published literature
under the exact comparable protocol. The next-POI model is therefore **not a separate deliverable** — it is
the personalized, contextual *engine* of the itinerary module named in the title.

## Research questions
- **RQ1** — Can a personalized, context-aware next-POI model (graph + sequence + user) reach a competitive
  tier on a standard check-in benchmark?
- **RQ2** — Can that engine be decoded into coherent, loop-free itineraries *without* dedicated itinerary
  training (the decoupled approach)?
- **RQ3** — Does an *integrated*, end-to-end itinerary model improve over the decoupled decoding — i.e. does
  the integration hypothesis (Halder/DLIR) hold on our data?
- **RQ4** — Where do our itinerary models stand against the published literature, under the *exact*
  comparable protocol?

## How the work maps to the RQs
| Component | Answers | What it is |
|---|---|---|
| **Phase 1** — next-POI engine (GCN + Δd/Δt context + user embedding; Foursquare NYC) | RQ1 | the personalized, context-aware **engine** |
| **Strategy A** — frozen rollout | RQ2 | **decoupled** decoding of the engine into itineraries |
| **Strategy B** — pointer (v1/v2), trained end-to-end | RQ3 (vs A) | the **integrated** itinerary model |
| **Strategy D** — Flickr benchmark (Chen-2016 protocol, pairs-F1) | RQ4 | **literature-comparable** validation |

## Contributions
1. A reproducible, leakage-free **personalized context-aware next-POI engine** (LSTM/STGCN tier on
   Foursquare NYC).
2. A **controlled decoupled-vs-integrated comparison** (Strategy A vs B) that directly tests the
   integration hypothesis.
3. A **literature-comparable Flickr benchmark** (Strategy D) with a pairs-F1 harness *validated by
   reproducing Chen 2016* (Random / PoiPopularity within noise).
4. An **honest analysis** of where simple baselines suffice and where integration / personalization help —
   and a clear path toward a full scheduling-aware module.

## How it engages Halder / DLIR
- Halder's 2024 survey defines the personalized-itinerary field; **DLIR (2025) argues integration >
  separation** and adds *dynamic temporal interest*, *queuing time*, and *time-budget scheduling*.
- We **test the integration claim directly** and report a *nuanced* result: on our data the integrated
  pointer did **not** automatically beat the decoupled rollout (a supervision-density effect), and on the
  standard Flickr benchmark a simple **Markov** baseline beats the neural model. This **nuances** Halder/DLIR
  rather than contradicting it.
- We **scope deliberately smaller** than DLIR — modelling user preference (embedding) and spatial-temporal
  context (Δd/Δt) but *not* queuing, explicit time budgets, or time-of-day dynamics — and frame those as the
  path toward a full DLIR-style Smart Visit Module.

## Key results (honest)
- **Phase 1 (RQ1):** HR@1 = 0.187 (HR@10 = 0.588, MRR = 0.316) — LSTM/STGCN tier of the LLM4POI table.
- **Strategy A vs B (RQ3; NYC, length ≥ 3):** A (frozen) ≈ **0.290** pairs-F1 > B (pointer) ≈ **0.259–0.261**
  — integration did **not** win (the engine's far denser supervision wins).
- **Strategy D (RQ4; Flickr):** Random reproduces Chen 2016 (validation); our classical pairs-F1 **0.23–0.59**
  and learned pointer **0.31–0.49** sit on the published **0.26–0.85** scale — with Markov the strongest of
  ours and the neural SOTA (AR-Trip / SelfTrip ≈ 0.8) above.

---
*Caveats to state in the thesis: single seed; NYC only for Phase 1; personalization is strong on Foursquare
but light in the comparable Flickr results; context is Δd/Δt only. All are addressed in Future Work.*
