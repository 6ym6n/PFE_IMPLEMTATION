# Next-POI recommendation — TLMR-faithful baseline

A reproducible PyTorch + PyG implementation of a next-POI top-k recommender on Foursquare NYC (Yang et al., TSMC 2015). Built as **path A** from `roadmap.md`: an honest LSTM/STGCN-tier baseline (target HR@1 ∈ [0.13, 0.18]), not a SOTA attempt. Designed to train end-to-end on a free Google Colab T4 in ~1.5–2 hours.

> **TKY scope note.** Tokyo is intentionally excluded from this baseline run. The src/ pipeline is dataset-agnostic (any city `name` works in `preprocess_dataset` / `train_model`); to add TKY later, re-introduce a parallel `df_tky` load → `preprocess_dataset(df_tky, 'TKY')` → `train_model('TKY', ...)` in the notebook.

The architecture, hyperparameters, and pipeline are locked to `implementation_guide.md`. There are no silent improvements.

---

## Architecture

```
                    ┌───────────────┐
poi_ids ───►  GCN(2-layer over hybrid POI graph)  ──► (B, T, d_p)  ─┐
                    └───────────────┘                                │
                                                                     ├─► concat
delta_d ──► MLP_d ─┐                                                 │
delta_t ──► MLP_t ─┴► (B, T, d_c) ───────────────────────────────────┘
                                                                     │
                                                                     ▼
                                                        GRU(d_h) → h_T (B, d_h)
                                                                     │
user_ids ──► nn.Embedding(d_u) ──► e_u (B, d_u) ──► concat ──► [h_T ; e_u]
                                                                     │
                                                                     ▼
                                            MLP head (d_hidden=256) ──► logits (B, |V|)
```

| Hyperparameter | Value |
|---|---|
| POI embedding dim `d_p` | 128 |
| User embedding dim `d_u` | 64 |
| Context dim `d_c` | 32 (16 + 16) |
| GRU hidden dim `d_h` | 128 |
| MLP head hidden dim | 256 |
| GCN layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam, lr=1e-3, wd=1e-5 |
| Batch size | 64 |
| Max epochs | 50 (early stop on val HR@10, patience=8) |
| Grad clip | L2 norm 5.0 |
| Co-visit threshold τ_cov | 3 (training data only) |
| kNN K | 10 (haversine) |
| Filter | ≥10 check-ins/user, ≥10 visitors/POI, iterative |
| Session split gap | 24 h |
| Train/val/test | 70 / 10 / 20 chronological per user |
| Random seed | 42 |

---

## Repo layout

```
PFE_IMPLEMTATION/
├── implementation_guide.md     # master spec
├── roadmap.md                  # week-by-week plan + reference values
├── train_poi.ipynb             # Colab notebook, sections 2–9
├── requirements.txt            # pinned deps
├── README.md                   # you are here
├── src/
│   ├── data/
│   │   ├── preprocess.py       # filter / reindex / sessionize / split
│   │   ├── graph.py            # co-visit ∪ kNN hybrid graph
│   │   └── dataset.py          # POISessionDataset + collate_fn
│   ├── models/
│   │   ├── components.py       # POIGraphEncoder, ContextEncoder
│   │   └── next_poi.py         # NextPOIModel
│   ├── train.py                # training loop + checkpoints
│   └── eval.py                 # HR@k, NDCG@k, MRR
└── tests/                      # pytest: smoke + per-module shape/length tests
```

---

## Running on Colab

1. **Push this repo to GitHub** (or upload `src/` to your Drive).
2. **Open `train_poi.ipynb` in Colab.** Runtime → Change runtime type → **T4 GPU**.
3. **In cell 2.3 ("Make `src/` importable"), set `REPO_URL`** to your fork's URL. The cell will `git clone` and add the repo to `sys.path`. Alternative — uncomment the Drive copy cell and skip the clone.
4. **Run all cells top-to-bottom.** End-to-end timing on free T4 (NYC only):
   - Sections 2–5 (install + download + preprocess + graph): ~3 min
   - Section 6 smoke test: instant
   - Section 7 loader build: ~15 s
   - Section 8 NYC training: ~1.5–2 h
   - Section 9 comparison: instant

   Total: ~2 h. Colab free sessions cap at 12 h — plenty of headroom.

Outputs land on Drive at `/content/drive/MyDrive/poi-rec/`:

```
data/raw/                            # Foursquare .txt files (zip ships both NYC + TKY)
data/processed/NYC/                  # train/val/test parquet, edge_index, meta
checkpoints/NYC/{best.pt,latest.pt}  # best by val HR@10, last epoch
results/NYC_history.csv              # per-epoch train_loss + val metrics
results/NYC_test.json                # final test metrics
```

If a Colab session disconnects mid-training, `latest.pt` lets you resume — see Section 10 of `implementation_guide.md` for the snippet.

---

## Running locally

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Local execution is for tests only — actual training expects GPU and the Drive paths the notebook constructs. The smoke test (`tests/test_next_poi.py::test_smoke_section_6`) is a direct pytest port of Section 6 of the guide; if it passes locally, the model is wired correctly.

---

## Expected results vs literature

Source: LLM4POI Table 3 (HR@1 / Acc@1 on Foursquare NYC).

| Model | NYC HR@1 |
|---|---|
| LSTM | 0.13 |
| STGCN | 0.18 |
| STAN | 0.22 |
| GETNext | 0.24 |
| STHGCN | 0.27 |
| LLM4POI | 0.34 |
| **This baseline (path A, NYC only)** | **target 0.13–0.18** |

A result above ~0.20 with this architecture is suspicious — almost always a chronological-split bug or co-visit edges leaked from val/test. Re-check `chronological_split` and `build_covisit_edges` (training data only).

---

## Module map

| File | Public API |
|---|---|
| `src/data/preprocess.py` | `load_foursquare`, `iterative_filter`, `reindex`, `haversine_km`, `build_sessions`, `chronological_split` |
| `src/data/graph.py` | `build_covisit_edges`, `build_knn_edges`, `build_hybrid_graph` |
| `src/data/dataset.py` | `POISessionDataset`, `collate_fn` |
| `src/models/components.py` | `POIGraphEncoder`, `ContextEncoder` |
| `src/models/next_poi.py` | `NextPOIModel` |
| `src/train.py` | `make_loaders`, `train_one_epoch`, `train_model` |
| `src/eval.py` | `evaluate_model`, `fmt_metrics` |

Every public function has a docstring with shape annotations. No global state — `PROJECT_ROOT`, `DEVICE`, and hyperparameters are passed explicitly.

---

## Tests

```bash
pytest tests/ -v
```

24 tests, all pass:
- `test_components.py` — POIGraphEncoder / ContextEncoder shapes, layer counts, NaN-free zero input
- `test_next_poi.py` — Section-6 smoke test, backward pass, parameter-count band
- `test_dataset.py` — example count, target alignment, left-truncation, padding shapes/dtypes
- `test_graph.py` — co-visit threshold, no self-loops, kNN bounds, PyG edge_index format
- `test_preprocess.py` — haversine values, filter behavior, contiguous reindexing, 24h session split, singleton drop, chronological invariant
- `test_eval.py` — HR/NDCG/MRR on a hand-crafted ranking with known ranks

---

## What's deliberately NOT in this baseline

These are path B from `roadmap.md` and are out of scope for this build:

1. Inner-product scoring instead of MLP-to-|V| head.
2. ST-GRU gates (Δd, Δt routed into GRU instead of side concat).
3. Hour-of-day / day-of-week cyclic features.
4. Weighted co-visit edges (transition frequency as edge weight).

Each of those is a thesis sub-chapter. Don't add them here.

---

## Deviations from the spec

None of substance for the architecture/training spec. Three notes:

- **TKY removed from this run.** The guide trains both NYC and TKY; this build runs NYC only by user request. The `src/` package is dataset-agnostic — adding TKY back is a one-cell change in the notebook.
- The `requirements.txt` pins `pytest==8.1.1` and `pyarrow==15.0.2` (test runner + parquet I/O) which are not in the guide's install cell. They don't affect training behavior.
- The Section-6 smoke test in the guide notes "Total parameters: ~50,000" for `n_pois=100`. The actual count with the locked dimensions (d_p=128, d_h=128, d_hidden=256) is ~234k for that toy size — the head's `Linear(d_h+d_u → d_hidden) + Linear(d_hidden → n_pois)` dominates. The "~50k" figure in the guide appears to be an estimate, not a reference value. The test (`test_param_count_in_expected_band`) checks the order of magnitude rather than an exact number.

If you spot any other deviations, treat the guide as authoritative and file an issue.

---

## Citation

Yang, D., Zhang, D., Zheng, V. W., & Yu, Z. (2015). *Modeling user activity preference by leveraging user spatial temporal characteristics in LBSNs.* IEEE TSMC.
