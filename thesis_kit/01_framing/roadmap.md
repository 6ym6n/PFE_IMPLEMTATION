# Next-POI recommendation — implementation roadmap

**Path:** (A) Educational / baseline build, TLMR-faithful adaptation for next-POI top-k.
**Datasets:** Foursquare NYC + TKY.
**Target:** HR@1 in 0.13–0.18 band (LSTM-to-STGCN tier). Honest baseline, not SOTA.
**Timeline:** 10 weeks at ~6–10 hours/week. Compressible to 8, expandable to 12.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Framework | PyTorch 2.x | Every comparable paper in the field releases code in PyTorch |
| GNN library | PyTorch Geometric (PyG) | Battle-tested `GCNConv`, sparse adjacency handling |
| Compute | Google Colab (free T4) | Foursquare scale fits comfortably; ~1–2h per training run |
| Storage | Google Drive (mounted) | Colab disk resets every session — Drive persists checkpoints + processed data |
| Tracking | TensorBoard (built-in) or Weights & Biases free tier | Loss + metric curves, hyperparameter logs |
| Other | pandas, numpy, scikit-learn, tqdm | Data wrangling and progress bars |

**Project structure on Drive:**
```
poi-rec/
├── data/
│   ├── raw/                    # Foursquare NYC + TKY downloads
│   └── processed/              # filtered + tokenized + split tensors
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_graph.ipynb
│   ├── 04_train.ipynb
│   └── 05_evaluate.ipynb
├── src/                         # Reusable modules
│   ├── models/
│   ├── data/
│   └── eval/
├── checkpoints/                # .pt files per epoch
└── results/                    # CSV + plots
```

---

## Week-by-week plan

### Week 1 — Setup + dataset acquisition

**Goal:** working Colab notebook with raw data loaded.

- Create the Colab project, mount Drive at `/content/drive/MyDrive/poi-rec/`
- Pin versions: `torch==2.3`, `torch-geometric==2.5`, `pandas`, `numpy`, `scikit-learn`, `tqdm`
- Download Foursquare NYC + TKY. Original source: Yang et al., "Modeling user activity preference by leveraging user spatial temporal characteristics in LBSNs", IEEE TSMC 2015. Mirrors are in the GETNext or STHGCN GitHub repos if the original link is dead.
- Run basic stats: number of users, POIs, check-ins, timestamp range, average sequence length
- Plot a histogram of check-ins per user — confirms the long-tail shape

**Deliverable:** `01_data.ipynb` with raw data and stats printed.

**Risk:** the Yang dataset URL has moved over the years. Check the GETNext repo (`songyang-cs/GETNext`) — they redistribute it.

---

### Week 2 — Preprocessing

**Goal:** clean train/val/test splits saved to Drive.

- Filter: users with ≥10 check-ins, POIs visited by ≥10 users (standard)
- Re-index user IDs and POI IDs to `0..N-1` — critical, your embedding tables depend on this
- Build sessions: split each user's history at temporal gaps > 24 hours
- Compute Δd (haversine, in km) and Δt (in hours) per consecutive pair
- Chronological split per user: 70% train / 10% val / 20% test (NOT random — POI rec leaks badly with random splits)
- Save processed tensors to Drive as `.pt` files

**Deliverable:** `02_preprocess.ipynb` + saved tensors + a one-page "data card" (final user/POI counts, average sequence length, train/val/test sizes).

**Pitfall:** if you re-index after splitting, your val/test will have POIs unseen during training and crash on embedding lookup. Re-index BEFORE splitting.

---

### Week 3 — Hybrid POI graph

**Goal:** sparse adjacency saved to Drive.

- Co-visit edges: from **training data only** (no leakage), add edge `(p_i, p_{i+1})` if at least τ_cov = 3 users transitioned this way
- Geographic kNN: K=10 nearest POIs by haversine distance — `sklearn.neighbors.BallTree` with the haversine metric does this in ~1 second
- Union both edge sets, store as a `torch_geometric.data.Data` object with `edge_index` of shape `(2, num_edges)`
- Sanity check: plot the degree distribution. Most POIs should have 5–30 neighbors. If you have isolated nodes, your GCN will silently skip them.

**Deliverable:** `03_graph.ipynb` + `hybrid_graph.pt`

**Pitfall:** building the graph from train+val+test combined is **leakage** — your model will look great in eval but fail in the wild. Train data only.

---

### Week 4 — Module implementation, part 1 (components)

**Goal:** every submodule works in isolation, unit-tested.

Files in `src/models/`:
- `gcn.py` — 2-layer GCN over POI embeddings, use PyG's `GCNConv`. Output: `(|V|, d_p)` matrix of POI features.
- `context_encoder.py` — two small MLPs, one for Δd, one for Δt. Each goes from scalar to `d_c/2` dims. Concat at the end.
- `user_embedding.py` — just `nn.Embedding(|U|, d_u)` with proper init.

For each module, write a smoke test in `tests/test_{module}.py`: build it, run a forward pass on random data, assert output shape. PyTorch's `unittest` or just plain assertion functions are fine.

**Deliverable:** passing test suite + standalone modules.

---

### Week 5 — Module implementation, part 2 (assembly)

**Goal:** end-to-end forward pass produces valid logits.

- `gru_encoder.py` — wraps `nn.GRU` with proper `pack_padded_sequence` / `pad_packed_sequence` for variable-length sessions
- `next_poi_model.py` — the full architecture:
  1. GCN over the POI graph → POI feature matrix
  2. For each step in the input sequence: look up POI feature, concatenate with context features
  3. GRU encodes the sequence → take final hidden state h_T
  4. Concatenate `[h_T ; e_u]`
  5. MLP → logits over `|V|` POIs

Test: feed a batch of size B with sequence length T, expect output of shape `(B, |V|)`, sums-to-1 after softmax.

**Deliverable:** `NextPOIModel` class instantiable, forward pass works.

**Pitfall:** the GCN runs once per forward pass on the full graph and produces all POI features. Don't run it inside a loop over the batch — that's massively wasteful.

---

### Week 6 — Training loop + first training run

**Goal:** model trains on NYC, loss decreases monotonically on training data.

- `DataLoader` with custom `collate_fn` that pads sequences and computes lengths
- Cross-entropy loss with `F.cross_entropy(logits, target, ignore_index=PAD_ID)` — `ignore_index` is essential for masking padding
- Adam optimizer, lr=1e-3, weight decay=1e-5
- Training loop with TensorBoard logging (loss per step, val HR@10 per epoch)
- Checkpoint to Drive after every epoch (`torch.save(model.state_dict(), ...)`)
- Run 50 epochs on NYC. By epoch 5, val HR@10 should be > 0.20. If still near zero at epoch 10, you have a bug.

**Deliverable:** trained NYC checkpoint + training/validation curves.

**Pitfall:** the most common silent bug is **off-by-one in the label** — predicting the input POI instead of the next one. Test: train on a tiny subset (10 users) for 100 epochs and confirm it can memorize. If it can't, the labels are wrong.

---

### Week 7 — Evaluation

**Goal:** test numbers on NYC, comparable to published baselines.

- Implement HR@k, NDCG@k, MRR (k ∈ {1, 5, 10}) — these are simple given the full ranking
- Run on test set, write results to a CSV in `results/`
- Compare to published numbers from LLM4POI's Table 3:

| Baseline | NYC HR@1 (Acc@1) | Source |
|---|---|---|
| LSTM | 0.13 | LLM4POI table 3 |
| STGCN | 0.18 | LLM4POI table 3 |
| STAN | 0.22 | LLM4POI table 3 |
| GETNext | 0.24 | LLM4POI table 3 |
| STHGCN | 0.27 | LLM4POI table 3 |
| LLM4POI | 0.34 | LLM4POI table 3 |
| **Your model** | **target 0.13–0.18** | (this work) |

**Deliverable:** results table for NYC.

**Sanity check:** if your number is much higher than 0.20, you almost certainly have leakage. Re-check the chronological split and the graph construction.

---

### Week 8 — TKY + ablations

**Goal:** second dataset + understand what each component contributes.

- Re-run the full pipeline on TKY. Smaller dataset, faster. Should land in similar band (0.13–0.18 HR@1).
- Ablation runs on NYC, three configurations:
  - **No GCN**: replace `GCNConv` layers with plain `nn.Embedding(|V|, d_p)` (no graph propagation)
  - **No user embedding**: drop the concat with `e_u`, score using `h_T` alone
  - **No context**: set `c_t = 0` (zero out the Δd, Δt features)
- Each ablation is one full training run (~2h on T4). Total: ~6 hours of compute.

**Deliverable:** NYC + TKY results + ablation table. Each component should contribute positively (HR drops when you remove any of them). If a component doesn't help, you've learned something.

---

### Week 9 — Robustness + hyperparameter tuning

**Goal:** confirm reported numbers are stable, not lucky seeds.

- Run NYC with 3 different random seeds (e.g., 42, 123, 2024). Report mean ± std.
- Hyperparameter sweep on val set:
  - Learning rate ∈ {5e-4, 1e-3, 2e-3}
  - Embedding dim ∈ {64, 128, 256}
  - GCN layers ∈ {1, 2, 3}
- Pick best config from val HR@10, lock it, re-run with multiple seeds.

**Deliverable:** final hyperparameter table + standard deviations on the main metrics.

---

### Week 10 — Documentation + write-up

**Goal:** thesis chapter + clean public artifact.

- Code cleanup: docstrings, README, `requirements.txt`, MIT or Apache license
- Architecture diagram (you already have one from our planning conversation — reuse it)
- Thesis baseline chapter, suggested structure:
  - **Problem formulation** — formal definition of next-POI rec, notation
  - **Architecture** — diagram + math (Stage 1, 2, 3)
  - **Experimental setup** — datasets, preprocessing, splits, hyperparameters, hardware
  - **Results** — main table (NYC + TKY), ablation table, hyperparameter sensitivity
  - **Discussion** — honest position vs. SOTA, what worked, what didn't, what next
- Push the cleaned code to a public GitHub repo

**Deliverable:** thesis chapter + GitHub repo.

---

## Success criteria for path (A)

You're done when all four boxes are checked:

- [ ] NYC HR@1 in the 0.13–0.18 range
- [ ] TKY HR@1 in similar band (0.13–0.18)
- [ ] Ablation table where every component contributes positively
- [ ] Clean reproducible code that runs end-to-end on Colab from scratch

---

## Things that will probably go wrong (so you're not surprised)

1. **Colab session disconnects mid-training.** Checkpoint every epoch and have a "resume from latest checkpoint" code path from day one.
2. **PyG installation issues on Colab.** PyG needs version-matched wheels. Use the install snippet from PyG's own Colab examples — don't pip install blindly.
3. **First training gives HR@10 of zero.** Almost always a label-shift bug. Memorization test on tiny data first.
4. **Padding bug in cross-entropy.** Forgetting `ignore_index=PAD_ID` makes the model learn to predict padding tokens. Symptom: training loss looks fine, eval is garbage.
5. **TKY works but NYC doesn't (or vice versa).** Different POI count → different embedding table size. Don't hardcode `|V|`.
6. **Validation curve goes up then down.** Classic overfitting. Increase dropout from 0.2 → 0.4, or reduce embedding dim, or add weight decay.
7. **GCN gives identical features for all POIs.** "Over-smoothing" — too many GCN layers (3+). Stick to 2.

---

## Reference values (from your project's papers)

| Setting | Standard value |
|---|---|
| User filter | ≥10 check-ins |
| POI filter | ≥10 visitors |
| Session split gap | 24 hours (Foursquare convention) |
| Train/val/test | 70/10/20 chronological per user |
| POI embedding dim | 128 |
| User embedding dim | 64 |
| GRU hidden dim | 128 |
| GCN layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam, lr=1e-3, wd=1e-5 |
| Batch size | 64 |
| Max epochs | 50 (early-stop on val HR@10) |
| Co-visit threshold τ_cov | 3 |
| Geographic kNN K | 10 |
| Evaluation k | {1, 5, 10} |

---

## When you're ready to extend (path B teaser)

After (A) is done and you want to push toward GETNext/STAN territory, the four highest-impact swaps in priority order:

1. **Swap MLP-to-|V| output for inner-product scoring** $\hat{y}_v = (h_T + e_u)^T h_v^{\text{GCN}}$ — same TLMR shape, much better transfer of graph signal
2. **Move Δd, Δt into ST-GRU gates** (STGN-style) instead of side concat — gains 3–5 HR points
3. **Add hour-of-day + day-of-week** as cyclic features at every step — typically gains 2–4 HR points on Foursquare
4. **Weight the co-visit graph edges** by transition frequency (DLIR-style adaptive adjacency) instead of unweighted

Each can be a thesis sub-chapter showing the gain over the (A) baseline. That's the natural extension story.
