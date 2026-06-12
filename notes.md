# Notes — Q&A summaries

Concise summaries of questions asked about the next-POI baseline and its report. Each entry is the compressed form of a longer chat answer.

---

## No negative sampling, no candidate restriction

The model's score for the true next POI is ranked against **every** other POI in the vocabulary (all 5,135 on NYC) — not against a sampled subset of 100 negatives, not against a pre-filtered shortlist. This is the strict evaluation protocol used by GETNext / STHGCN / LLM4POI. Sampled or restricted evaluation can inflate HR/NDCG by 2–3× and isn't comparable across papers (Krichene & Rendle, KDD 2020).

---

## The benchmark ladder (Foursquare NYC HR@1)

The literature numbers form a deliberate progression:

- **LSTM (0.13)** — pure RNN on POI sequence, no graph, no spatial info. The honest floor.
- **STGCN (0.18)** — adds a co-visit GCN before the sequence model. +5 points from graph signal.
- **Ours / TLMR (0.187)** — GCN + GRU + user embedding. Same tier as STGCN by design.
- **STAN (0.22)** — self-attention with spatio-temporal biases replaces the RNN. +4 points from attention.
- **GETNext (0.24)** — weighted trajectory-flow graph + time-aware category embeddings + transformer.
- **STHGCN (0.27)** — hyper-graph (edges connecting groups of POIs, not just pairs).
- **LLM4POI (0.34)** — frozen big LLM prompted with check-in history; no training at all.

Our model sits intentionally at the second rung (STGCN tier). Each +0.05 step in the ladder represents significant added complexity, not a quick win.

---

## HR@k, NDCG@k, MRR

All three metrics derive from the **rank** of the true POI in the model's full sorted output (rank 1 = perfect, rank 5135 = worst).

- **HR@k** = fraction of test examples where rank ≤ k. Pure hit/miss within top-k, no positional weighting. HR@1 = exact-match accuracy.
- **NDCG@k** = hit/miss within top-k weighted by `1 / log₂(rank + 1)`. Rewards higher ranks more. Rank 1 contributes 1.0, rank 2 contributes 0.63, rank 5 contributes 0.39.
- **MRR** = mean of `1 / rank` across all test examples. No cutoff. Sharply favors top ranks; an MRR of 0.316 ≈ true POI averages around rank ~3.

Reporting all three is standard because different product surfaces care about different cutoffs (notification = HR@1, search-style UI = HR@10 / NDCG@10, ranking-as-a-service = MRR).

---

## Logits

Logits are the raw, unnormalized scores the final `Linear` layer outputs — one per POI, shape `(B, |V|)`. They can be any real number; only their relative order matters. Sorting POIs by logit gives the same ranking as sorting by softmax probabilities (softmax is monotone), so for HR/NDCG/MRR we work with logits directly and skip the softmax. PyTorch's `F.cross_entropy` takes logits and applies softmax + log internally for numerical stability.

---

## "Building this from train+val+test would be subtle leakage"

If you build the co-visit graph from the entire dataset before splitting, val/test transitions contribute edges. Those edges encode the very next-POI patterns the model is later evaluated on recovering — the GCN's POI representations already "know" Park leads to Bookstore because future users (including the test user) made that transition. The leakage is subtle because the graph looks like static preprocessing, not label flow. Fix: build co-visit edges from training data only ([src/data/graph.py](src/data/graph.py)). kNN edges are safe — they're computed from fixed POI coordinates.

---

## Union, dedup, symmetrize → `edge_index` shape (2, 2|E|)

Three operations to convert two undirected edge lists into a PyG-ready tensor:

1. **Union with dedup**: `set(covisit) | set(knn)` — drops pairs that appear in both lists (≈2k overlap on NYC).
2. **Symmetrize**: each undirected pair `{a,b}` becomes two directed entries `a→b` and `b→a`, because PyG's message passing is directional and we need information flowing both ways for an undirected graph.
3. **Format**: stack source indices in row 0, destination indices in row 1, giving shape `(2, 2|E|)` where `|E|` = unique undirected edges. On NYC: 34,898 unique edges → `(2, 69,796)` tensor.

---

## Is storing edges as `(min, max)` a problem?

It's a documented design choice that's a slight deviation from the strictest reading of the spec. Storing co-visit pairs as `(min, max)` collapses direction at counting time, so 2 users going A→B and 2 going B→A combine to count=4 (passes threshold ≥3), whereas a strictly directional reading would require 3 users in one specific direction. It's defensible because (a) the GCN treats edges as undirected anyway after symmetrization, (b) bidirectional traversal is *more* evidence of functional connection, not less, (c) STGCN and similar papers do the same. The consequences: slightly more edges, easier threshold, asymmetric flows invisible (but the GRU sees them via sequence order). Worth a one-liner in a thesis "deviations" section.

---

## Why call the `[[0,1,2,3],[1,0,3,2]]` thing a tensor instead of a matrix?

In PyTorch/PyG, "tensor" is the framework's name for any n-dimensional array — scalars (0-D), vectors (1-D), matrices (2-D), and higher. The Python object returned by `torch.tensor(...)` is literally of type `torch.Tensor`. The convention exists because the same type generalizes to any rank, and "tensor" is what PyG's docs and APIs use. So `edge_index` is a 2×|E| matrix mathematically — calling it a tensor just matches the PyTorch vocabulary.

---

## How the 10 kNN neighbors are computed

In [src/data/graph.py](src/data/graph.py): convert POI lat/lon from degrees to radians, build a `sklearn.neighbors.BallTree` with `metric='haversine'` (great-circle distance on a sphere — needed because Euclidean distance on (lat, lon) is wrong since 1° of longitude shrinks with latitude), then query each POI for its `k+1=11` nearest neighbors. The `+1` is because each POI's own row sits at distance 0 (itself is the nearest neighbor) — slice it off with `neighbors[1:]`. Edges are stored as `(min, max)` to dedup. kNN relations aren't symmetric — A might list B but B might have 10 closer POIs than A — so the edge is added if either direction lists the other, which is why the actual average degree on NYC (13.6) is slightly above k=10. BallTree brings the cost from O(N²) to O(N log N) so the full 5,135-POI kNN finishes in well under a second.

---

## Why `D̃⁻¹ᐟ² · Ã · D̃⁻¹ᐟ²` instead of `D̃⁻¹ · Ã`

Concrete star example: hub `H` (deg=5 with self-loop) connected to 4 leaves `Lᵢ` (each deg=2), with H=100 and all leaves=0.

- **Row normalization** (asymmetric): edge weight = `1 / deg(receiver)` only. Each leaf receives `(1/2) · 100 = 50` — the hub's signal hits at half strength regardless of how many leaves the hub has. Scale to 99 leaves: each still gets 50.
- **Symmetric normalization**: edge weight = `1 / √(deg(i) · deg(j))`. Each leaf receives `(1/√10) · 100 ≈ 31.6`. Scale to 99 leaves: each gets `(1/√200) · 100 ≈ 7.1` — the hub's signal is automatically dampened by `1/√deg(H)`.

The intuition: a node connected to *everyone* provides weak evidence for any specific neighbor; a node connected to *a few specific* others provides strong evidence for those. Symmetric normalization encodes this; row normalization doesn't. For our POI graph this matters because hub POIs (Times Square, Penn Station, deg up to 79) would otherwise overwrite specific signals from small neighborhood POIs.

Mathematical bonus: `D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ²` is symmetric → eigenvalues real and bounded in [-1, 1] → numerically stable across stacked layers. Row normalization breaks this property.

---

## Why we need nonlinearity

Without nonlinearity, stacking layers collapses to a single linear map: `W₂(W₁x) = (W₂W₁)x = Wx`. Two GCN layers without ReLU = one GCN layer with a different weight matrix; the depth buys nothing.

In our model:
- **GCN layers** — ReLU between layer 1 and layer 2 lets the second hop combine neighbor features in ways the first hop couldn't express linearly (e.g., "near a park *and* in a co-visit cluster" rather than just a weighted sum).
- **GRU** — sigmoid/tanh gates are what let the hidden state decide *whether* to remember vs. overwrite past check-ins, instead of just averaging them.
- **Final classifier** — `Linear → logits` is intentionally linear; softmax provides the nonlinearity at the output and we want the penultimate representation to be linearly separable.

Nonlinearity is what makes "deep" actually deep — it lets the network represent compositions (AND/OR/XOR over features) rather than just weighted sums.

---

## Notation key

**Dimensions (`d_*` family)**
- **d_p = 128** — POI embedding dim. Each POI is a 128-dim vector both as the initial learnable embedding `E⁽⁰⁾ ∈ ℝ^(|V| × d_p)` and as the GCN output `h_p^GCN`.
- **d_u = 64** — user embedding dim. `e_u ∈ ℝ^d_u`.
- **d_c = 32** — context dim (16 from Δd + 16 from Δt, concatenated).
- **d_h = 128** — GRU hidden dim. Final hidden state `h_T ∈ ℝ^d_h`.

**Sets and counts**
- **U** — set of users; **|U|** = user count.
- **V** — POI vocabulary; **|V| = 5,135** on NYC after filtering.
- **G = (V, E)** — hybrid graph. **E_cov** = co-visit edges, **E_geo** = kNN edges, **E = E_cov ∪ E_geo**.
- **τ_cov = 3** — co-visit edge threshold (≥3 user transitions).
- **k = 10** — kNN neighbors per POI.

**Sequence variables (per session)**
- **S = (p₁, …, p_L)** — session of length L.
- **p_t** — POI index at step t.
- **T** — current prefix length; we predict **p_{T+1}**.
- **Δd_t** — haversine distance (km) from p_{t-1} to p_t.
- **Δt_t** — time gap (h) from p_{t-1} to p_t.
- **c_t** — per-step context vector, in ℝ^d_c.
- **x_t = [h_{p_t}^GCN ; c_t]** — GRU input, in ℝ^(d_p + d_c) = ℝ^160.
- **h_T** — final GRU hidden state, in ℝ^d_h.
- **z = [h_T ; e_u]** — pre-classifier vector, in ℝ^(d_h + d_u) = ℝ^192.

**GCN math**
- **E⁽ℓ⁾** — POI embedding matrix at layer ℓ.
- **W⁽ℓ⁾** — GCN layer weight matrix.
- **A** — adjacency; **Ã = A + I** adds self-loops.
- **D̃** — diagonal degree matrix of Ã; **D̃^(−½) Ã D̃^(−½)** is symmetric normalization.
- **σ** — nonlinearity (ReLU).

**Output / loss**
- **ŷ ∈ ℝ^|V|** — logits, one per POI.
- **f_θ(S, u; G)** — the whole model.
- **y / yᵢ** — true next-POI index (label).
- **N** — number of examples.
- **rank(y)** — position of the true POI in sorted logits (1 = best).

**Batch shapes**
- **B** — batch size (64).
- `(V, d_p)` — full POI feature matrix from GCN.
- `(B, T, d_c)` — batched context tensor.
- `(B, d_u)` — batched user embeddings.

---

## "Computed once per forward pass, not once per training example"

The GCN's output `(|V|, d_p)` depends only on the POI embedding table and the fixed graph — not on per-example inputs (`poi_ids`, user, context). So it's the same for every example in a batch. Compute once at the top of `forward()`, then per-example work is just `poi_features[poi_ids]` — a differentiable index gather of cost `O(B·T)`. On NYC: ~1,170 GCN runs per epoch instead of ~75,000 if done per example (~64× cheaper, matching batch size). Gradients still flow correctly because the gather is differentiable — each example contributes gradient to the rows it touched, and all examples accumulate into the shared GCN parameters. Without this trick, training wouldn't fit in a Colab session.

---

## "One row per POI in the whole vocabulary, not per example"

Two different matrices in the model that are easy to confuse:

- **GCN output `(|V|, d_p)` = `(5135, 128)` on NYC** — one row per POI. Row 42 = feature for POI #42. Same matrix regardless of which examples or users are in the batch.
- **Sequence input to GRU `(B, T, d_p)` = `(64, 10, 128)`** — one row per (example, timestep). This is built from the first matrix by **indexing**: each example's history of POI IDs picks out the corresponding rows.

If Times Square (POI 0) appears in both Example A (step 1) and Example B (step 3), both reads pull the *same* row 0 from the GCN output — no duplicate work. The GCN's job is to assign a feature vector to each POI in the city; examples don't get their own rows in this table, they consume it by gather.

---

## T — length of the session prefix

`T` = how many POIs are in the history the model sees for one example. **Per-example, not a fixed constant.** A session of length L produces L−1 training examples with T = 1, 2, …, L−1 (each prefix length is one example, predicting the next POI). Average T on NYC is ~2–3 since session length averages 5.2. Hard cap `max_seq_len = 100` (left-truncate longer histories).

In tensor shapes like `(B, T, d_p)`, T is the *batch maximum* — examples with shorter real lengths are right-padded with zeros, and a separate `lengths` tensor `(B,)` tells `pack_padded_sequence` to ignore the padding.

Lowercase `t` is the *index* (1..T) into the prefix; `p_t`, `c_t`, `x_t` are the values at step t. `p_{T+1}` is the prediction target. `h_T` is the GRU hidden state after consuming all T real steps — the prefix summary.

---

## Forward pass

One execution of `model(batch)` — data flowing through every layer in order, from inputs to logits. "Forward" because data moves in the direction the architecture arrows point (input → hidden → output). The companion is the **backward pass**, which goes the other way and computes gradients via the chain rule. One forward + one backward = one training step.

In our model, one forward pass does 8 steps in sequence: GCN over the full graph → gather per-step POI features by index → context MLPs on Δd/Δt → concat → GRU → user embedding lookup → concat → MLP head. Output is `(B, |V|)` logits.

"Once per forward pass" = once per batch (1 call to `model(batch)`), not once per example or per timestep. NYC training does ~1,170 forward passes per epoch (75K examples ÷ batch 64). At training time each forward pass is followed by a backward pass; at eval time there's no backward pass (`torch.no_grad()`).

---

## Batch size 64 = 64 training examples

A **training example** in our setup = one `(user, history, target)` pair, where `history` is a session prefix and `target` is the next POI to predict. A session of length L generates L−1 examples (prefix lengths 1..L−1). NYC has ~74,916 such examples after the prefix expansion.

A **batch** packs 64 examples together so the GPU processes them in parallel. Examples in one batch can come from completely different users and sessions — they're just grouped for compute. After collate, every batch tensor has leading dim 64: `poi_ids (64, T)`, `targets (64,)`, etc. Model output is `(64, 5135)` logits.

**Why batch:** GPU matmul on `(64, 128) × (128, 5135)` takes ~same wall time as 64 separate matmuls on single examples — batching is essentially free compute. Also gradient noise averages out across 64 examples, stabilizing optimizer updates.

**Why 64:** memory fits comfortably on T4 (room for the 5135×128 GCN matrix + GRU activations + gradients); STGCN, GETNext, and most POI-rec papers use 64 → results stay directly comparable. NYC: 74,916 examples ÷ 64 ≈ 1,170 batches = 1,170 forward passes per epoch.

---

## Forward pass during training vs inference

**At training time:**
1. Forward pass → produces logits ŷ.
2. Compute loss: `F.cross_entropy(ŷ, target)`.
3. Backward pass: `loss.backward()` — gradients flow back through every operation in the forward pass.
4. Optimizer step: `optimizer.step()` — Adam uses gradients to update parameters.

**At inference (eval) time:**
1. Forward pass → produces logits.
2. Sort logits, return top-k.

That's it — no backward pass, no parameter updates. `model.eval()` and `torch.no_grad()` tell PyTorch not to track gradients, which makes it faster and uses less memory.

---

## manually writeen note (they should always stay at the end of the file)

 The identity matrix I puts a 1 on the diagonal, which adds a self-edge p → p for every POI. Why? Without self-loops, the formula new_features(p) = average(neighbors) replaces p's old features with its neighbors' features — p's own information is lost. 


