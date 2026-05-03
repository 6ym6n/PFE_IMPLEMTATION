# Next-POI Recommendation: Full Implementation Guide

**TLMR-adapted next-POI top-k recommendation on Foursquare NYC + TKY.**

This document is self-contained: every cell of code below can be pasted directly into a Google Colab notebook, in order. Expected runtime end-to-end on a free T4: about 3–4 hours including training. By the end you'll have a trained model with HR@k, NDCG@k, and MRR numbers comparable to the LSTM/STGCN tier from the literature.

---

## Table of contents

1. [Architecture spec (math + dimensions)](#1-architecture-spec)
2. [Colab setup and Drive mounting](#2-colab-setup)
3. [Dataset download](#3-dataset-download)
4. [Preprocessing](#4-preprocessing)
5. [Hybrid POI graph construction](#5-graph-construction)
6. [Model implementation](#6-model)
7. [Dataset and DataLoader](#7-dataloader)
8. [Training loop](#8-training)
9. [Evaluation](#9-evaluation)
10. [Running everything end-to-end](#10-running)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Architecture spec

### Notation

| Symbol | Meaning | Default |
|---|---|---|
| $\mathcal{V}$ | set of all POIs | ~5K (NYC), ~7K (TKY) after filtering |
| $\mathcal{U}$ | set of users | ~1K (NYC), ~2K (TKY) |
| $S_u$ | check-in session of user $u$ | variable length |
| $p_t \in \mathcal{V}$ | POI at step $t$ | |
| $\Delta d_t$ | haversine distance from $p_{t-1}$ to $p_t$ (km) | scalar |
| $\Delta t_t$ | time gap from $p_{t-1}$ to $p_t$ (hours) | scalar |
| $d_p$ | POI embedding dim | 128 |
| $d_u$ | user embedding dim | 64 |
| $d_c$ | context embedding dim | 32 |
| $d_h$ | GRU hidden dim | 128 |

### Stage 1 — feature extraction

**1.1 Hybrid POI graph.** $G = (\mathcal{V}, \mathcal{E}_{\text{cov}} \cup \mathcal{E}_{\text{geo}})$:
- $\mathcal{E}_{\text{cov}}$: $(p_i, p_j) \in \mathcal{E}_{\text{cov}}$ if at least 3 users transitioned $p_i \rightarrow p_j$ within 24h in **training data**
- $\mathcal{E}_{\text{geo}}$: each POI connected to its 10 nearest neighbors by haversine distance

**1.2 GCN encoder.** Initial POI embeddings $E^{(0)} \in \mathbb{R}^{|\mathcal{V}| \times d_p}$, two layers:

$$E^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} E^{(l)} W^{(l)}\right), \quad l = 0, 1$$

Output: $h_p^{\text{GCN}} = E^{(2)}_p \in \mathbb{R}^{d_p}$ for each POI.

**1.3 Context.** Per step:
$$c_t = [\text{MLP}_d(\Delta d_t) \,;\, \text{MLP}_t(\Delta t_t)] \in \mathbb{R}^{d_c}$$

For the first POI of a session: $\Delta d_1 = \Delta t_1 = 0$.

**1.4 User embedding.** $e_u \in \mathbb{R}^{d_u}$ from a learnable lookup table.

### Stage 2 — model learning

Per step input: $x_t = [h_{p_t}^{\text{GCN}} \,;\, c_t] \in \mathbb{R}^{d_p + d_c}$

GRU encodes $X = [x_1, ..., x_T]$, take final hidden state $h_T \in \mathbb{R}^{d_h}$.

Concatenate with user: $z = [h_T \,;\, e_u] \in \mathbb{R}^{d_h + d_u}$

MLP scoring head:

$$\hat{y} = W_2 \, \text{ReLU}(W_1 z + b_1) + b_2 \in \mathbb{R}^{|\mathcal{V}|}$$

### Stage 3 — output

$$P(p_{T+1} = p_j \mid S_u, u) = \frac{\exp(\hat{y}_j)}{\sum_k \exp(\hat{y}_k)}$$

Top-k = highest-probability $k$ POIs.

### Loss

$$\mathcal{L} = -\frac{1}{N} \sum_i \log P(y^{(i)} \mid S_u^{(i)}, u^{(i)})$$

---

## 2. Colab setup

Open a fresh Colab notebook. Runtime → Change runtime type → **T4 GPU**.

### Cell: install packages

```python
!pip install -q torch_geometric tensorboardX
print("Done. Runtime should NOT need to restart.")
```

Expected: ~30 seconds, no errors. PyG 2.5+ installs cleanly on Colab without needing `torch_scatter` or `torch_sparse` wheels.

### Cell: mount Drive and set up paths

```python
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = '/content/drive/MyDrive/poi-rec'
os.makedirs(PROJECT_ROOT, exist_ok=True)
for sub in ['data/raw', 'data/processed', 'checkpoints', 'results']:
    os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(os.listdir(PROJECT_ROOT))
```

Expected output: `['data', 'checkpoints', 'results']`

### Cell: imports + reproducibility

```python
import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.neighbors import BallTree
from collections import defaultdict
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)
```

Expected output: `Device: cuda`. If CPU, your runtime didn't get a GPU — change runtime type.

---

## 3. Dataset download

The standard benchmark is Yang et al.'s Foursquare-NYC and Foursquare-TKY ("Modeling user activity preference by leveraging user spatial temporal characteristics in LBSNs", IEEE TSMC 2015).

### Cell: download the data

```python
import urllib.request, zipfile

# Source: this is the original distribution. If the URL 404s,
# alternates: GETNext repo on GitHub, or Kaggle "foursquare-nyc-and-tokyo-checkin-dataset".
URL = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
RAW_DIR = os.path.join(PROJECT_ROOT, 'data/raw')
ZIP_PATH = os.path.join(RAW_DIR, 'dataset_tsmc2014.zip')

if not os.path.exists(ZIP_PATH):
    print("Downloading... ~30MB")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Done.")

if not os.path.exists(os.path.join(RAW_DIR, 'dataset_TSMC2014_NYC.txt')):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(RAW_DIR)
    print("Extracted.")

print(os.listdir(RAW_DIR))
```

Expected to see two files in addition to the zip: `dataset_TSMC2014_NYC.txt` and `dataset_TSMC2014_TKY.txt`.

**If the URL returns 404:** download manually from one of the mirrors (GETNext repo on GitHub: `https://github.com/songyangco/GETNext`, look in their `data/` folder; or search Kaggle), upload to `data/raw/` via the Files panel.

### Cell: load and inspect

```python
COLUMNS = ['user_id', 'poi_id', 'cat_id', 'cat_name',
           'lat', 'lon', 'tz_offset', 'utc_time']

def load_foursquare(path):
    df = pd.read_csv(path, sep='\t', names=COLUMNS, encoding='latin-1')
    df['timestamp'] = pd.to_datetime(df['utc_time'],
                                     format='%a %b %d %H:%M:%S +0000 %Y')
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    return df[['user_id', 'poi_id', 'cat_id', 'cat_name', 'lat', 'lon', 'timestamp']]

df_nyc = load_foursquare(os.path.join(RAW_DIR, 'dataset_TSMC2014_NYC.txt'))
df_tky = load_foursquare(os.path.join(RAW_DIR, 'dataset_TSMC2014_TKY.txt'))

for name, df in [('NYC', df_nyc), ('TKY', df_tky)]:
    print(f"{name}: {len(df):,} check-ins | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['poi_id'].nunique():,} POIs | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
```

Expected output (approximately):
```
NYC: 227,428 check-ins | 1,083 users | 38,333 POIs | 2012-04-03 → 2013-02-16
TKY: 573,703 check-ins | 2,293 users | 61,858 POIs | 2012-04-03 → 2013-02-16
```

These are the raw counts before filtering. POI counts will drop dramatically after filtering.

---

## 4. Preprocessing

### Cell: filtering + re-indexing + sessionization

```python
def iterative_filter(df, min_user=10, min_poi=10, max_iter=10):
    """Iteratively filter out low-activity users and rare POIs until stable."""
    for i in range(max_iter):
        n0 = len(df)
        uc = df['user_id'].value_counts()
        df = df[df['user_id'].isin(uc[uc >= min_user].index)]
        pc = df['poi_id'].value_counts()
        df = df[df['poi_id'].isin(pc[pc >= min_poi].index)]
        if len(df) == n0:
            break
    return df.reset_index(drop=True)

def reindex(df):
    """Re-map user_id and poi_id to contiguous integers 0..N-1."""
    user2idx = {u: i for i, u in enumerate(sorted(df['user_id'].unique()))}
    poi2idx = {p: i for i, p in enumerate(sorted(df['poi_id'].unique()))}
    df = df.copy()
    df['user_idx'] = df['user_id'].map(user2idx)
    df['poi_idx'] = df['poi_id'].map(poi2idx)
    return df, user2idx, poi2idx

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_sessions(df, gap_hours=24):
    """Group consecutive check-ins into sessions; gaps > 24h start a new session.
       Compute Δd (km) and Δt (hours) per consecutive pair within session."""
    df = df.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)

    # gap from previous check-in (within same user)
    gap = df.groupby('user_idx')['timestamp'].diff().dt.total_seconds() / 3600
    new_session = (gap > gap_hours) | gap.isna()
    df['session_id'] = new_session.cumsum()  # globally unique session id

    # Δt within session (NaN at session boundary → 0)
    df['delta_t'] = df.groupby('session_id')['timestamp'].diff().dt.total_seconds() / 3600
    df['delta_t'] = df['delta_t'].fillna(0.0).clip(0, 24)  # cap at 24h within session

    # Δd within session
    lat_prev = df.groupby('session_id')['lat'].shift()
    lon_prev = df.groupby('session_id')['lon'].shift()
    df['delta_d'] = haversine_km(df['lat'], df['lon'], lat_prev, lon_prev)
    df['delta_d'] = df['delta_d'].fillna(0.0).clip(0, 100)  # cap at 100km

    # drop sessions with only 1 check-in (no prediction possible)
    sess_lengths = df.groupby('session_id').size()
    valid = sess_lengths[sess_lengths >= 2].index
    df = df[df['session_id'].isin(valid)].reset_index(drop=True)

    return df
```

### Cell: chronological train/val/test split per user

```python
def chronological_split(df, train_frac=0.7, val_frac=0.1):
    """Per-user chronological split. Each user contributes sessions to all three sets
       in chronological order so no future leaks into the past."""
    train_parts, val_parts, test_parts = [], [], []
    for user, g in df.groupby('user_idx'):
        # Sessions sorted by their first timestamp
        sess_order = (g.groupby('session_id')['timestamp']
                       .min().sort_values().index.tolist())
        n = len(sess_order)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_sess = set(sess_order[:n_train])
        val_sess = set(sess_order[n_train:n_train + n_val])
        test_sess = set(sess_order[n_train + n_val:])

        train_parts.append(g[g['session_id'].isin(train_sess)])
        val_parts.append(g[g['session_id'].isin(val_sess)])
        test_parts.append(g[g['session_id'].isin(test_sess)])

    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(val_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))
```

### Cell: run preprocessing for one dataset

```python
def preprocess_dataset(df_raw, name):
    print(f"\n=== {name} ===")
    print(f"Raw: {len(df_raw):,} check-ins, "
          f"{df_raw['user_id'].nunique()} users, "
          f"{df_raw['poi_id'].nunique()} POIs")

    df = iterative_filter(df_raw)
    print(f"Filtered: {len(df):,} check-ins, "
          f"{df['user_id'].nunique()} users, "
          f"{df['poi_id'].nunique()} POIs")

    df, user2idx, poi2idx = reindex(df)
    df = build_sessions(df)
    print(f"Sessionized: {df['session_id'].nunique():,} sessions, "
          f"avg length {df.groupby('session_id').size().mean():.1f}")

    train, val, test = chronological_split(df)
    print(f"Train: {train['session_id'].nunique()} sessions ({len(train)} check-ins)")
    print(f"Val:   {val['session_id'].nunique()} sessions ({len(val)} check-ins)")
    print(f"Test:  {test['session_id'].nunique()} sessions ({len(test)} check-ins)")

    # Persist
    out = os.path.join(PROJECT_ROOT, 'data/processed', name)
    os.makedirs(out, exist_ok=True)
    train.to_parquet(os.path.join(out, 'train.parquet'))
    val.to_parquet(os.path.join(out, 'val.parquet'))
    test.to_parquet(os.path.join(out, 'test.parquet'))

    # POI coordinates (lat, lon) indexed by poi_idx — needed for kNN graph
    poi_coords = (df.groupby('poi_idx')[['lat', 'lon']].first()
                    .sort_index().values)
    np.save(os.path.join(out, 'poi_coords.npy'), poi_coords)

    meta = {
        'n_users': len(user2idx),
        'n_pois': len(poi2idx),
        'n_train_sessions': int(train['session_id'].nunique()),
        'n_val_sessions': int(val['session_id'].nunique()),
        'n_test_sessions': int(test['session_id'].nunique()),
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return train, val, test, meta

train_nyc, val_nyc, test_nyc, meta_nyc = preprocess_dataset(df_nyc, 'NYC')
train_tky, val_tky, test_tky, meta_tky = preprocess_dataset(df_tky, 'TKY')
```

Expected output (approximate):
```
=== NYC ===
Raw: 227,428 check-ins, 1,083 users, 38,333 POIs
Filtered: ~106,000 check-ins, ~1,000 users, ~5,000 POIs
Sessionized: ~17,000 sessions, avg length ~6
Train: ~12,000 sessions (~74,000 check-ins)
Val:   ~1,700 sessions (~10,000 check-ins)
Test:  ~3,400 sessions (~21,000 check-ins)

=== TKY ===
Raw: 573,703 check-ins, 2,293 users, 61,858 POIs
Filtered: ~280,000 check-ins, ~2,200 users, ~7,500 POIs
...
```

Numbers vary slightly by exact filter implementation — that's OK.

---

## 5. Graph construction

### Cell: build hybrid graph

```python
def build_covisit_edges(train_df, threshold=3):
    """Co-visit edges: count transitions p_i → p_{i+1} within sessions in training data."""
    counts = defaultdict(int)
    for _, g in train_df.groupby('session_id'):
        seq = g.sort_values('timestamp')['poi_idx'].values
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i+1]
            if a != b:
                # store as undirected: smaller index first
                key = (min(a, b), max(a, b))
                counts[key] += 1
    return [edge for edge, c in counts.items() if c >= threshold]

def build_knn_edges(poi_coords, k=10):
    """Geographic kNN edges using BallTree with haversine metric.
       poi_coords: numpy array (n_pois, 2) of [lat_deg, lon_deg]."""
    coords_rad = np.radians(poi_coords)
    tree = BallTree(coords_rad, metric='haversine')
    # Query k+1 because each POI is its own nearest neighbor
    _, idx = tree.query(coords_rad, k=k+1)
    edges = set()
    for i, neighbors in enumerate(idx):
        for j in neighbors[1:]:  # skip self at position 0
            edges.add((min(i, j), max(i, j)))
    return list(edges)

def build_hybrid_graph(train_df, poi_coords, n_pois,
                       covisit_threshold=3, knn_k=10):
    cov = build_covisit_edges(train_df, threshold=covisit_threshold)
    knn = build_knn_edges(poi_coords, k=knn_k)
    all_edges = list(set(cov) | set(knn))
    print(f"  Co-visit edges: {len(cov):,}")
    print(f"  kNN edges:      {len(knn):,}")
    print(f"  Union:          {len(all_edges):,}")

    # Convert to symmetric edge_index for PyG (each undirected edge → two directed)
    src, dst = [], []
    for a, b in all_edges:
        src.extend([a, b])
        dst.extend([b, a])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Sanity: degree distribution
    degree = torch.zeros(n_pois, dtype=torch.long)
    for s in src:
        degree[s] += 1
    print(f"  Degree: min={degree.min().item()}, max={degree.max().item()}, "
          f"mean={degree.float().mean().item():.1f}, "
          f"median={degree.median().item()}")
    if (degree == 0).any():
        n_iso = (degree == 0).sum().item()
        print(f"  WARNING: {n_iso} isolated POIs (will get no graph signal)")

    return edge_index

def make_and_save_graph(name, train_df, meta):
    print(f"\n=== Building graph for {name} ===")
    out = os.path.join(PROJECT_ROOT, 'data/processed', name)
    poi_coords = np.load(os.path.join(out, 'poi_coords.npy'))
    edge_index = build_hybrid_graph(train_df, poi_coords, meta['n_pois'])
    torch.save(edge_index, os.path.join(out, 'edge_index.pt'))
    return edge_index

edge_index_nyc = make_and_save_graph('NYC', train_nyc, meta_nyc)
edge_index_tky = make_and_save_graph('TKY', train_tky, meta_tky)
```

Expected output (approximate, for NYC):
```
Co-visit edges: ~3,000-5,000
kNN edges:      ~25,000  (= n_pois × k / 2 deduplicated)
Union:          ~27,000
Degree: min=10, max=~100, mean=~10, median=10
```

Most POIs end up with degree close to k=10 (from kNN). Highly co-visited ones get higher degree. Zero isolated POIs is the goal.

---

## 6. Model

Now the architecture itself. Each module is plain PyTorch.

### Cell: model components

```python
class POIGraphEncoder(nn.Module):
    """2-layer GCN over the POI graph. Returns POI features for the whole vocabulary."""
    def __init__(self, n_pois, d_p, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_pois, d_p)
        self.convs = nn.ModuleList([GCNConv(d_p, d_p) for _ in range(n_layers)])
        self.dropout = dropout
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        x = self.embedding.weight  # (n_pois, d_p)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ContextEncoder(nn.Module):
    """Lift Δd and Δt scalars into d_c-dim embedding."""
    def __init__(self, d_c=32):
        super().__init__()
        d_half = d_c // 2
        self.mlp_d = nn.Sequential(
            nn.Linear(1, d_half), nn.ReLU(),
            nn.Linear(d_half, d_half))
        self.mlp_t = nn.Sequential(
            nn.Linear(1, d_half), nn.ReLU(),
            nn.Linear(d_half, d_half))

    def forward(self, delta_d, delta_t):
        # delta_d, delta_t: (B, T) float
        d = self.mlp_d(delta_d.unsqueeze(-1))  # (B, T, d_half)
        t = self.mlp_t(delta_t.unsqueeze(-1))
        return torch.cat([d, t], dim=-1)       # (B, T, d_c)


class NextPOIModel(nn.Module):
    def __init__(self, n_pois, n_users,
                 d_p=128, d_u=64, d_c=32, d_h=128, d_hidden=256,
                 dropout=0.2):
        super().__init__()
        self.gcn = POIGraphEncoder(n_pois, d_p, n_layers=2, dropout=dropout)
        self.context = ContextEncoder(d_c)
        self.user_emb = nn.Embedding(n_users, d_u)
        self.gru = nn.GRU(d_p + d_c, d_h, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_h + d_u, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_pois)
        )
        nn.init.xavier_uniform_(self.user_emb.weight)

    def forward(self, poi_ids, delta_d, delta_t, user_ids, lengths, edge_index):
        """
        poi_ids:   (B, T) long  — sequence of POI indices, padded with 0
        delta_d:   (B, T) float — spatial gaps (km), padded with 0
        delta_t:   (B, T) float — temporal gaps (h), padded with 0
        user_ids:  (B,) long
        lengths:   (B,) long  — actual sequence lengths
        edge_index: (2, E) long — full graph
        Returns:   (B, n_pois) float — logits
        """
        poi_features = self.gcn(edge_index)            # (V, d_p)
        seq = poi_features[poi_ids]                    # (B, T, d_p)
        ctx = self.context(delta_d, delta_t)           # (B, T, d_c)
        x = torch.cat([seq, ctx], dim=-1)              # (B, T, d_p + d_c)

        # Pack to handle variable lengths
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_T = self.gru(packed)                      # h_T: (1, B, d_h)
        h_T = h_T.squeeze(0)                           # (B, d_h)

        u = self.user_emb(user_ids)                    # (B, d_u)
        z = torch.cat([h_T, u], dim=-1)                # (B, d_h + d_u)
        return self.head(z)                            # (B, V)
```

### Cell: smoke test

```python
def smoke_test():
    """Quick check: model forward pass with dummy data on small graph."""
    n_pois, n_users = 100, 20
    model = NextPOIModel(n_pois, n_users).to(DEVICE)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]],
                              dtype=torch.long, device=DEVICE)

    B, T = 8, 5
    poi_ids = torch.randint(0, n_pois, (B, T), device=DEVICE)
    delta_d = torch.rand(B, T, device=DEVICE)
    delta_t = torch.rand(B, T, device=DEVICE)
    user_ids = torch.randint(0, n_users, (B,), device=DEVICE)
    lengths = torch.tensor([5, 4, 3, 5, 2, 4, 5, 3])

    logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)
    print(f"Output shape: {logits.shape}  (expected ({B}, {n_pois}))")
    print(f"After softmax, row sum: {F.softmax(logits, dim=-1).sum(dim=-1)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

smoke_test()
```

Expected:
```
Output shape: torch.Size([8, 100])  (expected (8, 100))
After softmax, row sum: tensor([1., 1., 1., 1., 1., 1., 1., 1.])
Total parameters: ~50,000  (small because n_pois=100)
```

If anything errors here, fix it before going further.

---

## 7. Dataset and DataLoader

### Cell: Dataset class

```python
class POISessionDataset(Dataset):
    """Each item is a (history, target) pair generated from a session.
       A session of length L gives L-1 training pairs (using prefixes 1..L-1)."""

    def __init__(self, df, max_seq_len=100):
        self.examples = []
        for _, g in df.groupby('session_id'):
            g = g.sort_values('timestamp')
            poi = g['poi_idx'].values
            dd = g['delta_d'].values.astype(np.float32)
            dt = g['delta_t'].values.astype(np.float32)
            user = int(g['user_idx'].iloc[0])

            # Generate prefix → next pairs: prefix length 1..L-1
            for i in range(1, len(poi)):
                start = max(0, i - max_seq_len)
                self.examples.append({
                    'user': user,
                    'history_pois': poi[start:i],
                    'history_dd': dd[start:i],
                    'history_dt': dt[start:i],
                    'target': int(poi[i]),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad variable-length histories to max length in batch."""
    lengths = torch.tensor([len(b['history_pois']) for b in batch], dtype=torch.long)
    max_len = lengths.max().item()
    B = len(batch)

    poi_ids = torch.zeros(B, max_len, dtype=torch.long)
    delta_d = torch.zeros(B, max_len, dtype=torch.float)
    delta_t = torch.zeros(B, max_len, dtype=torch.float)

    for i, b in enumerate(batch):
        L = len(b['history_pois'])
        poi_ids[i, :L] = torch.from_numpy(b['history_pois']).long()
        delta_d[i, :L] = torch.from_numpy(b['history_dd'])
        delta_t[i, :L] = torch.from_numpy(b['history_dt'])

    return {
        'poi_ids': poi_ids,
        'delta_d': delta_d,
        'delta_t': delta_t,
        'user_ids': torch.tensor([b['user'] for b in batch], dtype=torch.long),
        'lengths': lengths,
        'targets': torch.tensor([b['target'] for b in batch], dtype=torch.long),
    }
```

### Cell: build datasets and loaders for one city

```python
def make_loaders(name, batch_size=64):
    """Load processed data and build train/val/test loaders for a city."""
    out = os.path.join(PROJECT_ROOT, 'data/processed', name)
    train_df = pd.read_parquet(os.path.join(out, 'train.parquet'))
    val_df   = pd.read_parquet(os.path.join(out, 'val.parquet'))
    test_df  = pd.read_parquet(os.path.join(out, 'test.parquet'))
    edge_index = torch.load(os.path.join(out, 'edge_index.pt')).to(DEVICE)
    with open(os.path.join(out, 'meta.json')) as f:
        meta = json.load(f)

    train_ds = POISessionDataset(train_df)
    val_ds   = POISessionDataset(val_df)
    test_ds  = POISessionDataset(test_df)

    print(f"{name}: train={len(train_ds):,} | val={len(val_ds):,} | test={len(test_ds):,} examples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    return train_loader, val_loader, test_loader, edge_index, meta
```

---

## 8. Training

### Cell: training and evaluation functions

```python
def train_one_epoch(model, loader, optimizer, edge_index):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc='train', leave=False)
    for batch in pbar:
        poi_ids  = batch['poi_ids'].to(DEVICE)
        delta_d  = batch['delta_d'].to(DEVICE)
        delta_t  = batch['delta_t'].to(DEVICE)
        user_ids = batch['user_ids'].to(DEVICE)
        lengths  = batch['lengths']
        targets  = batch['targets'].to(DEVICE)

        logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(model, loader, edge_index, ks=(1, 5, 10)):
    """Compute HR@k, NDCG@k, MRR over the loader."""
    model.eval()
    sums = {f'HR@{k}': 0.0 for k in ks}
    sums.update({f'NDCG@{k}': 0.0 for k in ks})
    sums['MRR'] = 0.0
    n = 0

    for batch in loader:
        poi_ids  = batch['poi_ids'].to(DEVICE)
        delta_d  = batch['delta_d'].to(DEVICE)
        delta_t  = batch['delta_t'].to(DEVICE)
        user_ids = batch['user_ids'].to(DEVICE)
        lengths  = batch['lengths']
        targets  = batch['targets'].to(DEVICE)

        logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)

        # rank of each target = number of POIs scored strictly higher than target + 1
        target_scores = logits.gather(1, targets.unsqueeze(1))   # (B, 1)
        ranks = (logits > target_scores).sum(dim=1) + 1          # (B,)
        ranks_f = ranks.float()

        for k in ks:
            hit = (ranks <= k).float()
            sums[f'HR@{k}']   += hit.sum().item()
            sums[f'NDCG@{k}'] += (hit / torch.log2(ranks_f + 1)).sum().item()
        sums['MRR'] += (1.0 / ranks_f).sum().item()
        n += targets.size(0)

    return {k: v / n for k, v in sums.items()}


def fmt_metrics(m):
    return ' | '.join(f'{k}={v:.4f}' for k, v in m.items())
```

### Cell: full training run

```python
def train_model(name, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-5,
                patience=8):
    print(f"\n{'='*60}\nTraining on {name}\n{'='*60}")
    train_loader, val_loader, test_loader, edge_index, meta = make_loaders(
        name, batch_size=batch_size)

    model = NextPOIModel(
        n_pois=meta['n_pois'],
        n_users=meta['n_users'],
        d_p=128, d_u=64, d_c=32, d_h=128, d_hidden=256, dropout=0.2,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_hr10 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, edge_index)
        val_metrics = evaluate_model(model, val_loader, edge_index)
        history.append({'epoch': epoch, 'train_loss': loss, **val_metrics})
        print(f"[{name}] Epoch {epoch:02d} | train_loss={loss:.4f} | val: {fmt_metrics(val_metrics)}")

        # Always save latest (so we can resume after disconnect)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'history': history,
        }, os.path.join(ckpt_dir, 'latest.pt'))

        # Save best by val HR@10
        if val_metrics['HR@10'] > best_hr10:
            best_hr10 = val_metrics['HR@10']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val HR@10={best_hr10:.4f})")
            break

    # Final test using best checkpoint
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best.pt')))
    test_metrics = evaluate_model(model, test_loader, edge_index)
    print(f"\n[{name}] TEST (best epoch {best_epoch}): {fmt_metrics(test_metrics)}")

    # Save results
    pd.DataFrame(history).to_csv(
        os.path.join(PROJECT_ROOT, 'results', f'{name}_history.csv'), index=False)
    with open(os.path.join(PROJECT_ROOT, 'results', f'{name}_test.json'), 'w') as f:
        json.dump({'best_epoch': best_epoch, **test_metrics}, f, indent=2)

    return model, test_metrics, history
```

### Cell: run training on NYC

```python
model_nyc, test_nyc_metrics, history_nyc = train_model('NYC', epochs=50, patience=8)
```

Expected progression on NYC (approximate):
- Epoch 1: train_loss ~7.5, val HR@10 ~0.05 (random ≈ 10/5000 = 0.002, so already learning)
- Epoch 5: train_loss ~5.5, val HR@10 ~0.20
- Epoch 10–15: train_loss ~4.5, val HR@10 ~0.30, val HR@1 ~0.13
- Plateau or slight decline by epoch 25–35

Total training time: ~1.5–2 hours on T4. If much longer, decrease batch size or shorten epochs.

### Cell: run training on TKY

```python
model_tky, test_tky_metrics, history_tky = train_model('TKY', epochs=50, patience=8)
```

TKY is larger (more POIs, more sequences) — expect ~2.5–3 hours.

---

## 9. Reading the results

After both runs finish, the test metrics are in `results/{NYC,TKY}_test.json`. Compare to the literature:

```python
def print_comparison():
    for city in ['NYC', 'TKY']:
        with open(os.path.join(PROJECT_ROOT, 'results', f'{city}_test.json')) as f:
            r = json.load(f)
        print(f"\n=== {city} ===")
        print(f"  HR@1  = {r['HR@1']:.4f}")
        print(f"  HR@5  = {r['HR@5']:.4f}")
        print(f"  HR@10 = {r['HR@10']:.4f}")
        print(f"  NDCG@5 = {r['NDCG@5']:.4f}")
        print(f"  NDCG@10= {r['NDCG@10']:.4f}")
        print(f"  MRR    = {r['MRR']:.4f}")

print_comparison()
print("\nLiterature reference (HR@1 / Acc@1 on Foursquare, from LLM4POI Table 3):")
print("  LSTM:     NYC=0.13, TKY=0.13")
print("  STGCN:    NYC=0.18, TKY=0.17")
print("  STAN:     NYC=0.22, TKY=0.20")
print("  GETNext:  NYC=0.24, TKY=0.23")
print("  STHGCN:   NYC=0.27, TKY=0.30")
```

If your HR@1 lands in the 0.13–0.18 band, the baseline is working as expected. Higher than 0.20 with this architecture is suspicious — re-check for leakage in the train/val/test split or graph construction.

---

## 10. Running everything end-to-end

Put it all together. Sequence to run in Colab:

1. **Section 2** cells (3 cells): install + mount + imports — about 1 minute
2. **Section 3** cells (2 cells): download + load — about 2 minutes
3. **Section 4** cells (3 cells): preprocessing for NYC + TKY — about 2 minutes
4. **Section 5** cell: build graphs — about 1 minute
5. **Section 6** cells: define model + smoke test — instant
6. **Section 7** cells: dataset + loaders — instant (datasets are built per-call)
7. **Section 8** cells: training functions + train NYC + train TKY — about 4 hours total
8. **Section 9** cell: print comparison

Total: ~4–5 hours. Colab free tier sessions cap at 12h, so you have headroom.

**If you get disconnected during training**, the latest checkpoint is on Drive at `checkpoints/{NYC,TKY}/latest.pt`. To resume, load it and continue:

```python
ckpt = torch.load(os.path.join(PROJECT_ROOT, 'checkpoints/NYC/latest.pt'))
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1
```

(Wrap your training loop to start from `start_epoch` if resuming.)

---

## 11. Troubleshooting

### Training loss doesn't decrease at all

Most likely a label-shift bug — predicting input instead of next POI. Run the **memorization test**: train on the first 50 examples for 200 epochs and check that train accuracy approaches 100%. If it can't memorize 50 examples, your labels are wrong somewhere.

### Loss is `nan` after a few steps

Almost always an exploding gradient on the GRU. Check that gradient clipping (`clip_grad_norm_`) is actually being called. Reduce learning rate to 5e-4.

### `RuntimeError: CUDA out of memory`

The full graph passes through GCN every forward pass; if `n_pois` is unusually large (e.g. you forgot to filter), the embedding table can blow memory. Reduce `batch_size` to 32 or 16. Or check that `iterative_filter` actually ran.

### HR@10 stays near zero forever

Most common cause: indexing bug. Check that `poi_idx` values in your DataFrame are all `< meta['n_pois']`. A quick assert:
```python
assert train_df['poi_idx'].max() < meta_nyc['n_pois']
```

### `pack_padded_sequence` errors

`lengths` must be on CPU (the `.cpu()` call in the model handles this) and contain values `≥ 1`. Sequences of length 0 will crash. The dataset class skips these but double-check.

### Val HR@10 increases then drops sharply

Overfitting — the dataset is small. Increase dropout to 0.4, reduce embedding dim to 64, or stop earlier. Patience=5 instead of 8.

### TKY results are much worse than NYC

Possible — TKY has more POIs and sequences, which is harder. But if the gap is huge (e.g. NYC=0.15, TKY=0.05), there's likely a bug in the data loading. Re-run preprocessing for TKY from scratch.

### Graph has many isolated nodes

A POI with degree 0 won't get any signal from neighbors. Check the warning printed by `build_hybrid_graph`. If many isolated POIs:
- Increase `knn_k` from 10 to 20
- Decrease `covisit_threshold` from 3 to 2

---

## What you'll have at the end

Deliverables in `/content/drive/MyDrive/poi-rec/`:

```
data/
├── raw/                          # original tsv files
└── processed/
    ├── NYC/
    │   ├── train.parquet
    │   ├── val.parquet
    │   ├── test.parquet
    │   ├── poi_coords.npy
    │   ├── edge_index.pt
    │   └── meta.json
    └── TKY/ (same)
checkpoints/
├── NYC/{best.pt, latest.pt}
└── TKY/{best.pt, latest.pt}
results/
├── NYC_history.csv     # epoch-by-epoch metrics
├── NYC_test.json       # final test results
├── TKY_history.csv
└── TKY_test.json
```

Plus a working notebook you can re-run, share, or extend.

---

## What to do once it works

1. Run **ablations**: re-train with one component disabled at a time. The roadmap document has the four ablation configurations to run.
2. Run with **3 different random seeds** (change `SEED` and re-run training) to report mean ± std.
3. Sweep **hyperparameters** on val set: `lr ∈ {5e-4, 1e-3, 2e-3}`, `d_p ∈ {64, 128, 256}`.
4. Write the **thesis chapter** using the architecture spec from Section 1 and the results table from Section 9.

When you're ready to push beyond the baseline (path B in the roadmap), the four highest-leverage changes are listed there: inner-product scoring, ST-GRU gates, hour-of-day features, weighted graph edges. Each is a sub-chapter of incremental improvement over this baseline.

Good luck.
