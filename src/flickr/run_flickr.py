"""Run the Flickr trip-recommendation harness and tabulate vs the literature.

Entry points:
    run_city(...)              one city, a set of recommender factories -> metrics
    run_classical_baselines(.) all cities, the classical baselines (CPU, local)
    run_neural(...)            all cities, the learned pointer model (GPU, Colab)
    format_comparison(...)     markdown: our pairs-F1/F1 beside the published numbers

CLI (local, classical only — neural is for Colab):
    py -3.11 -m src.flickr.run_flickr --data_dir <dir> [--out results/flickr/our_results.json]

The harness reproduces the canonical protocol (leave-one-out, endpoints given,
length>=3) so the numbers sit on the same scale as :mod:`src.flickr.published`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

from src.flickr.baselines import classical_factories
from src.flickr.data import CITIES, load_city
from src.flickr.evaluate import RecommenderFactory, evaluate_loocv
from src.flickr.published import published_for_city


def run_city(
    data_dir: str,
    city: str,
    factories: Dict[str, RecommenderFactory],
    *,
    min_len: int = 3,
    train_pool: str = "ge3",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every factory on one city via leave-one-out CV.

    Args:
        data_dir: directory with the city's CSVs.
        city: full city name.
        factories: ``{method_name: factory}``.
        min_len: minimum trajectory length (default 3).
        train_pool: ``"ge3"`` (default, faithful) or ``"all"`` (see :func:`evaluate_loocv`).
        verbose: print a one-line summary per method.

    Returns:
        ``{method_name: metrics_dict}``.
    """
    c = load_city(data_dir, city)
    out: Dict[str, Dict[str, float]] = {}
    for name, fac in factories.items():
        m = evaluate_loocv(c, fac, min_len=min_len, train_pool=train_pool)
        out[name] = m
        if verbose:
            print(
                f"[{city:10s}] {name:14s} F1={m['F1']:.3f}  pairs-F1={m['pairs-F1']:.3f}  "
                f"exact={m['exact-match']:.3f}  n={int(m['n'])}"
            )
    return out


def run_classical_baselines(
    data_dir: str,
    cities: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    *,
    min_len: int = 3,
    train_pool: str = "ge3",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run the four classical baselines over every city (CPU, local).

    Args:
        data_dir: directory with the CSVs.
        cities: cities to run (default all five).
        out_path: if given, write the results JSON here.
        min_len: minimum trajectory length (default 3).
        train_pool: ``"ge3"`` (default, faithful) or ``"all"``.

    Returns:
        ``{city: {method: metrics}}``.
    """
    cities = cities or list(CITIES)
    facs = classical_factories()
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for city in cities:
        results[city] = run_city(
            data_dir, city, facs, min_len=min_len, train_pool=train_pool
        )
    if out_path:
        _dump(results, out_path, {"protocol": "leave-one-out; endpoints given; "
              f"length>={min_len}; train_pool={train_pool}", "kind": "classical"})
    return results


def run_neural(
    data_dir: str,
    device,
    cities: Optional[List[str]] = None,
    config=None,
    out_path: Optional[str] = None,
    *,
    min_len: int = 3,
    train_pool: str = "ge3",
    greedy_and_beam: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run the learned pointer model over every city (GPU, Colab).

    Imports :mod:`src.flickr.pointer` lazily so the classical path never needs a
    GPU-trained model. Trains a fresh model per leave-one-out fold.

    Args:
        data_dir: directory with the CSVs.
        device: torch device (``"cuda"`` strongly recommended).
        cities: cities to run (default all five).
        config: :class:`~src.flickr.pointer.PointerConfig` (defaults if None).
        out_path: if given, write the results JSON here.
        min_len: minimum trajectory length (default 3).
        train_pool: ``"ge3"`` (default, faithful) or ``"all"``.
        greedy_and_beam: also report a greedy (beam=1) variant beside the beam.

    Returns:
        ``{city: {method: metrics}}`` with method ``"Pointer"`` (beam from config)
        and optionally ``"Pointer-greedy"``.
    """
    from src.flickr.pointer import PointerConfig, pointer_factory

    cfg = config or PointerConfig()
    cities = cities or list(CITIES)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for city in cities:
        facs: Dict[str, RecommenderFactory] = {
            "Pointer": pointer_factory(device, cfg, min_len=min_len),
        }
        if greedy_and_beam:
            greedy_cfg = PointerConfig(**{**cfg.__dict__, "beam": 1})
            facs["Pointer-greedy"] = pointer_factory(device, greedy_cfg, min_len=min_len)
        results[city] = run_city(
            data_dir, city, facs, min_len=min_len, train_pool=train_pool
        )
    if out_path:
        _dump(results, out_path, {"protocol": "leave-one-out; endpoints given; "
              f"length>={min_len}; train_pool={train_pool}", "kind": "neural",
              "config": cfg.__dict__})
    return results


def run_personalization(
    data_dir: str,
    device,
    cities: Optional[List[str]] = None,
    *,
    epochs: int = 30,
    dim: int = 64,
    seed: int = 0,
    beam: int = 3,
    min_len: int = 3,
    train_pool: str = "ge3",
    out_path: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Personalization ablation: the pointer WITHOUT vs WITH the user embedding.

    Runs leave-one-out twice per city — ``use_user=False`` then ``use_user=True``,
    identical otherwise — and reports the pairs-F1 / F1 of each plus the delta.
    Gives the comparable (Flickr) chapter a real "based on user preferences"
    number. Whether the delta is positive depends on how often a test user recurs
    in training (see the user-repeat statistics).

    Args:
        data_dir: directory with the CSVs.
        device: torch device.
        cities: cities to run (default all five).
        epochs, dim, seed, beam: pointer hyperparameters (shared by both arms).
        min_len: minimum trajectory length (default 3).
        train_pool: ``"ge3"`` or ``"all"``.
        out_path: if given, write the results JSON here.

    Returns:
        ``{city: {"no_user": metrics, "user": metrics, "delta_pairs_f1": float}}``.
    """
    from src.flickr.pointer import PointerConfig, pointer_factory

    cities = cities or list(CITIES)
    results: Dict[str, Dict[str, object]] = {}
    for city in cities:
        c = load_city(data_dir, city)
        row: Dict[str, object] = {}
        for tag, use_user in (("no_user", False), ("user", True)):
            cfg = PointerConfig(dim=dim, epochs=epochs, beam=beam, seed=seed, use_user=use_user)
            m = evaluate_loocv(
                c, pointer_factory(device, cfg, min_len=min_len),
                min_len=min_len, train_pool=train_pool,
            )
            row[tag] = {"pairs-F1": m["pairs-F1"], "F1": m["F1"], "n": m["n"]}
        row["delta_pairs_f1"] = row["user"]["pairs-F1"] - row["no_user"]["pairs-F1"]  # type: ignore[index]
        results[city] = row
        print(
            f"[{city:10s}] no_user pairs-F1={row['no_user']['pairs-F1']:.3f} | "  # type: ignore[index]
            f"user pairs-F1={row['user']['pairs-F1']:.3f} | "  # type: ignore[index]
            f"delta={row['delta_pairs_f1']:+.3f}"
        )
    if out_path:
        _dump(results, out_path, {
            "experiment": "personalization — pointer user embedding on/off",
            "epochs": epochs, "dim": dim, "seed": seed, "train_pool": train_pool,
        })
    return results


def _dump(results: dict, out_path: str, meta: dict) -> None:
    """Write ``{meta, results}`` to ``out_path`` as JSON (creating dirs)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)


# ---------------------------------------------------------------------------
# Comparison formatting
# ---------------------------------------------------------------------------
def format_comparison(
    our_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "pairs-F1",
    cities: Optional[List[str]] = None,
) -> str:
    """Markdown table: our metric per city beside the published range + SOTA.

    Args:
        our_results: ``{city: {method: metrics}}`` from a run.
        metric: ``"pairs-F1"`` or ``"F1"``.
        cities: column order (default all five).

    Returns:
        A markdown string with one row per *our* method (per city), then the
        published min/median/max and SOTA method for context.
    """
    cities = cities or list(CITIES)
    pub_key = "pairs_F1" if metric == "pairs-F1" else "F1"

    methods = []
    for city in cities:
        for m in our_results.get(city, {}):
            if m not in methods:
                methods.append(m)

    head = "| Method (ours) | " + " | ".join(cities) + " |"
    sep = "|" + "---|" * (len(cities) + 1)
    lines = [head, sep]
    for m in methods:
        cells = []
        for city in cities:
            v = our_results.get(city, {}).get(m, {}).get(metric)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.append(sep)
    # published context rows
    for label, fn in (("published min", min), ("published max (SOTA)", max)):
        cells = []
        for city in cities:
            pub = published_for_city(city, pub_key)
            if pub:
                if label.startswith("published max"):
                    best = max(pub, key=pub.get)
                    cells.append(f"{pub[best]:.3f} ({best})")
                else:
                    cells.append(f"{min(pub.values()):.3f}")
            else:
                cells.append("—")
        lines.append(f"| *{label}* | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Run Flickr classical baselines (local).")
    ap.add_argument("--data_dir", required=True, help="dir with traj-*/poi-* or userVisits-*/POI-* CSVs")
    ap.add_argument("--out", default="results/flickr/our_results.json")
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--train_pool", choices=["ge3", "all"], default="ge3")
    args = ap.parse_args()

    res = run_classical_baselines(
        args.data_dir, out_path=args.out, min_len=args.min_len, train_pool=args.train_pool
    )
    print("\n=== pairs-F1: ours vs published ===")
    print(format_comparison(res, "pairs-F1"))
    print("\n=== F1: ours vs published ===")
    print(format_comparison(res, "F1"))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    _main()
