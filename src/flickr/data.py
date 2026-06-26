"""Flickr "User-POI visits" loader (Lim / Chen trip-recommendation benchmark).

This is the data layer for **Strategy D** of the thesis: re-running itinerary
recommendation on the small Flickr photo-trajectory datasets so that pairs-F1 /
F1 numbers are *directly comparable* to the published trip-recommendation
literature (Chen 2016, DeepTrip 2019, SelfTrip 2022, DeepAltTrip 2021, ...).

Two on-disk formats are supported and auto-detected:

* **Chen / DeepTrip canonical** (the format the published numbers are computed
  on; mirrored in the ``computationalmedia/tour-cikm16`` and ``gcooq/DeepTrip``
  repos):
    - ``poi-{stem}.csv``  : ``poiID,poiCat,poiLon,poiLat``
    - ``traj-{stem}.csv`` : ``userID,trajID,poiID,startTime,endTime,#photo,trajLen,poiDuration``
  Here a *trajectory* is the POI sequence of one ``trajID``, ordered by
  ``startTime``. Each row is already one distinct POI visit (Chen's
  preprocessing collapsed the raw photos), so grouping by ``trajID`` reproduces
  the published trajectories exactly.

* **Lim raw "User-POI visits"** (Kwan Hui Lim's data page):
    - ``POI-{City}.csv``        : ``poiID,poiName,lat,lon,theme``
    - ``userVisits-{City}.csv`` : ``photoID,userID,dateTaken,poiID,poiTheme,poiFreq,seqID,...``
  Here a *trajectory* is the ordered **distinct** POIs within one ``seqID``
  (consecutive duplicate POIs collapsed), ordered by ``dateTaken``.

Both reduce to the same in-memory representation: a :class:`FlickrCity` holding
POI coordinates and a list of :class:`Trajectory` objects with contiguous POI
indices ``0..n_pois-1``. POI ids and user ids are re-indexed to contiguous
integers (so they can index an embedding table) exactly as the Phase-1
preprocessing does for Foursquare.

All functions are pure: paths in, dataclasses out. No global state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Full city name -> the abbreviated file stem used by the Chen/DeepTrip CSVs.
CITY_ABBREV: Dict[str, str] = {
    "Toronto": "Toro",
    "Osaka": "Osak",
    "Glasgow": "Glas",
    "Edinburgh": "Edin",
    "Melbourne": "Melb",
}

#: The five cities shared by every paper in the comparison.
CITIES: Tuple[str, ...] = ("Toronto", "Osaka", "Glasgow", "Edinburgh", "Melbourne")


@dataclass(frozen=True)
class Trajectory:
    """One photo-trajectory = an ordered sequence of distinct POIs.

    Attributes:
        traj_id: the source ``trajID`` / ``seqID`` (for provenance only).
        user_idx: contiguous user index (0..n_users-1).
        pois: ordered tuple of contiguous POI indices, with consecutive
            duplicates already collapsed. ``len(pois)`` is the trajectory length.
    """

    traj_id: int
    user_idx: int
    pois: Tuple[int, ...]

    def __len__(self) -> int:  # convenience: len(traj) == number of POIs
        return len(self.pois)


@dataclass
class FlickrCity:
    """All data for one city, with contiguous POI / user indexing.

    Attributes:
        name: full city name (e.g. ``"Edinburgh"``).
        n_pois: vocabulary size |V| (number of distinct POIs).
        n_users: number of distinct users |U|.
        poi_coords: (n_pois, 2) float array of [lat, lon] in degrees, indexed by
            contiguous POI index.
        poi_cat: length-n_pois list of POI category/theme strings.
        trajectories: every trajectory in the city (all lengths; filter with
            :func:`trajectories_min_len`).
        poi2idx: raw ``poiID`` -> contiguous index.
        user2idx: raw ``userID`` -> contiguous index.
    """

    name: str
    n_pois: int
    n_users: int
    poi_coords: np.ndarray
    poi_cat: List[str]
    trajectories: List[Trajectory]
    poi2idx: Dict[int, int]
    user2idx: Dict[str, int]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _collapse_consecutive(seq: Sequence[int]) -> List[int]:
    """Collapse consecutive duplicate POIs in a sequence.

    A trajectory ``[a, a, b, b, b, a]`` becomes ``[a, b, a]`` — repeated *visits*
    to the same POI in a row are a single stop, but a genuine return later is
    kept. This matches the standard preprocessing for these datasets.

    Args:
        seq: raw ordered POI ids.

    Returns:
        List with consecutive runs collapsed to one element.
    """
    out: List[int] = []
    for p in seq:
        if not out or out[-1] != p:
            out.append(int(p))
    return out


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first column in ``candidates`` present in ``df`` (case-insensitive).

    Args:
        df: the DataFrame to search.
        candidates: ordered preferred column names.

    Returns:
        The actual column name in ``df``.

    Raises:
        KeyError: if none of the candidates is present.
    """
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {list(candidates)} found in columns {list(df.columns)}")


def load_poi_table(path: str) -> Tuple[Dict[int, int], np.ndarray, List[str]]:
    """Load a POI table (either ``poi-*.csv`` or ``POI-*.csv``).

    Args:
        path: path to the POI CSV. Recognised lon/lat column names are
            ``poiLon/poiLat`` (Chen) and ``lon|lng|long / lat`` (Lim); category
            comes from ``poiCat`` or ``theme``/``poiTheme``.

    Returns:
        ``(poi2idx, coords, cats)``:
            - poi2idx: raw ``poiID`` (int) -> contiguous index in sorted-id order.
            - coords: (n_pois, 2) float array of [lat, lon] in degrees.
            - cats: length-n_pois list of category strings ("" if absent).
    """
    df = pd.read_csv(path)
    id_col = _pick_column(df, ["poiID", "poiId", "poi_id", "id"])
    lon_col = _pick_column(df, ["poiLon", "lon", "lng", "long", "longitude"])
    lat_col = _pick_column(df, ["poiLat", "lat", "latitude"])
    try:
        cat_col: Optional[str] = _pick_column(df, ["poiCat", "theme", "poiTheme", "category"])
    except KeyError:
        cat_col = None

    poi_ids = sorted(int(x) for x in df[id_col].unique())
    poi2idx = {pid: i for i, pid in enumerate(poi_ids)}
    n = len(poi_ids)
    coords = np.zeros((n, 2), dtype=float)
    cats = ["" for _ in range(n)]
    for _, row in df.iterrows():
        idx = poi2idx[int(row[id_col])]
        coords[idx, 0] = float(row[lat_col])
        coords[idx, 1] = float(row[lon_col])
        if cat_col is not None:
            cats[idx] = str(row[cat_col])
    return poi2idx, coords, cats


def _load_visits(
    path: str,
    poi2idx: Dict[int, int],
    *,
    group_col_candidates: Sequence[str],
    time_col_candidates: Sequence[str],
    user_col_candidates: Sequence[str],
    poi_col_candidates: Sequence[str],
    user2idx: Dict[str, int],
) -> List[Trajectory]:
    """Build trajectories from a visit/photo table by grouping + time-ordering.

    Generic over both supported formats — the caller supplies the candidate
    column names for the grouping key (``trajID`` / ``seqID``), the timestamp,
    the user, and the POI.

    Args:
        path: path to the visits CSV.
        poi2idx: mapping from raw poiID to contiguous index (from the POI table).
        group_col_candidates: column names that identify one trajectory.
        time_col_candidates: column names giving visit time (for ordering).
        user_col_candidates: column names giving the user id.
        poi_col_candidates: column names giving the POI id.
        user2idx: mutated in place — raw userID -> contiguous index (assigned in
            first-seen order so it is deterministic for a given file).

    Returns:
        List of :class:`Trajectory`, one per group, with contiguous POI indices
        and consecutive duplicates collapsed. Order follows sorted group id.
    """
    df = pd.read_csv(path)
    gcol = _pick_column(df, group_col_candidates)
    tcol = _pick_column(df, time_col_candidates)
    ucol = _pick_column(df, user_col_candidates)
    pcol = _pick_column(df, poi_col_candidates)

    trajs: List[Trajectory] = []
    for gid, g in df.sort_values(tcol).groupby(gcol, sort=True):
        raw_pois = [int(p) for p in g[pcol].tolist()]
        # Skip POIs not present in the POI table (defensive; should not happen).
        mapped = [poi2idx[p] for p in raw_pois if p in poi2idx]
        pois = _collapse_consecutive(mapped)
        if not pois:
            continue
        user_raw = str(g[ucol].iloc[0])
        if user_raw not in user2idx:
            user2idx[user_raw] = len(user2idx)
        trajs.append(
            Trajectory(traj_id=int(gid), user_idx=user2idx[user_raw], pois=tuple(pois))
        )
    return trajs


def _locate(data_dir: str, prefix: str, city: str) -> Optional[str]:
    """Find ``{prefix}{full}.csv`` or ``{prefix}{abbrev}.csv`` in ``data_dir``.

    Args:
        data_dir: directory to search.
        prefix: filename prefix, e.g. ``"traj-"`` or ``"userVisits-"``.
        city: full city name (e.g. ``"Edinburgh"``).

    Returns:
        The matching path, or None if neither full nor abbreviated name exists.
    """
    stems = [city, CITY_ABBREV.get(city, city)]
    for stem in stems:
        cand = os.path.join(data_dir, f"{prefix}{stem}.csv")
        if os.path.isfile(cand):
            return cand
    return None


def load_city(data_dir: str, city: str) -> FlickrCity:
    """Load one city, auto-detecting the Chen/DeepTrip or Lim file layout.

    Tries the Chen/DeepTrip ``traj-*.csv`` + ``poi-*.csv`` layout first (the
    format the published numbers are computed on), then falls back to Lim's
    ``userVisits-*.csv`` + ``POI-*.csv``.

    Args:
        data_dir: directory containing the city's CSVs.
        city: full city name (one of :data:`CITIES`) or a custom stem.

    Returns:
        A fully populated :class:`FlickrCity`.

    Raises:
        FileNotFoundError: if no recognised file pair is found for the city.
    """
    user2idx: Dict[str, int] = {}

    poi_path = _locate(data_dir, "poi-", city)
    traj_path = _locate(data_dir, "traj-", city)
    if poi_path is not None and traj_path is not None:
        poi2idx, coords, cats = load_poi_table(poi_path)
        trajs = _load_visits(
            traj_path, poi2idx,
            group_col_candidates=["trajID", "trajId", "traj_id"],
            time_col_candidates=["startTime", "start_time", "dateTaken"],
            user_col_candidates=["userID", "userId", "user_id"],
            poi_col_candidates=["poiID", "poiId", "poi_id"],
            user2idx=user2idx,
        )
    else:
        poi_path = _locate(data_dir, "POI-", city)
        visits_path = _locate(data_dir, "userVisits-", city)
        if poi_path is None or visits_path is None:
            raise FileNotFoundError(
                f"No Flickr files for '{city}' in {data_dir} "
                f"(looked for traj-/poi- and userVisits-/POI- with full and "
                f"abbreviated stems)."
            )
        poi2idx, coords, cats = load_poi_table(poi_path)
        trajs = _load_visits(
            visits_path, poi2idx,
            group_col_candidates=["seqID", "seqId", "seq_id"],
            time_col_candidates=["dateTaken", "date_taken", "startTime"],
            user_col_candidates=["userID", "userId", "user_id"],
            poi_col_candidates=["poiID", "poiId", "poi_id"],
            user2idx=user2idx,
        )

    return FlickrCity(
        name=city,
        n_pois=len(poi2idx),
        n_users=len(user2idx),
        poi_coords=coords,
        poi_cat=cats,
        trajectories=trajs,
        poi2idx=poi2idx,
        user2idx=user2idx,
    )


def trajectories_min_len(city: FlickrCity, min_len: int = 3) -> List[Trajectory]:
    """Return the city's trajectories with at least ``min_len`` POIs.

    The published protocol evaluates on trajectories of length >= 3 (a query of
    length 2 is the trivial ``[start, end]`` with no ordering to recover).

    Args:
        city: the loaded city.
        min_len: minimum trajectory length to keep (default 3).

    Returns:
        Filtered list (a new list; the city is not mutated).
    """
    return [t for t in city.trajectories if len(t.pois) >= min_len]
