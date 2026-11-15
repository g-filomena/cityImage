"""How much the landmarks depend on the weights of their score.

The global landmark score of Filomena, Verstegen & Manley (2019) weighs indexes into components
(visual, structural, cultural, pragmatic) and components into one score. The weights of the
paper's Table 1, the package defaults, are a reasoned choice, not a measured one.
``landmark_weight_sensitivity`` perturbs them and measures how far the ranking of the buildings,
their strongest buildings and their Lynch classes move from those obtained with the given weights.
Only the weighted sum is recomputed: the indexes come from the buildings, already scored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .data_utils import scaling_columnDF
from .landmarks import COMPONENT_INDEXES
from .pipeline import LYNCH_BINS, lynch_classes
from .scoring import (
    DEFAULT_GLOBAL_COMPONENT_WEIGHTS,
    DEFAULT_GLOBAL_INDEX_WEIGHTS,
    score_buildings_global,
)

METHODS = ("random", "one_at_a_time")


def landmark_weight_sensitivity(
    buildings: pd.DataFrame,
    *,
    index_weights: Mapping[str, float] | None = None,
    component_weights: Mapping[str, float] | None = None,
    method: str = "random",
    perturbation: float = 0.2,
    n_samples: int = 200,
    top_k: int = 20,
    lynch_bins: tuple[float, ...] = LYNCH_BINS,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """How stable the global landmark score is to its weights.

    The weights are perturbed, the score recomputed and compared with the score from the given
    weights (the paper's by default). Only the weights that act on these buildings are perturbed:
    an index without values (e.g. the visual ones, without heights) or a component without any
    index is left as it is. After each perturbation the component weights are brought back to a
    sum of 1, and the index weights of each component to their original sum.

    Parameters
    ----------
    buildings
        Buildings with the indexes of the score (``area``, ``2dvis``, ``neigh``, ``road``,
        ``cult``, ``prag``, and ``height``, ``3dvis``, ``fac`` where there are heights), such as
        ``CityImage.landmarks`` or the output of ``score_cityimage_buildings``.
    index_weights, component_weights
        The weights to start from; by default those of the paper (``DEFAULT_GLOBAL_*``).
    method
        ``"random"``: ``n_samples`` sets of weights, each weight multiplied by its own factor
        drawn uniformly in ``[1 - perturbation, 1 + perturbation]``. ``"one_at_a_time"``: each
        weight in turn multiplied by ``1 - perturbation`` and by ``1 + perturbation``, the
        others unchanged (before the sums are restored).
    perturbation
        The largest relative change of a weight, in (0, 1).
    n_samples
        The number of sets of weights for ``"random"``.
    top_k
        The number of strongest buildings compared by ``top_k_overlap``.
    lynch_bins
        The Lynch ranges of the rescaled score, as in ``image_of_the_city``.
    random_state
        Seed of the random perturbations.

    Returns
    -------
    pandas.DataFrame
        One row per set of weights: ``varied`` (the weight changed, or ``"all"``), ``factor``
        (its multiplier; NaN for ``"random"``), the weights used (``w_<name>``), and

        - ``spearman``: the rank correlation of the score with the original one;
        - ``top_k_overlap``: the share of the original ``top_k`` buildings still in the top
          ``top_k``;
        - ``same_class``: the share of buildings in the same Lynch class;
        - ``strong_kept``: the share of buildings of class 1-2 still in class 1-2 (NaN if
          there are none).

        ``df.describe()`` summarises a random run; for one-at-a-time, the rows of a weight show
        how much the ranking depends on it.

    Examples
    --------
    >>> img = ci.image_of_the_city("Susa, Italy", crs="EPSG:32632")
    >>> runs = ci.landmark_weight_sensitivity(img.landmarks)
    >>> runs[["spearman", "top_k_overlap", "same_class"]].describe()
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if not 0 < perturbation < 1:
        raise ValueError("perturbation must be in (0, 1)")
    if n_samples < 1 or top_k < 1:
        raise ValueError("n_samples and top_k must be at least 1")

    base_indexes = dict(index_weights or DEFAULT_GLOBAL_INDEX_WEIGHTS)
    base_components = dict(component_weights or DEFAULT_GLOBAL_COMPONENT_WEIGHTS)
    columns = {i for indexes in COMPONENT_INDEXES.values() for i in indexes} | {"height"}
    if len(buildings) == 0:
        raise ValueError("there are no buildings")
    frame = pd.DataFrame(buildings[[c for c in buildings.columns if c in columns]])
    if frame.columns.empty:
        raise ValueError("the buildings have none of the indexes of the landmark score")

    def has_values(index: str) -> bool:
        return index in frame.columns and bool(frame[index].notna().any())

    has_heights = has_values("height")
    active_indexes = [
        index
        for component, indexes in COMPONENT_INDEXES.items()
        if len(indexes) > 1 and (component != "vScore" or has_heights)
        for index in indexes
        if index in base_indexes and has_values(index)
    ]
    active_components = [
        component
        for component, indexes in COMPONENT_INDEXES.items()
        if component in base_components
        and (component != "vScore" or has_heights)
        and any(has_values(index) for index in indexes)
    ]

    def score(indexes: dict[str, float], components: dict[str, float]) -> pd.Series:
        scored = score_buildings_global(frame, index_weights=indexes, component_weights=components)
        return scored["gScore"]

    def classes(values: pd.Series) -> np.ndarray:
        return lynch_classes(scaling_columnDF(values), lynch_bins)

    base = score(base_indexes, base_components)
    base_rank = base.rank()
    base_class = classes(base)
    k = min(top_k, len(base))
    base_top = set(base.nlargest(k).index)
    strong = np.isin(base_class, (1, 2))

    def restore(factors: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        indexes = {i: w * factors.get(i, 1.0) for i, w in base_indexes.items()}
        for component_indexes in COMPONENT_INDEXES.values():
            members = [i for i in component_indexes if i in indexes]
            before = sum(base_indexes[i] for i in members)
            after = sum(indexes[i] for i in members)
            if members and after > 0:
                for i in members:
                    indexes[i] *= before / after
        components = {c: w * factors.get(c, 1.0) for c, w in base_components.items()}
        total = sum(components.values())
        return indexes, {c: w / total for c, w in components.items()}

    runs: list[tuple[str, float, dict[str, float]]] = []
    names = active_indexes + active_components
    if method == "random":
        rng = np.random.default_rng(random_state)
        for _ in range(n_samples):
            draws = rng.uniform(1 - perturbation, 1 + perturbation, len(names))
            runs.append(("all", np.nan, dict(zip(names, draws.tolist(), strict=True))))
    else:
        for name in names:
            for factor in (1 - perturbation, 1 + perturbation):
                runs.append((name, factor, {name: factor}))

    rows: list[dict[str, Any]] = []
    for varied, factor, factors in runs:
        indexes, components = restore(factors)
        new = score(indexes, components)
        new_class = classes(new)
        rows.append(
            {
                "varied": varied,
                "factor": factor,
                **{f"w_{c}": w for c, w in components.items()},
                **{f"w_{i}": w for i, w in indexes.items()},
                "spearman": float(base_rank.corr(new.rank())),
                "top_k_overlap": len(base_top & set(new.nlargest(k).index)) / k,
                "same_class": float(np.mean(new_class == base_class)),
                "strong_kept": (
                    float(np.mean(np.isin(new_class[strong], (1, 2)))) if strong.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
