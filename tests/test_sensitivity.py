import numpy as np
import pandas as pd
import pytest

import cityImage as ci
from cityImage.pipeline import lynch_classes
from cityImage.scoring import DEFAULT_GLOBAL_COMPONENT_WEIGHTS


def _buildings(n=60, heights=True, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "area": rng.lognormal(5, 1, n),
            "2dvis": rng.uniform(0, 1e4, n),
            "neigh": rng.integers(0, 30, n).astype(float),
            "road": rng.uniform(0, 50, n),
            "cult": rng.integers(0, 3, n).astype(float),
            "prag": rng.uniform(0, 1, n),
        }
    )
    if heights:
        frame["height"] = rng.uniform(3, 60, n)
        frame["3dvis"] = rng.uniform(0, 1e3, n)
        frame["fac"] = rng.uniform(10, 500, n)
    return frame


def test_random_runs_compare_with_the_given_weights():
    runs = ci.landmark_weight_sensitivity(_buildings(), n_samples=30, top_k=10)
    assert len(runs) == 30
    assert (runs["varied"] == "all").all()
    for column in ("spearman", "top_k_overlap", "same_class", "strong_kept"):
        assert runs[column].between(0, 1).all(), column
    # a 20% perturbation leaves the ranking close to the original
    assert runs["spearman"].min() > 0.8


def test_weights_keep_their_sums():
    runs = ci.landmark_weight_sensitivity(_buildings(), n_samples=10)
    components = runs[[f"w_{c}" for c in DEFAULT_GLOBAL_COMPONENT_WEIGHTS]].sum(axis=1)
    assert np.allclose(components, 1.0)
    assert np.allclose(runs[["w_area", "w_2dvis", "w_neigh", "w_road"]].sum(axis=1), 1.0)
    assert np.allclose(runs[["w_fac", "w_height", "w_3dvis"]].sum(axis=1), 1.0)


def test_one_at_a_time_varies_each_active_weight_both_ways():
    runs = ci.landmark_weight_sensitivity(_buildings(), method="one_at_a_time", perturbation=0.3)
    expected = {
        "fac", "height", "3dvis", "area", "2dvis", "neigh", "road",
        "vScore", "sScore", "cScore", "pScore",
    }  # fmt: skip
    assert set(runs["varied"]) == expected
    assert len(runs) == 2 * len(expected)
    assert sorted(runs["factor"].unique()) == pytest.approx([0.7, 1.3])


def test_without_heights_the_visual_weights_are_left_alone():
    runs = ci.landmark_weight_sensitivity(_buildings(heights=False), method="one_at_a_time")
    assert not {"vScore", "fac", "height", "3dvis"} & set(runs["varied"])


def test_runs_are_reproducible():
    a = ci.landmark_weight_sensitivity(_buildings(), n_samples=5, random_state=3)
    b = ci.landmark_weight_sensitivity(_buildings(), n_samples=5, random_state=3)
    pd.testing.assert_frame_equal(a, b)


def test_a_tiny_perturbation_changes_nothing():
    runs = ci.landmark_weight_sensitivity(_buildings(), n_samples=5, perturbation=1e-9)
    assert runs["spearman"].min() == pytest.approx(1.0)
    assert (runs["top_k_overlap"] == 1.0).all()
    assert (runs["same_class"] == 1.0).all()


def test_invalid_arguments():
    buildings = _buildings()
    with pytest.raises(ValueError, match="method"):
        ci.landmark_weight_sensitivity(buildings, method="sobol")
    with pytest.raises(ValueError, match="perturbation"):
        ci.landmark_weight_sensitivity(buildings, perturbation=1.5)
    with pytest.raises(ValueError, match="indexes"):
        ci.landmark_weight_sensitivity(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="no buildings"):
        ci.landmark_weight_sensitivity(buildings.iloc[:0])


def test_lynch_classes_count_down_from_the_strongest():
    scaled = [0.0, 0.1, 0.125, 0.3, 0.6, 0.75, 1.0, float("nan")]
    assert lynch_classes(scaled).tolist() == [0, 0, 4, 3, 2, 1, 1, 0]
