import numpy as np
from fraudshield.blend import blend_scores, pick_threshold, tune_alpha


def test_blend_falls_back_to_xgb_when_no_graph_score():
    out = blend_scores([0.2, 0.8], [np.nan, 0.0], alpha=0.5)
    assert out[0] == 0.2 and np.isclose(out[1], 0.4)


def test_tune_alpha_prefers_informative_model():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(int)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 4000), 0, 1)
    noise = rng.random(4000)
    a, ap = tune_alpha(y, noise, good, 0.1)
    assert a <= 0.3 and ap > 0.5


def test_threshold_respects_review_cost():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    s = np.clip(y * 0.5 + rng.random(5000) * 0.5, 0, 1)
    amt = np.full(5000, 100.0)
    cheap = pick_threshold(y, s, amt, review_cost=0.1)
    pricey = pick_threshold(y, s, amt, review_cost=90.0)
    assert pricey["alerts"] <= cheap["alerts"] and pricey["precision"] >= cheap["precision"]


def test_tune_alpha_ties_prefer_xgboost_when_graph_scores_missing():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    xgb = np.clip(y * 0.4 + rng.random(2000) * 0.5, 0, 1)
    a, _ = tune_alpha(y, xgb, np.full(2000, np.nan), 0.05)      # graph never available -> every alpha ties
    assert a == 1.0
