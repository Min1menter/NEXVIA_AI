import numpy as np
import pandas as pd
from fraudshield.blend import Stacker, combine, run_blend, stack_features
from fraudshield.config import load_params


def make_df(informative_graph=True, n_users=600, per_user=40, seed=0):
    """informative: fraud = ring account AND risky txn type. XGBoost only sees 'risky', the (noisy, overlapping) graph score
    only sees 'ring' -> the right combination is an AND, which a weighted average of probabilities handles badly.
    not informative: the graph score is pure noise and XGBoost alone already separates fraud."""
    rng = np.random.default_rng(seed)
    ring = rng.random(n_users) < 0.15
    g = np.where(ring, rng.beta(5, 2, n_users), rng.beta(2, 5, n_users)) if informative_graph else rng.random(n_users)
    rows = []
    for u in range(n_users):
        risky = rng.random(per_user) < 0.4
        if informative_graph:
            fraud = (ring[u] & risky & (rng.random(per_user) < 0.8)).astype(int)
            x = np.where(risky, 0.25, 0.02) * np.exp(rng.normal(0, 0.3, per_user))
        else:
            fraud = (rng.random(per_user) < 0.03).astype(int)
            x = np.where(fraud == 1, 0.5, 0.02) * np.exp(rng.normal(0, 0.3, per_user))
        rows.append(pd.DataFrame(dict(user_id=f"U{u}", xgb_score=np.clip(x, 1e-4, 0.99), graph_score=g[u],
                                      cluster_size=8.0 if ring[u] else 1.0, is_fraud=fraud, amount=100.0,
                                      ring_id=np.where(ring[u], 0, -1))))
    df = pd.concat(rows, ignore_index=True)
    df["split"] = np.where(df.user_id.str[1:].astype(int) % 2 == 0, "val", "test")
    return df


def params(min_gain=0.01):
    p = load_params()
    p["blend"]["bootstrap"] = 30
    p["blend"]["stack"]["min_gain"] = min_gain
    return p


def test_stack_beats_weighted_when_signals_interact():
    res, meta = run_blend(make_df(True), params())
    sel, t = res["selection_on_validation"], res["test"]
    assert meta["blend_method"] == "stack" and sel["stack_oof_ap"] > sel["weighted_ap"] + 0.05
    assert t["stack"]["pr_auc"] > t["weighted"]["pr_auc"] > t["xgb"]["pr_auc"]
    assert t["bootstrap_ap_gain_vs_xgb"]["stack"]["ci_low"] > 0


def test_falls_back_to_weighted_when_graph_is_noise():
    _, meta = run_blend(make_df(False, seed=3), params(min_gain=0.03))
    assert meta["blend_method"] == "weighted"


def test_combine_matches_serving_math_and_handles_missing_graph():
    _, meta = run_blend(make_df(True), params())
    x, g, c = np.array([0.25, 0.25]), np.array([0.9, np.nan]), np.array([8.0, np.nan])
    out = combine(meta, x, g, c)
    np.testing.assert_allclose(out, Stacker.from_dict(meta["stacker"]).predict(stack_features(x, g, c)))
    assert out[0] > out[1] and np.isfinite(out).all()      # ring-linked account outranks unknown account


def test_ablation_runs_and_orders_variants():
    from fraudshield.ablation import run_ablation
    res = run_ablation(make_df(True), params())
    assert set(res) >= {"xgb only", "xgb + cluster size", "xgb + GNN score", "full stack"}
    assert res["full stack"]["test_ap"] > res["xgb only"]["test_ap"] + 0.1
