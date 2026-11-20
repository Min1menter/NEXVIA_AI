import numpy as np
import pytest

pytest.importorskip("torch_geometric")
from sklearn.metrics import roc_auc_score
from fraudshield.graph import build_graph, train_gnn
from fraudshield.synthetic import generate


def test_graph_builds_and_gnn_finds_rings():
    df = generate(n_users=800, n_txn=30_000, n_rings=15, fraud_rate=0.01, seed=5)
    g = build_graph(df)
    assert g["x"].shape[0] > g["n_users"] and g["cluster_size"].max() >= 5
    cfg = dict(hidden=32, layers=2, epochs=80, lr=0.01, dropout=0.2, patience=20)
    _, scores, m = train_gnn(g, cfg, seed=0)
    assert roc_auc_score(g["y"], scores) > 0.8
