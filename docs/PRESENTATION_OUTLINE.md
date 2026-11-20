# 10-minute presentation (suggested)
1. **Problem (1 min):** <1% fraud, cost asymmetry, rings evade per-transaction models.
2. **Design (2.5 min):** architecture diagram; Feast parity idea (one feature function); streaming path.
3. **Models (2 min):** XGBoost + imbalance handling; GNN on shared devices/cards; blend and cost-based threshold.
4. **Live demo (2 min):** `make replay` or Kafka producer -> alerts appear -> open case -> confirm fraud -> label saved -> drift/retrain tab.
5. **Results (1 min):** PR-AUC lift, ring recall with vs without GNN, latency, parity = 0 diff.
6. **Challenges (1 min):** parity, leakage (time split, train-only graph), Kafka ordering, Feast API/version quirks.
7. **Next steps (0.5 min):** streaming/inductive GNN, random-audit sampling, shadow deployment, calibration, auth/RBAC.
