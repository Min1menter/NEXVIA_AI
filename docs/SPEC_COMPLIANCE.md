# Spec compliance - milestone by milestone
Legend: DONE = run and verified with output | WRITTEN = code exists, never executed by the author | TODO = needs you.

| # | Requirement (from the brief) | Where | Status / evidence |
|---|---|---|---|
| 1 | Dataset (IEEE-CIS / PaySim / ...) + EDA noting class imbalance | `data.py` loaders, `scripts/eda.py`, `reports/eda.json/png` | **DONE** on two real datasets: IEEE-CIS (590,540 txns, 3.5% fraud, 27.6:1) and **PaySim (500,000 txns, 0.047% fraud, 2,145:1 - the "under 1%" case)**. ULB / BAF / Elliptic loaders are not implemented. |
| 2 | Features: velocity, amount vs user average, device & location changes, stored in Feast so training and serving are identical | `features.py`, `feature_repo/`, `scripts/feast_parity.py` | **DONE** on synthetic data via Feast (apply, materialize, point-in-time parity identical, scoring through Feast online store, online-vs-offline difference 0.0). On real IEEE-CIS the same feature code ran with the in-memory store (difference 0.0); Feast itself was not run on IEEE. |
| 3 | XGBoost with class weights or SMOTE; PR-AUC; MLflow | `train_xgb.py` | **DONE** (class weights; PR-AUC 0.134 vs 0.034 baseline on IEEE, 0.692 vs 0.007 on synthetic; runs in `mlflow.db`). SMOTE option is **WRITTEN**, not run. |
| 4 | Graph of accounts, devices, cards; PyTorch Geometric GNN for fraud rings; blend with XGBoost | `graph.py`, `blend.py`, `ablation.py` | **DONE** (GraphSAGE; stack/weighted blend chosen on validation). Result: big gain on synthetic planted rings, **no significant gain on real IEEE-CIS** - report both. Analyst "linked accounts" view (`/rings/{user}`) is **WRITTEN**. |
| 5 | Kafka producer replaying transactions; FastAPI consumer fetching features and returning a score | `producer.py`, `consumer.py`, `api.py`, `scoring.py` | Scoring path **DONE** (`replay_local`: 20,000 txns, p50/p99 3.7/7.0 ms on IEEE, features fetched from state store). Kafka producer/consumer and the FastAPI service over HTTP: **WRITTEN, unit-tested (API) but never run against a real broker** - TODO on your machine. |
| 6 | Evidently drift reports + automatic retraining trigger | `monitoring.py`, `retrain.py`, `drift_demo.py` | Evidently HTML report **DONE** (Kaggle); PSI/KS trigger logic **DONE** (unit tests). End-to-end retrain run (`drift_demo --retrain`) **WRITTEN**, not run. |
| 7 | Analyst dashboard (alerts, case status, feedback labels); Kubernetes (Minikube) | `dashboard/app.py`, `k8s/`, `docker-compose.yml` | **WRITTEN**, never executed. TODO: run Compose or Minikube and record it. |

| Deliverable | Status |
|---|---|
| Working demo (deployed or screen-recorded) | TODO - record `docs/DEMO_RUNBOOK.md` (fallback: no-Kafka path + Kaggle screenshots; say honestly what ran) |
| GitHub repo, clean commits, README, setup | Ready - `docs/GITHUB_SETUP.md` |
| Architecture diagram + data/API docs | DONE - `docs/ARCHITECTURE.md`, `docs/API.md` |
| Test results + performance numbers | DONE - `docs/TEST_RESULTS.md`, `docs/RESULTS.md` (19 tests pass in the sandbox; 17 on Kaggle before the last additions) |
| 10-minute presentation | Script ready - `docs/PRESENTATION_SCRIPT.md`; build slides |
| Individual reflections | TODO - yours; `docs/REFLECTION_TEMPLATE.md` |

Beyond the brief (optional, mention briefly or skip): stacking combiner, ablation + bootstrap CIs, drift demo script, Prometheus metrics, CI workflow, rule override, cost-based threshold.
