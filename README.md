# FraudShield - real-time fraud detection with graph-based ring detection

Streaming fraud scoring: **Kafka -> FastAPI/consumer -> Feast features -> XGBoost + GraphSAGE blend -> analyst dashboard**,
with MLflow tracking, Evidently/PSI drift monitoring and an automatic retraining trigger. Runs locally, in Docker Compose, or on Minikube.

> Educational project. Data is synthetic (or public Kaggle data); do not use for real payment decisions.

## Results at a glance (synthetic data, time-based split, single run - see `docs/RESULTS.md`)
| Test period | XGBoost | Final (stacked with graph features) |
|---|---|---|
| PR-AUC (baseline 0.007) | 0.692 | **0.812** (+0.121, 95% CI [0.089, 0.156]) |
| Fraud-ring recall at alert threshold | 0.53 | **0.99** |
| Net savings (fraud caught - review cost) | 36.6k | **41.9k (+14%)** |
**Real data - PaySim (0.047% fraud):** XGBoost test PR-AUC 0.571 vs 0.00025 baseline, ROC 0.99, driven by the balance-drain feature; only ~25 test fraud cases; graph part adds nothing (single-use accounts).
**Real data - IEEE-CIS (590k txns, 3.5% fraud):** XGBoost PR-AUC **0.134 vs 0.034 baseline (3.9x)**, ROC 0.76; graph features and stacking gave no significant gain (validation selection kept the weighted blend). Online features match offline exactly on real data too.
Synthetic-data ablation: a simple device/card cluster-size feature gives most of the PR-AUC gain; the GNN's extra PR-AUC is not statistically established (CI includes 0).
Online features match offline exactly (difference 0.0, also through Feast). Not verified by the author: Kafka, Docker Compose, Minikube, Streamlit.
Kaggle walkthrough: `fraudshield_kaggle_runbook.ipynb`. Demo + presentation: `docs/DEMO_RUNBOOK.md`, `docs/PRESENTATION_SCRIPT.md`. Checklist: `docs/FINAL_CHECKLIST.md`.

## What is where
| Step | Code |
|---|---|
| 1. Data + EDA (imbalance <1%) | `src/fraudshield/{synthetic,data}.py`, `scripts/eda.py` -> `reports/eda.json/png` |
| 2. Features + Feast | `src/fraudshield/features.py` (single code path), `feature_repo/`, `scripts/feast_parity.py` |
| 3. XGBoost (class weights/SMOTE, PR-AUC, MLflow) | `src/fraudshield/train_xgb.py` |
| 4. Graph + GNN + blend | `src/fraudshield/{graph,blend}.py` |
| 5. Kafka producer / consumer / FastAPI | `producer.py`, `consumer.py`, `api.py`, `scoring.py` |
| 6. Drift + retrain trigger | `monitoring.py`, `retrain.py` |
| 7. Dashboard + Kubernetes | `dashboard/app.py`, `k8s/`, `docker-compose.yml` |

Extras beyond the brief: SHAP-style per-alert reasons (XGBoost `pred_contribs`), cost-based alert threshold,
rule override for classic account-takeover, ring clusters (shared device/card), idempotent consumer + dead-letter topic,
Prometheus `/metrics`, API-key protected analyst actions, feedback labels flowing into retraining, training/serving parity tests.

## Quick start (no Docker, no Kafka)
```bash
python -m venv .venv && source .venv/bin/activate
make install                  # torch + torch_geometric may need the platform wheels from pytorch.org / pyg.org
make test
make pipeline                 # data -> features -> EDA -> XGBoost -> GNN -> blend  (writes artifacts/meta.json, metrics)
make replay                   # replays the test period through the real scorer, fills the DB, prints online metrics
make api &                    # http://localhost:8000/docs
make dashboard                # http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://localhost:5000
make drift                    # drift report + retrain decision (needs >=500 scored txns from `make replay`)
```
Real datasets: `FRAUD_PARAMS=params.ieee.yaml` (IEEE-CIS) or `params.paysim.yaml` (PaySim) deep-merge over `params.yaml`; edit the CSV paths inside.
On Kaggle use `fraudshield_realdata_runbook.ipynb` (sets everything up, keeps each dataset's outputs in its own folder). See "Data caveats" below.

## Full stack (Docker Compose, Feast + Redis + Kafka + Postgres + MLflow)
```bash
cp .env.example .env
docker compose --profile pipeline run --rm pipeline     # builds data, applies Feast, trains, materializes
docker compose up -d --build
docker compose --profile replay run --rm producer      # stream transactions through Kafka
# dashboard http://localhost:8501   API http://localhost:8000/docs   MLflow http://localhost:5000
```
Minikube: see `k8s/README.md`.

## Design decisions worth defending in the presentation
* **Training/serving parity by construction.** Features depend only on a tiny per-user state (running mean/variance, two
  exponentially-decayed counters, last device/country) kept in Feast. `txn_features(state, txn)` is the only feature code;
  offline we replay history, online we read state from Feast, call the same function, then push the new state back.
  `tests/test_features.py` asserts exact equality, `scripts/feast_parity.py` checks Feast point-in-time retrieval too.
* **Velocity = decayed counts** (not hard 1h/24h windows) so the state is a handful of scalars Feast can store and update incrementally.
* **Time-based splits** (60/20/20) - never random - because fraud patterns drift. The last 20% is what Kafka replays.
* **PR-AUC, precision@recall and expected savings** instead of accuracy; the alert threshold maximises
  `fraud amount caught - review_cost * alerts` on the validation period.
* **Graph model has no label leakage:** graph built from the train period only; combiner and threshold fitted on validation; test is untouched.
  Users without a graph score fall back to XGBoost alone.
* **Stacking instead of averaging.** The GNN scores *accounts*; ring members also make ordinary transactions, so averaging probabilities
  gave the GNN zero weight in our first Kaggle run. `blend.py` now fits a logistic-regression stacker on
  `[logit(xgb), logit(gnn), has_graph, xgb*gnn, log cluster size]` (an AND-like combination), compares it with the weighted blend
  using *grouped out-of-fold* validation PR-AUC, and keeps the stacker only if it wins by `blend.stack.min_gain`. Each method gets its own
  validation-fitted threshold; `meta.json` reports all three on test plus a user-level bootstrap CI for the PR-AUC gain.
* **At-least-once Kafka** + idempotent DB writes by `txn_id`; messages keyed by `user_id` preserve per-user order.
* **Retraining triggers:** (a) share of features with PSI >= 0.2 exceeds 30%, or (b) precision of analyst-reviewed alerts falls
  below 25%. Retraining adds analyst-labelled rows, re-blends, re-materializes Feast and hot-reloads the API (`/admin/reload`).

## Data caveats (state these honestly)
* **Synthetic** (default): planted account-takeover bursts and fraud rings, ~0.8% fraud. Great for demonstrating the architecture; absolute numbers are *not* comparable to real-world performance.
* **PaySim** (first 500k rows: 0.047% fraud, only ~25 fraud cases in the test period - very uncertain numbers) has no device/card/geo and nearly no repeat originators, so velocity/device features are weak; the loader derives device/card/country from an account hash.
* **IEEE-CIS** has no user id; the loader builds a proxy (`card1 + addr1 + email domain`) and uses `DeviceInfo` when present.
* Feedback labels come only from *alerted* transactions (selection bias) - see `docs/MODEL_CARD.md`.

## Repo layout
```
src/fraudshield/  synthetic, data, features, store, train_xgb, graph, blend, model_bundle, scoring,
                  db, api, consumer, producer, replay_local, monitoring, retrain
feature_repo/     Feast entity/feature views + store configs (local sqlite, docker redis)
dashboard/        Streamlit analyst console        k8s/  Minikube manifests        tests/  pytest suite
docs/             ARCHITECTURE, API, RESULTS, MODEL_CARD, PRESENTATION_OUTLINE, REFLECTION_TEMPLATE
```
