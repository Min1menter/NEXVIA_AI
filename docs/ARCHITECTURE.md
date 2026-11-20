# Architecture
```mermaid
flowchart LR
  subgraph Batch["Batch / training (make pipeline, retrain job)"]
    D[Raw txns<br/>synthetic / IEEE-CIS / PaySim] --> F[features.py<br/>replay in time order]
    F --> OFF[(Feast offline<br/>state_snapshots.parquet)]
    F --> X[XGBoost<br/>class weights / SMOTE]
    D --> G[Graph builder<br/>user-device-card-counterparty]
    G --> GNN[GraphSAGE - PyG]
    X --> B[Stack / weighted combiner + threshold<br/>fitted on validation]
    GNN --> B
    X -.-> ML[(MLflow)]
    GNN -.-> ML
    B -.-> ML
    OFF -->|materialize| ON[(Feast online - Redis)]
    GNN -->|graph_scores| ON
  end
  subgraph Stream["Real-time"]
    P[Kafka producer<br/>replays test period] -->|transactions, key=user_id| K{{Kafka}}
    K --> C[Consumer / FastAPI scorer]
    C -->|get state + graph score| ON
    C -->|push updated state| ON
    C --> M[Model bundle<br/>XGB + stacker + rules]
    C -->|scores| K2{{fraud-scores}}
    C -->|failed msgs| DLQ{{transactions-dlq}}
    C --> PG[(Postgres<br/>scored, alerts, feedback)]
  end
  PG --> UI[Streamlit analyst console<br/>alerts, cases, feedback]
  UI -->|PATCH /alerts| API[FastAPI]
  API --> PG
  PG --> DR[Drift job: PSI/KS + Evidently]
  DR -->|drift or low alert precision| RT[Retrain job]
  RT --> Batch
  RT -->|POST /admin/reload| API
```
## Feature state (Feast `user_state`)
`n_txn, mean_amt, m2_amt, ew_cnt_short, ew_cnt_long, last_ts, last_device, last_country` + `graph_scores(graph_score, cluster_size)`.

## Engineered features (22)
Amount: `log_amount, amount_vs_avg, amount_zscore, drain_ratio, amt_mean`. Velocity: `velocity_short (1h), velocity_long (24h), time_since_last, n_prior_txn`.
Behaviour change: `is_new_user, device_changed, geo_changed, device_geo_changed`. Context: `hour, dow, type_*`.

## Failure handling
Poison message -> DLQ topic, offset committed. Duplicate delivery -> ignored by unique `txn_id`. Missing graph score -> XGBoost only.
Unknown user -> empty state (`is_new_user=1`). API `/health` stays up without a model; `/score` loads lazily.
