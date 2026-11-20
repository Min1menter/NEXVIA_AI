# API (OpenAPI at `/docs`)
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness + model version |
| POST | `/score?commit=true` | Score one transaction. `commit=false` = dry run (state not updated, nothing stored) |
| GET | `/alerts?status=&min_score=&limit=` | Alert queue, highest score first |
| PATCH | `/alerts/{id}` | `status` in new/investigating/confirmed_fraud/false_positive, `assignee`, `feedback_label` (0/1), `note`. Needs `X-API-Key` if `API_KEY` is set. Confirming/rejecting sets the label automatically |
| GET | `/stats` | scored, alerts, alert rate, status counts, reviewed alert precision |
| GET | `/drift/latest`, `/drift/report.html` | latest drift JSON / Evidently report |
| POST | `/admin/reload` | hot-reload models after retraining (API key) |
| GET | `/metrics` | Prometheus |

`POST /score` body: `txn_id, user_id, timestamp (ISO or epoch), amount>=0, type (PAYMENT|TRANSFER|CASH_OUT|CASH_IN|DEBIT), device_id, card_id, country, merchant_id, balance_before?`

Response: `score, xgb_score, graph_score, cluster_size, threshold, decision (alert|allow), reasons[], model_version, latency_ms, alert_id`.

Kafka topics: `transactions` (input, key=user_id), `fraud-scores` (output), `transactions-dlq`.
