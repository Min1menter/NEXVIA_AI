# Test & verification results
* **Unit tests:** 17 passed on Kaggle (Python 3.13); 2 tests were added after that run (ablation, drift-demo generator) and pass in the author's sandbox but were not re-run on Kaggle. Re-run with `pytest -q`.
* Covered: synthetic data properties, training/serving feature parity (exact), Welford statistics, blend/stack/threshold logic, drift PSI/trigger, API workflow (score -> alert -> case -> feedback, idempotency, validation, API key), GNN finds rings.
* **Feast parity:** 2,000 transactions, max abs difference 0.0 (`artifacts/feast_parity.json`).
* **Online replay:** 20,000 txns, p50/p99 3.0/5.6 ms in-memory; 2,000 txns via Feast online store p50/p99 16/27 ms; online-vs-offline feature difference 0.0 in both.
* **Model metrics:** `docs/RESULTS.md`. Not verified: Kafka, Docker Compose, Minikube, Streamlit dashboard (written but never executed), real IEEE-CIS/PaySim data.
