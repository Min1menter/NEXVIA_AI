# 10-minute presentation script (synthetic numbers = planted fraud rings; real numbers = IEEE-CIS; one run each)

**Slide 1 - Title & problem (0:45).** "Card/payment fraud is rare, costly and organised. We built FraudShield: real-time scoring, a graph model for fraud rings, and an analyst console - and tested it on real data."

**Slide 2 - Why it is hard (0:45).** Real data: PaySim 500k transactions at **0.047% fraud (2,145:1)** and IEEE-CIS 590k at 3.5% (27:1); our synthetic set 0.8%. Accuracy is meaningless ("always legit" = 96.5%), so we use PR-AUC against the prevalence baseline.

**Slide 3 - Architecture (1:15).** `docs/ARCHITECTURE.md`: producer -> Kafka (keyed by user) -> consumer/FastAPI -> Feast features -> XGBoost + graph score -> alert -> Postgres -> Streamlit; MLflow tracks runs; drift job triggers retraining.

**Slide 4 - Features and train/serve parity (1:00).** Velocity, amount vs the user's average, device and location change from ONE feature function; state in Feast. Online-vs-offline feature difference **0.0** on synthetic AND on real IEEE-CIS data; Feast point-in-time parity identical; Feast adds ~13 ms (p50 16 ms vs 3 ms).

**Slide 5 - Real data: XGBoost on IEEE-CIS (1:15).** Time-based split, class weights. Test PR-AUC **0.134 vs 0.034 baseline (3.9x)**, ROC 0.76. Honest read: clearly useful, modest in absolute terms - we use 22 behavioural features and a proxy user id (no real user id in the data), not the full IEEE feature set. Device change is a real signal (16.7% of fraud vs 7.3% of legit); location change carries none here (country nearly constant).

**Slide 6 - Why a graph model: synthetic rings (1:15).** Planted rings share devices/cards. GNN alone 0.90 account-level but only 0.18 per transaction; a weighted average gave it weight 0. Stacking on [XGBoost, GNN, cluster size, interaction] raised PR-AUC 0.692 -> **0.812** (+0.121, CI [0.089, 0.156]) and ring recall 0.53 -> 0.99, at the cost of ~50% more alerts.

**Slide 7 - Does it transfer to real data? No (1:00).** On IEEE-CIS the validation protocol *rejected* the stacker; test gains +0.001 to +0.004 PR-AUC with CIs including 0. GNN: account-level PR-AUC 0.34 (7x lift) but no usable transaction-level gain. Our proxy graph has no recoverable ring structure; the synthetic gains reflect planted rings, not real fraud.

**Slide 7b - PaySim (fold into slide 7, +0:20).** PaySim: PR-AUC 0.571 vs 0.00025 baseline, ROC 0.99, almost all from the `drain_ratio` (balance-drain) feature; only ~25 test frauds, so wide uncertainty; graph/history features useless (single-use accounts).

**Slide 8 - Lessons (0:45).** (1) Select methods on validation with confidence intervals, not on hope. (2) A simple device/card cluster-size feature gave most of the synthetic gain; the GNN's extra was not significant. (3) Synthetic benchmarks flatter complex models.

**Slide 9 - Live demo (1:15).** Dashboard alert -> reasons -> "Confirm fraud" (label saved for retraining) -> `python -m fraudshield.drift_demo` -> retrain trigger -> Evidently report.

**Slide 10 - Limits and next steps (0:45).** Limits: proxy users, only behavioural features, single seed, feedback labels only on alerted txns, 5,000-txn drift window too short on real data. Next: add the IEEE C/D/V and categorical columns as request-time features, better user proxy (card1+e-mail, addr1 as location), week-long drift windows, streaming GNN.

**Likely questions.** *Why is PR-AUC only 0.13?* Prevalence is 3.4% so that is 3.9x; ROC 0.76; limited by our deliberately small behavioural feature set and the user proxy. *Did the GNN help?* On planted rings yes (via stacking); on real IEEE-CIS no - we measured it. *Why not a random split?* Fraud drifts; random splits leak the future. *Is `dow` drift real?* No - the monitor window was 1.5 days.
