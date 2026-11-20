# Model card & limitations
**Purpose:** rank card/account transactions by fraud risk for analyst review. Educational; not for production payment decisions.
**Models:** XGBoost on 22 behavioural features; GraphSAGE node classifier on the account-device-card-counterparty graph; weighted blend; cost-optimised threshold.
**Metrics:** PR-AUC (primary), precision/recall at threshold, expected savings; accuracy is deliberately not reported (<1% fraud).
**Known limitations**
* Synthetic fraud patterns are simpler than real fraud; real data will need more features (merchant risk, IP/geo-velocity, device fingerprinting).
* Graph scores are batch-refreshed (train-period graph), so brand-new rings are invisible until the next retrain; an inductive/streaming GNN is future work.
* Feedback labels come only from alerted transactions (selection bias); unflagged fraud is never labelled. Mitigate with random audit sampling.
* Velocity uses exponential decay, an approximation of hard-window counts.
* Fairness: country/device features can proxy for protected attributes; monitor alert rate and precision by segment before any real use.
* Latency measured on one process; no load test beyond replay.

**Real-data findings (IEEE-CIS):** modest absolute performance (PR-AUC 0.134, 3.9x baseline) with 22 behavioural features and a proxy user id; geo-change feature is uninformative (country nearly constant); graph/stacking gave no significant gain; drift monitor window must span at least a week.

**PaySim findings:** the first 500k rows contain only ~233 frauds (~25 in the test period), so metrics are very uncertain; detection relies on the balance-drain feature; history/device/graph features carry no signal because accounts are single-use and devices/cards are derived from a hash.
