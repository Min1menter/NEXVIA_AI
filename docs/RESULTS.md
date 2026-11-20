# Results

## Run 1 - Kaggle, synthetic data, weighted blend (real output)
* Tests 14/14 pass. 200,008 txns, 5,202 users, 0.804% fraud (imbalance ~123:1). Time split 60/20/20.
* XGBoost (class weights), test period: **PR-AUC 0.692** vs 0.0070 prevalence (~99x), ROC-AUC 0.992; precision@80% recall 0.38, @50% recall 0.73.
* At the validation-chosen threshold: precision 0.53, recall 0.72, ring-fraud recall 0.53, 9.4 alerts per 1,000 txns.
* GNN (account level): test PR-AUC 0.907, ROC-AUC 0.985. At transaction level (graph score used alone) only PR-AUC 0.176.
* Weighted blend tuned alpha = 1.0, so blend == XGBoost (no gain). Reason: the GNN score is per account, and ring members also make normal transactions.
* Online replay (`FEATURE_BACKEND=memory`) of 20,000 test txns through the real scorer: PR-AUC 0.726, 55% alert precision, p50/p99 latency 4.3 / 11.1 ms (in-process),
  **max online-vs-offline feature difference 0.0**. Drift check: 15% of features drifted (< 30% trigger) -> no retrain.

## Run 2 - Kaggle, synthetic data, stacking combiner (real output, latest notebook run)
Selection on validation (grouped out-of-fold): XGBoost AP 0.808, weighted blend 0.808 (alpha=1.0, i.e. identical), **stack 0.847 -> stack chosen** (min gain 0.01).

| Test period (40,002 txns, 0.70% fraud) | XGBoost | Weighted | Stack (chosen) |
|---|---|---|---|
| PR-AUC | 0.692 | 0.692 (= XGBoost) | **0.812** |
| ROC-AUC | 0.992 | 0.992 | 0.998 |
| Precision @ 80% recall | 0.38 | 0.38 | **0.66** |
| Precision @ 50% recall | 0.73 | 0.73 | **0.82** |
| At own validation-fitted threshold: precision / recall | 0.53 / 0.72 | same | 0.48 / 0.98 |
| Ring-fraud recall at threshold | 0.53 | 0.53 | **0.99** |
| Other (account-takeover) fraud recall | 1.00 | 1.00 | 0.96 |
| Alerts per 1,000 txns | 9.4 | 9.4 | 14.1 |
| Net savings (fraud amount caught - 5 x alerts) | 36,608 | 36,608 | **41,888 (+14.4%)** |

* PR-AUC gain of stack over XGBoost: **+0.121, 95% CI [0.089, 0.156]** (200-resample bootstrap over users, P(better) = 1.0).
* Trade-off to state openly: the stacker raises ~50% more alerts (precision 0.53 -> 0.48) in exchange for catching almost all fraud rings; at equal recall its precision is much higher.
* GNN alone at transaction level: PR-AUC 0.176 (account-level test PR-AUC 0.904) - the score is per account, so it must be combined with transaction signals.
* Online replay (`FEATURE_BACKEND=memory`) of 20,000 test txns through the real scorer: PR-AUC 0.811, recall 0.97, alert precision 0.48, p50/p99 latency 3.2 / 6.0 ms (in-process, no network),
  **max online-vs-offline feature difference 0.0**.
* Feast point-in-time parity on 2,000 txns: **max abs difference 0.0, identical = true** (Feast 0.4x, sqlite online store, Python 3.11).
* Scoring through the **Feast online store** (state read + pushed back per transaction, 2,000 test txns): online-vs-offline feature difference **0.0**; p50/p99 latency **16.4 / 27.5 ms** vs 3.0 / 5.6 ms with the in-memory store, i.e. Feast adds ~13 ms (local SQLite online store; Redis in Docker/K8s will differ).
* Drift check on 5,000 scored txns: 15% of features drifted (< 30% trigger) -> no retrain triggered. Evidently HTML report generated (`reports/drift_report.html`).
* **Ablation** (`python -m fraudshield.ablation`, test period, thresholds fitted on validation only):

| Stacker inputs | PR-AUC | Ring-fraud recall | Precision | Net savings |
|---|---|---|---|---|
| XGBoost only | 0.692 | 0.58 | 0.48 | 37.0k |
| + device/card cluster size | 0.804 | 0.71 | 0.63 | 38.9k |
| + GNN score | 0.801 | 0.92 | 0.58 | 41.5k |
| + GNN + cluster (no interaction) | 0.802 | 0.73 | 0.63 | 39.1k |
| Full stack (GNN + cluster + interaction) | 0.812 | 0.99 | 0.49 | 41.9k |

  Paired bootstrap over users (200 resamples), PR-AUC gain: cluster size vs XGBoost **+0.114 [0.083, 0.145]**; full stack vs cluster size **+0.008 [-0.006, +0.026]** (P(better)=0.81);
  GNN score vs cluster size **-0.004 [-0.021, +0.013]**.
  **Conclusion:** graph structure gives a large, statistically clear gain, and a simple cluster-size feature captures nearly all of it in PR-AUC terms. The GNN's extra PR-AUC is not statistically established.
  Its possible benefit is at the savings-optimal operating point (ring recall 0.99 vs 0.71, net savings +7.5% vs cluster size alone, at lower precision), but these threshold metrics come from a single run without confidence intervals.
* Run-to-run variation: two runs gave stack PR-AUC 0.8118 and 0.8125 (XGBoost identical); GNN training is not bit-reproducible, so threshold-dependent metrics moved slightly
  (precision 0.44 -> 0.48, alerts/1k 15.6 -> 14.1, savings 41.7k -> 41.9k). Quote numbers from one saved run.

Caveats: synthetic data with planted rings (absolute numbers are not real-world numbers); one seed; the GNN was trained on labels of ~70% of users, so many graph scores in val/test are in-sample for the GNN;
see the ablation below for how much of the gain really comes from the GNN versus the device/card cluster size.

## Run 3 - REAL data: IEEE-CIS (Kaggle, real output; first 20k test txns replayed online)
590,540 transactions, 90,375 proxy users (card1 + addr1 + e-mail domain), **fraud rate 3.50%** (imbalance 27.6:1). Time split 60/20/20 (test prevalence 3.44%).

| Test period (118k txns) | XGBoost | Weighted (chosen) | Stack |
|---|---|---|---|
| PR-AUC (baseline 0.0344) | **0.134** (3.9x lift) | 0.136 (3.9x) | 0.138 (4.0x) |
| ROC-AUC | 0.763 | 0.763 | 0.769 |
| Precision @ 50% recall | 0.116 | 0.118 | 0.123 |
| At own threshold: precision / recall | 0.23 / 0.12 | 0.22 / 0.14 | 0.27 / 0.07 |
| Alerts per 1,000 txns | 18.0 | 21.0 | 9.1 |
| Net savings (fraud caught - 20 x alerts) | 17.3k | 13.1k | 16.3k |

* **Selection on validation:** XGBoost AP 0.168, weighted 0.168 (alpha 0.95), stacker 0.168 -> the stacker did **not** win (min gain 0.01), so the weighted blend (almost pure XGBoost) was chosen.
* **Gain over XGBoost (bootstrap over users):** weighted +0.001 [-0.000, +0.003]; stack +0.004 [-0.000, +0.007]. Not distinguishable from zero.
* **GNN:** account-level test PR-AUC 0.343 (account fraud rate ~4.9%, ~7x lift, ROC 0.80) but only 0.069 PR-AUC (2x baseline) as a transaction score; graph covers 85% of test transactions.
* **Ablation:** every variant is within noise (test PR-AUC 0.134-0.138; all gains' 95% CIs include 0).
* **Online replay** (20,000 txns through the real scorer): PR-AUC 0.127, p50/p99 3.7 / 7.0 ms, **online-vs-offline feature difference 0.0 on real data**.
* **Drift check:** 10% of features flagged (below the 30% trigger) -> no retrain. The flagged `dow` (PSI 4.6) is an artifact of the monitor comparing the last 5,000 scored txns (~1.5 days) with a multi-month reference, not real drift; a production monitor needs a window of at least a week.
* **Data observations:** device change rate fraud 16.7% vs legit 7.3% (a real signal); geo-change rate is ~0 for both because `addr2` (country) is almost constant per user, so that feature carries no signal on this dataset. Fraud rate by product code: TRANSFER-mapped `C` 11.7%, `W` (PAYMENT) 2.0%.

**What this means.** With only our 22 behavioural features and a proxy user id, XGBoost is clearly better than chance (3.9x the baseline) but modest in absolute terms (ROC 0.76; the full IEEE-CIS feature set - C/D/V columns - is known to reach much higher). The gains from graph features and stacking seen on the synthetic data (+0.12 PR-AUC) did **not** transfer: the synthetic rings were planted, whereas real IEEE-CIS has no ring structure recoverable from this proxy graph. The validation-based selection correctly declined the stacker, which is the intended behaviour of the protocol.

## Run 4 - REAL data: PaySim (Kaggle, real output; first 500,000 rows in time order)
500,000 transactions, 499,953 distinct originators (almost every account appears once), **fraud rate 0.047% (233 frauds; imbalance 2,145:1)** - the "under 1%" case in the brief.
Fraud occurs only in TRANSFER (0.275% of them) and CASH_OUT (0.066%). Median amount: legit 81k, fraud 143k.
**Important caveat: the test period holds only ~25 fraud cases (validation ~22), so every number below is very uncertain - one case moves the PR-AUC a lot.**

| Test period (100k txns, prevalence 0.00025) | XGBoost |
|---|---|
| PR-AUC (baseline 0.00025) | **0.571** (about 2,300x lift; validation PR-AUC was 0.383) |
| ROC-AUC | 0.989 |
| Precision @ 80% recall / @ 50% recall | 0.43 / 0.52 |
| At validation-chosen threshold: precision / recall | 0.11 / 0.92 |
| Alerts per 1,000 txns | 2.1 |

* **What drives the model:** feature importance is dominated by `drain_ratio` (amount / balance before, ~0.50), then the transaction type indicators, hour, log amount and amount-vs-average. History, velocity, device and location features have zero importance - they cannot work here (accounts are almost all unique; devices/cards/countries are derived from an account hash, so "change" never happens: device/geo change rate is 0.0 for fraud and legit alike).
* **Graph model:** useless on this dataset, as expected. User-level GNN test PR-AUC 0.017 (accounts are single-use, so the graph has almost no structure); only **0.01% of test transactions** belong to an account seen in the training graph. The stacker's validation PR-AUC (0.328) was below XGBoost's (0.383), and its test gain is 0.000 [0.000, 0.000]. All ablation variants give the same test PR-AUC 0.571.
* **Bug found by this run (fixed in the code):** with no usable graph scores, every blend weight ties, and the tuner picked alpha = 0.0 (graph-only) instead of 1.0 (XGBoost-only), so the stored "weighted" blend was slightly different from XGBoost for the few transactions with a graph score. `tune_alpha` now breaks ties toward 1.0. Regenerate with `python -m fraudshield.blend`; **quote the XGBoost row as the PaySim result.**
* **Online replay** (20,000 txns, 7 frauds): PR-AUC 0.439, recall 1.00, 55 alerts (precision 0.13), p50/p99 latency 1.8 / 7.0 ms, **online-vs-offline feature difference 0.0**.
* **Drift check:** 5% of features flagged -> no retrain. The flagged `hour` (PSI 6.0) is again a window artifact (last 5,000 scored txns span a few hours of simulated time vs. the whole reference period).
* **Conclusion to state:** on PaySim the signal is the balance-drain pattern of fraudulent TRANSFER/CASH_OUT, which XGBoost finds well (ROC 0.99); the history-based and graph parts of the design add nothing because the dataset has no repeat users or real devices/cards. To get more reliable numbers, rerun with `max_rows: 2000000` (more fraud cases; slower, GNN epochs 20).
