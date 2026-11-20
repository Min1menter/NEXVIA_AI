"""EDA: class imbalance, amount/time/behaviour differences. Writes reports/eda.md + PNGs."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fraudshield.config import DATA_DIR, ROOT

from fraudshield.config import REPORTS_DIR
out = REPORTS_DIR
out.mkdir(exist_ok=True)
df = pd.read_parquet(DATA_DIR / "raw" / "transactions.parquet")
f = pd.read_parquet(DATA_DIR / "features.parquet")
rate = df.is_fraud.mean()
summary = dict(n_txn=len(df), n_users=int(df.user_id.nunique()), fraud_rate=float(rate), imbalance_ratio=float((1 - rate) / rate),
               fraud_by_type=df.groupby("type").is_fraud.mean().round(5).to_dict(),
               median_amount_legit=float(df[df.is_fraud == 0].amount.median()), median_amount_fraud=float(df[df.is_fraud == 1].amount.median()),
               fraud_device_change_rate=float(f[f.is_fraud == 1].device_changed.mean()), legit_device_change_rate=float(f[f.is_fraud == 0].device_changed.mean()),
               fraud_geo_change_rate=float(f[f.is_fraud == 1].geo_changed.mean()), legit_geo_change_rate=float(f[f.is_fraud == 0].geo_changed.mean()))
json.dump(summary, open(out / "eda.json", "w"), indent=2)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(["legit", "fraud"], [(df.is_fraud == 0).sum(), (df.is_fraud == 1).sum()]); ax[0].set_yscale("log"); ax[0].set_title(f"Class balance ({rate:.2%} fraud)")
for lab, d in df.groupby("is_fraud"):
    ax[1].hist(d.amount.clip(upper=df.amount.quantile(.99)), bins=50, alpha=.6, density=True, label=["legit", "fraud"][lab])
ax[1].legend(); ax[1].set_title("Amount distribution")
f.groupby(f.timestamp.dt.floor("D")).is_fraud.mean().plot(ax=ax[2], title="Daily fraud rate")
plt.tight_layout(); fig.savefig(out / "eda.png", dpi=120)
print(json.dumps(summary, indent=2))
