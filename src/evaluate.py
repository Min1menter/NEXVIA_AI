"""Honest evaluation of the chest X-ray classifier on the official CheXpert valid set.

Produces in --out:
  metrics.json     per-class AUROC + 95% patient-level bootstrap CI, n_pos, ECE
  calibration.png  reliability diagram (all labels pooled)
  subgroups.csv    AUROC by sex and age band
  failures.png     highest-confidence false positives / false negatives for one label
  failures.csv

    python -m src.evaluate
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_section  # noqa: E402
from src.data.datasets import CheXpertDataset, add_patient_id, get_transforms, prepare_labels  # noqa: E402
from src.labels import LABELS  # noqa: E402
from src.models.densenet_cls import load_checkpoint  # noqa: E402
from src.train_cls import predict  # noqa: E402


def auc_or_none(y, p):
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def bootstrap_ci(y, p, groups, n=1000, seed=0):
    """Cluster (patient-level) bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    aucs = []
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        a = auc_or_none(y[idx], p[idx])
        if a is not None:
            aucs.append(a)
    if len(aucs) < 50:
        return None, None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def ece_and_bins(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, curve = 0.0, []
    for b in range(bins):
        m = ids == b
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        curve.append((conf, acc, int(m.sum())))
    return float(ece), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--dataset-dir", default="CheXpert-v1.0-small")
    ap.add_argument("--ckpt", default="runs/cls/best.pt")
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--failure-label", default="Pleural Effusion")
    ap.add_argument("--workers", type=int, default=4)
    ap.set_defaults(**load_section("evaluate"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    train_args = ckpt.get("args", {})

    df = add_patient_id(pd.read_csv(Path(args.root) / args.dataset_dir / "valid.csv"))
    if train_args.get("frontal_only"):
        df = df[df["Frontal/Lateral"] == "Frontal"]
    df = prepare_labels(df, train_args.get("uncertain", "ones")).reset_index(drop=True)

    ds = CheXpertDataset(df, args.root, get_transforms(train_args.get("img_size", 224), train=False))
    p, y = predict(model, DataLoader(ds, 64, num_workers=args.workers), device)
    groups = df["patient"].values

    # --- per-class AUROC with CI, ECE
    per_class = {}
    for i, name in enumerate(LABELS):
        a = auc_or_none(y[:, i], p[:, i])
        lo, hi = bootstrap_ci(y[:, i], p[:, i], groups, args.bootstrap) if a is not None else (None, None)
        ece, _ = ece_and_bins(y[:, i], p[:, i])
        per_class[name] = {"auroc": a, "ci95": [lo, hi], "n_pos": int(y[:, i].sum()),
                           "n": int(len(y)), "ece": ece}
    aucs = [v["auroc"] for v in per_class.values() if v["auroc"] is not None]
    summary = {"mean_auroc": float(np.mean(aucs)) if aucs else None, "n_studies": int(len(df)),
               "checkpoint_epoch": ckpt.get("epoch"), "per_class": per_class}
    pooled_ece, curve = ece_and_bins(y.ravel(), p.ravel())
    summary["pooled_ece"] = pooled_ece
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # --- calibration plot
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    plt.plot([c for c, _, _ in curve], [a for _, a, _ in curve], "o-", label=f"model (ECE {pooled_ece:.3f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration (all labels pooled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=150)
    plt.close()

    # --- subgroup analysis
    age_band = pd.cut(df["Age"], [0, 40, 65, 200], labels=["<40", "40-64", "65+"], right=False)
    rows = []
    for gname, series in (("Sex", df["Sex"]), ("Age", age_band)):
        for level in pd.Series(series).dropna().unique():
            m = (series == level).values
            for i, name in enumerate(LABELS):
                if m.sum() < 20 or y[m, i].sum() < 5 or (1 - y[m, i]).sum() < 5:
                    continue
                rows.append({"group": gname, "level": str(level), "label": name,
                             "n": int(m.sum()), "n_pos": int(y[m, i].sum()),
                             "auroc": auc_or_none(y[m, i], p[m, i])})
    pd.DataFrame(rows).to_csv(out / "subgroups.csv", index=False)

    # --- failure gallery (highest-confidence mistakes)
    j = LABELS.index(args.failure_label)
    neg, pos = np.where(y[:, j] == 0)[0], np.where(y[:, j] == 1)[0]
    fp = neg[np.argsort(-p[neg, j])[:5]]
    fn = pos[np.argsort(p[pos, j])[:5]]
    fails = pd.DataFrame({"type": ["FP"] * len(fp) + ["FN"] * len(fn),
                          "path": df.loc[np.r_[fp, fn], "Path"].values,
                          "prob": p[np.r_[fp, fn], j]})
    fails.to_csv(out / "failures.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.5))
    for ax in axes.ravel():
        ax.axis("off")
    for k, (idx, kind) in enumerate([(i, "FP") for i in fp] + [(i, "FN") for i in fn]):
        ax = axes[0 if kind == "FP" else 1, k % 5]
        ax.imshow(Image.open(Path(args.root) / df.loc[idx, "Path"]).convert("L"), cmap="gray")
        ax.set_title(f"{kind} p={p[idx, j]:.2f}", fontsize=9)
    fig.suptitle(f"Confident errors: {args.failure_label} (top: false positives, bottom: false negatives)")
    plt.tight_layout()
    plt.savefig(out / "failures.png", dpi=120)
    plt.close()
    print(json.dumps({k: summary[k] for k in ("mean_auroc", "pooled_ece", "n_studies")}, indent=2))


if __name__ == "__main__":
    main()
