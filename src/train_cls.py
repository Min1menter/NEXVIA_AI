"""Train DenseNet121 (ImageNet transfer learning) on CheXpert-small, multi-label.

    python -m src.train_cls                          # defaults from params.yaml
    python -m src.train_cls --epochs 1 --max-samples 2000   # quick smoke test

Splits: train/val = patient-level split of train.csv; test = official valid.csv.
Outputs (in --out): best.pt, metrics.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_section
from src.data.datasets import (CheXpertDataset, add_patient_id, get_transforms,
                               patient_split, prepare_labels)
from src.labels import LABELS
from src.models.densenet_cls import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_backbone_trainable(model, trainable: bool):
    for p in model.features.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in tqdm(loader, leave=False, desc="eval"):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def per_class_auroc(y_true, y_prob):
    out = {}
    for i, name in enumerate(LABELS):
        out[name] = None if len(np.unique(y_true[:, i])) < 2 else float(roc_auc_score(y_true[:, i], y_prob[:, i]))
    vals = [v for v in out.values() if v is not None]
    out["mean"] = float(np.mean(vals)) if vals else None
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw", help="folder containing CheXpert-v1.0-small")
    ap.add_argument("--dataset-dir", default="CheXpert-v1.0-small")
    ap.add_argument("--out", default="runs/cls")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--freeze-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--uncertain", choices=["ones", "zeros"], default="ones")
    ap.add_argument("--frontal-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.set_defaults(**load_section("train_cls"))
    return ap


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = Path(args.root) / args.dataset_dir
    train_df = add_patient_id(pd.read_csv(base / "train.csv"))
    test_df = add_patient_id(pd.read_csv(base / "valid.csv"))
    if args.frontal_only:
        train_df = train_df[train_df["Frontal/Lateral"] == "Frontal"]
        test_df = test_df[test_df["Frontal/Lateral"] == "Frontal"]
    train_df = prepare_labels(train_df, args.uncertain)
    test_df = prepare_labels(test_df, args.uncertain)
    if args.max_samples:
        train_df = train_df.sample(args.max_samples, random_state=args.seed)

    tr_df, va_df = patient_split(train_df, val_frac=0.1, seed=args.seed)
    print(f"train {len(tr_df)} | val {len(va_df)} | test {len(test_df)}")

    tr_ds = CheXpertDataset(tr_df, args.root, get_transforms(args.img_size, train=True))
    va_ds = CheXpertDataset(va_df, args.root, get_transforms(args.img_size, train=False))
    te_ds = CheXpertDataset(test_df, args.root, get_transforms(args.img_size, train=False))
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, drop_last=True, **kw)
    va_dl = DataLoader(va_ds, args.batch_size * 2, **kw)
    te_dl = DataLoader(te_ds, args.batch_size * 2, **kw)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, history = -1.0, []
    for epoch in range(args.epochs):
        set_backbone_trainable(model, epoch >= args.freeze_epochs)
        model.train()
        running = 0.0
        for x, y in tqdm(tr_dl, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()

        p, t = predict(model, va_dl, device)
        auc = per_class_auroc(t, p)
        history.append({"epoch": epoch + 1, "train_loss": running / len(tr_dl), "val_auroc": auc})
        print(f"epoch {epoch + 1}: loss {running / len(tr_dl):.4f} | val mean AUROC {auc['mean']:.4f}")
        if auc["mean"] is not None and auc["mean"] > best:
            best = auc["mean"]
            torch.save({"model": model.state_dict(), "labels": LABELS,
                        "args": vars(args), "epoch": epoch + 1}, out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    p, t = predict(model, te_dl, device)
    test_auc = per_class_auroc(t, p)
    print("TEST per-class AUROC:", json.dumps(test_auc, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"best_val_mean_auroc": best, "test_mean_auroc": test_auc["mean"],
         "test_auroc": test_auc, "history": history}, indent=2))


if __name__ == "__main__":
    main()
