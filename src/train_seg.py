"""Train a MONAI 3D U-Net on a Medical Segmentation Decathlon CT task (default Task09_Spleen).

    python -m src.train_seg            # defaults come from params.yaml (train_seg)

Outputs in --out: best.pt ({"model", "roi", "dice", "epoch"}) and metrics.json.
Copy best.pt to models/seg_unet.pt for the inference service.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose

from src.config import load_section
from src.data.msd import train_transforms, val_transforms
from src.models.unet3d_monai import build_unet


def make_files(task_dir: Path):
    meta = json.loads((task_dir / "dataset.json").read_text())
    return [{"image": str(task_dir / e["image"]), "label": str(task_dir / e["label"])}
            for e in meta["training"]]


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/msd")
    ap.add_argument("--task", default="Task09_Spleen")
    ap.add_argument("--out", default="runs/seg")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--roi", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--cache-rate", type=float, default=0.5)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.set_defaults(**load_section("train_seg"))
    return ap


def main():
    args = build_parser().parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    roi = tuple(args.roi)

    files = make_files(Path(args.root) / args.task)
    random.Random(args.seed).shuffle(files)
    n_val = max(1, int(0.2 * len(files)))
    train_files, val_files = files[n_val:], files[:n_val]
    print(f"train {len(train_files)} | val {len(val_files)}")

    tr_ds = CacheDataset(train_files, train_transforms(roi), cache_rate=args.cache_rate, num_workers=args.workers)
    va_ds = CacheDataset(val_files, val_transforms(), cache_rate=args.cache_rate, num_workers=args.workers)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    va_dl = DataLoader(va_ds, batch_size=1, num_workers=args.workers)

    model = build_unet().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dice = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in tr_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        rec = {"epoch": epoch + 1, "train_loss": total / max(1, len(tr_dl))}

        if (epoch + 1) % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                for v in va_dl:
                    vx, vy = v["image"].to(device), v["label"].to(device)
                    pred = sliding_window_inference(vx, roi, 4, model)
                    dice([post_pred(i) for i in decollate_batch(pred)],
                         [post_label(i) for i in decollate_batch(vy)])
                d = dice.aggregate().item()
                dice.reset()
            rec["val_dice"] = d
            if d > best:
                best = d
                torch.save({"model": model.state_dict(), "roi": list(roi), "dice": d,
                            "epoch": epoch + 1}, out / "best.pt")
        history.append(rec)
        print(rec)

    (out / "metrics.json").write_text(json.dumps({"best_val_dice": best, "history": history}, indent=2))


if __name__ == "__main__":
    main()
