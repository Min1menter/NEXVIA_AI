"""Build a clean submission zip (no secrets, caches, raw datasets, or run outputs).

    python scripts/package_submission.py                      # code + docs + reports (small)
    python scripts/package_submission.py --with-weights       # also include models/*.pt
    python scripts/package_submission.py --with-data          # also include data/dicom*, screenshots stay in docs/

Writes dist/medimaging-ai-submission.zip and prints what it contains.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "runs",
             "pgdata", ".idea", ".vscode"}
SKIP_PATHS = {"data/raw", ".dvc/cache", ".dvc/tmp"}
SKIP_NAMES = {".env", ".DS_Store", "config.local"}
SKIP_SUFFIX = {".pyc", ".zip", ".log"}


def keep(rel: Path, with_weights: bool, with_data: bool) -> bool:
    parts = set(rel.parts)
    posix = rel.as_posix()
    if parts & SKIP_DIRS or rel.name in SKIP_NAMES or rel.suffix in SKIP_SUFFIX:
        return False
    if any(posix == p or posix.startswith(p + "/") for p in SKIP_PATHS):
        return False
    if posix.startswith("data/") and not with_data and not posix.endswith(".dvc") and rel.name != ".gitignore":
        return False
    if posix.startswith("models/") and rel.suffix == ".pt" and not with_weights:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/medimaging-ai-submission.zip")
    ap.add_argument("--with-weights", action="store_true")
    ap.add_argument("--with-data", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and keep(p.relative_to(ROOT), args.with_weights, args.with_data)]
    assert not any(p.name == ".env" for p in files), "refusing to package a .env file"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("medimaging-ai") / p.relative_to(ROOT))
    total = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {total:.1f} MB uncompressed -> {out}")
    top = {}
    for p in files:
        top[p.relative_to(ROOT).parts[0]] = top.get(p.relative_to(ROOT).parts[0], 0) + 1
    for k, v in sorted(top.items()):
        print(f"  {k}: {v}")
    big = [p for p in files if p.stat().st_size > 100e6]
    if big:
        print("WARNING: files over 100 MB:", [str(b.relative_to(ROOT)) for b in big])


if __name__ == "__main__":
    main()
