"""Tiny helper: let params.yaml provide argparse defaults (so `dvc repro` tracks params)."""
from pathlib import Path

import yaml


def load_section(name: str, path: str = "params.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return (yaml.safe_load(p.read_text()) or {}).get(name, {}) or {}
