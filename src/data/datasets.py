"""CheXpert dataset, patient-level splits, and transforms."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_transforms  # noqa: F401 (re-export)

from src.labels import LABELS  # noqa: F401  (re-exported)

_PATIENT_RE = re.compile(r"(patient\d+)")


def add_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patient"] = df["Path"].str.extract(_PATIENT_RE)[0]
    return df


def prepare_labels(df: pd.DataFrame, uncertain: str = "ones") -> pd.DataFrame:
    """CheXpert: 1 positive, 0 negative, -1 uncertain, NaN not mentioned.
    uncertain='ones' (U-Ones) or 'zeros' (U-Zeros). NaN -> 0."""
    df = df.copy()
    df[LABELS] = df[LABELS].fillna(0.0)
    df[LABELS] = df[LABELS].replace(-1.0, 1.0 if uncertain == "ones" else 0.0)
    return df


def patient_split(df: pd.DataFrame, val_frac=0.1, seed=42):
    """Split by patient so no patient appears in both train and val (avoids leakage)."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["patient"]))
    tr, va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    assert set(tr["patient"]).isdisjoint(va["patient"])
    return tr, va


class CheXpertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.y = self.df[LABELS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.root / self.df.loc[i, "Path"])
        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(self.y[i])
