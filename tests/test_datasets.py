import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from src.data.datasets import LABELS, add_patient_id, patient_split, prepare_labels  # noqa: E402


def frame(n_patients=40, per_patient=3):
    rows = [{"Path": f"CheXpert-v1.0-small/train/patient{p:05d}/study{s}/view1_frontal.jpg"}
            for p in range(n_patients) for s in range(1, per_patient + 1)]
    df = pd.DataFrame(rows)
    for l in LABELS:
        df[l] = np.nan
    return df


def test_patient_ids_are_parsed():
    df = add_patient_id(frame(2, 1))
    assert list(df["patient"]) == ["patient00000", "patient00001"]


def test_split_has_no_patient_overlap():
    tr, va = patient_split(add_patient_id(frame()), val_frac=0.25, seed=1)
    assert set(tr["patient"]).isdisjoint(va["patient"]) and len(va) > 0


def test_uncertain_label_policies():
    df = frame(1, 1)
    df.loc[0, LABELS[:3]] = [1.0, -1.0, np.nan]
    ones = prepare_labels(df, "ones")
    zeros = prepare_labels(df, "zeros")
    assert list(ones.loc[0, LABELS[:3]]) == [1.0, 1.0, 0.0]   # U-Ones, NaN -> 0
    assert list(zeros.loc[0, LABELS[:3]]) == [1.0, 0.0, 0.0]  # U-Zeros
