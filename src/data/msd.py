"""MONAI transforms for Medical Segmentation Decathlon CT tasks (default: Task09_Spleen)."""
from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, RandFlipd,
                              RandShiftIntensityd, ScaleIntensityRanged, Spacingd)

KEYS = ["image", "label"]
SPACING = (1.5, 1.5, 2.0)      # mm, (x, y, z)
HU_MIN, HU_MAX = -57, 164      # soft-tissue window used for the spleen task


def _base():
    return [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(keys=KEYS, pixdim=SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=KEYS, source_key="image", allow_smaller=True),
    ]


def train_transforms(roi=(96, 96, 96)):
    return Compose(_base() + [
        RandCropByPosNegLabeld(keys=KEYS, label_key="label", spatial_size=roi, pos=1, neg=1,
                               num_samples=4, image_key="image", image_threshold=0),
        RandFlipd(keys=KEYS, spatial_axis=0, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])


def val_transforms():
    return Compose(_base())
