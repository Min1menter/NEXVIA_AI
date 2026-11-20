# Data documentation

> FOR EDUCATION ONLY - NOT FOR CLINICAL USE. All data is public and de-identified; nothing in this repo is real patient care data.

## Datasets
| | Chest X-ray classification | CT segmentation |
|---|---|---|
| Dataset | CheXpert-v1.0-small (Stanford ML Group) | Medical Segmentation Decathlon, Task09_Spleen |
| Obtained from | An unofficial Kaggle mirror (`ashery/chexpert`) because the official small version was no longer offered; original terms still apply | MONAI-hosted copy of the MSD archive (MD5 checked) |
| Used | Frontal views only (`frontal_only: true`) | 41 labelled training volumes (the 20 test volumes have no labels and are unused) |
| Labels | 14 observations from a text labeler (`src/labels.py`): 1 positive, 0 negative, -1 uncertain, blank = not mentioned | spleen mask |
| License / terms | Stanford CheXpert research-use terms; cite the paper; do not redistribute the dataset | CC-BY-SA 4.0 |

The full datasets are **not** in this repository. Only small demo DICOMs and the trained weights are shipped, tracked with DVC (`data/dicom`, `data/dicom_ct`, `models/*.pt`).

## Labels and splits
- **Uncertain labels:** U-Ones (uncertain -> positive) for the deployed model; blank -> 0. A U-Zeros model was trained as an ablation (`docs/MODEL_CARD.md`).
- **Chest X-ray splits:** by **patient** (no patient appears in two sets). Training and validation are a 90/10 patient-level split of CheXpert `train.csv` (seed 42);
  the test set is the official `valid.csv` (202 frontal studies, radiologist-consensus labels).
- **Segmentation split:** random 80/20 of the 41 labelled volumes (33 train / 8 validation, seed 42). There is no independent test set.
- Several test classes have 0-1 positives and cannot be scored; see the model card.

## Preprocessing and augmentation
**Chest X-ray** (`src/data/transforms.py`, shared by training and serving): grayscale -> 3 channels, resize to 224x224, ImageNet mean/std normalization.
Training augmentation: random affine (rotation up to 7 degrees, translation up to 3%, scale 0.95-1.05) and brightness/contrast jitter 0.2.
There is **no horizontal flip**, because left/right matters in chest X-rays. Wrapped 8-bit DICOMs are passed through unchanged at inference so intensities match training.

**CT** (`src/data/msd.py`): orientation RAS, resample to 1.5 x 1.5 x 2.0 mm, clip the HU window to [-57, 164] and scale to [0, 1].
Training: random 96^3 patches (4 per volume, positive/negative balanced), random flip (p=0.2, axis 0), random intensity shift (+-0.1, p=0.5).
Inference uses sliding-window prediction and maps the result back to the original DICOM space.

## DICOM mapping
CheXpert JPEGs are wrapped into DICOM (CR) by `src/data/dicom_wrap.py`; MSD NIfTI volumes become DICOM CT series via `src/data/nifti_to_dicom.py`.
UIDs are derived deterministically from patient/study/view names, so re-running produces identical files. Details in `docs/DICOM_NOTES.md`.

| Demo set | Content |
|---|---|
| `data/dicom/` | 50 chest X-ray files from 48 studies (official `valid.csv`, frontal) |
| `data/dicom_ct/` | `spleen_2` (90 slices) and `spleen_3` (40 slices), 5 mm slices |

The demo chest X-rays come from the official validation list and the demo CT volumes were seen in training or validation, so the demo is for illustration, not evaluation.

## References
- Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison", AAAI 2019.
- Antonelli et al., "The Medical Segmentation Decathlon", Nature Communications 2022.
- Huang et al., "Densely Connected Convolutional Networks", CVPR 2017.
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
- MONAI: Medical Open Network for AI (monai.io).
