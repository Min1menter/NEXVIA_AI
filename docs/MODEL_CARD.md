# Model card

> **FOR EDUCATION ONLY - NOT FOR CLINICAL USE.** These models were trained for a course project. They have not been
> validated on any clinical population and must never inform patient care.

## Models

| | Chest X-ray classifier | CT spleen segmenter |
|---|---|---|
| Architecture | DenseNet121, ImageNet-pretrained, 14-way multi-label head | MONAI 3D residual U-Net (16-32-64-128-256) |
| Task | Predict 14 CheXpert observations per image | Segment the spleen on CT |
| Data | CheXpert-small (Stanford), frontal views | Medical Segmentation Decathlon Task09_Spleen |
| Training | `python -m src.train_cls` (frozen backbone for the first epochs, AdamW, BCE, AMP) | `python -m src.train_seg` (DiceCE loss, 96^3 patches, sliding-window validation) |
| Explanation | Grad-CAM on the last DenseNet feature map (top-3 findings) | Predicted mask overlay + volume |
| Weights file | `models/cls_best.pt` | `models/seg_unet.pt` |

## Intended use
Learning exercise: demonstrates an end-to-end flow (DICOM -> inference -> explanation -> human review -> audit).
**Out of scope:** diagnosis, triage, screening, second reading, any real patient data.

## Training details
- Splits are **patient-level**: train/validation from CheXpert `train.csv`; the official `valid.csv` is the held-out test set.
- Uncertain CheXpert labels (-1) are mapped with the U-Ones policy by default (`params.yaml: train_cls.uncertain`); unmentioned labels become 0.
- Augmentation: small affine + brightness/contrast. No horizontal flip (laterality matters).
- Input: 224x224, replicated to 3 channels, ImageNet normalization. Seeds and hyperparameters live in `params.yaml`.

## Results
Evaluated with `python -m src.evaluate` on the official CheXpert **valid** set (202 frontal studies), using the
epoch-7 checkpoint (chosen on a patient-level validation split of the training data, best validation mean AUROC 0.804).
The test set was used once. CIs are 95% patient-level bootstrap intervals. Note the label-source shift: training labels come from the
CheXpert text labeler, test labels from radiologist consensus.

| Summary | Mean AUROC |
|---|---|
| All 13 classes with at least one positive | 0.800 |
| 9 classes with >= 25 positives | 0.844 |
| 5 CheXpert competition labels (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) | 0.870 |

| Class | AUROC | 95% CI | Positives (of 202) | ECE | Note |
|---|---|---|---|---|---|
| No Finding | 0.897 | 0.844-0.942 | 26 | 0.062 |  |
| Enlarged Cardiomediastinum | 0.502 | 0.420-0.583 | 105 | 0.430 | chance level - see below |
| Cardiomegaly | 0.811 | 0.742-0.874 | 66 | 0.204 |  |
| Lung Opacity | 0.907 | 0.866-0.941 | 117 | 0.153 |  |
| Lung Lesion | not evaluable | - | 1 | 0.039 | only 1 positive(s) |
| Edema | 0.946 | 0.908-0.975 | 42 | 0.066 |  |
| Consolidation | 0.845 | 0.784-0.902 | 32 | 0.041 |  |
| Pneumonia | 0.794 | 0.587-0.941 | 8 | 0.062 | wide CI (few positives) |
| Atelectasis | 0.824 | 0.764-0.883 | 75 | 0.077 |  |
| Pneumothorax | 0.905 | 0.793-0.999 | 7 | 0.041 | wide CI (few positives) |
| Pleural Effusion | 0.927 | 0.891-0.958 | 64 | 0.053 |  |
| Pleural Other | not evaluable | - | 1 | 0.025 | only 1 positive(s) |
| Fracture | not evaluable | - | 0 | 0.047 | only 0 positive(s) |
| Support Devices | 0.937 | 0.904-0.966 | 99 | 0.054 |  |

Pooled ECE is 0.039, but this flatters the model: it pools many mostly-negative labels. Per-class ECE is much higher for
common findings (Enlarged Cardiomediastinum 0.43, Cardiomegaly 0.20, Lung Opacity 0.15), and the pooled reliability curve
(`reports/eval/calibration.png`) sits above the diagonal at low and mid scores, i.e. the model under-states probability. Treat the displayed
confidence as an uncalibrated score.

**Subgroups** (`reports/eval/subgroups.csv`, exploratory): male vs female AUROC differ by a few points per class with no consistent direction;
patients 65+ score lower on Cardiomegaly (0.78), Consolidation (0.79) and Atelectasis (0.76) than other age bands, but each cell has
only 5-60 positives and no significance test was run, so these are hypotheses, not findings. Age <40 has only 29 studies.

**Segmentation (spleen, CT):** MONAI 3D U-Net trained for 100 epochs on 33 MSD Task09_Spleen volumes (random 80/20 split of the 41 labelled
volumes, seed 42). Best validation Dice **0.917** (epoch 90, 8 validation volumes). Validation Dice first passed 0.8 at epoch 48 and was noisy
afterwards: the mean of the last 10 validation checks was 0.865 (range 0.816-0.917). Caveats: the 8 validation volumes were also used to pick the
checkpoint, so 0.917 is optimistic; there is no independent test set (the MSD test volumes ship without labels); and the data comes from a single
centre. Training loss was still falling slowly at epoch 100, so the model is probably not fully converged.

## Ablation: uncertain-label policy (U-Ones vs U-Zeros)
CheXpert marks some labels as "uncertain" (-1). The main model maps them to positive (U-Ones). To test whether that choice caused the poor
Enlarged Cardiomediastinum result, an otherwise identical model was trained with uncertain mapped to negative (U-Zeros; same seed, data and
hyperparameters; checkpoint epoch 6 vs 7). Same test set, same patient-level bootstrap.

| Class (>= 25 test positives) | Positives | U-Ones AUROC (95% CI) | U-Zeros AUROC (95% CI) | ECE U-Ones | ECE U-Zeros |
|---|---|---|---|---|---|
| No Finding | 26 | 0.897 (0.84-0.94) | 0.905 (0.86-0.95) | 0.06 | 0.07 |
| Enlarged Cardiomediastinum | 105 | 0.502 (0.42-0.58) | 0.569 (0.49-0.65) | 0.43 | 0.47 |
| Cardiomegaly | 66 | 0.811 (0.74-0.87) | 0.809 (0.74-0.87) | 0.20 | 0.23 |
| Lung Opacity | 117 | 0.907 (0.87-0.94) | 0.911 (0.87-0.94) | 0.15 | 0.22 |
| Edema | 42 | 0.946 (0.91-0.97) | 0.926 (0.88-0.96) | 0.07 | 0.07 |
| Consolidation | 32 | 0.845 (0.78-0.90) | 0.898 (0.85-0.94) | 0.04 | 0.11 |
| Atelectasis | 75 | 0.824 (0.76-0.88) | 0.773 (0.71-0.84) | 0.08 | 0.23 |
| Pleural Effusion | 64 | 0.927 (0.89-0.96) | 0.922 (0.89-0.95) | 0.05 | 0.09 |
| Support Devices | 99 | 0.937 (0.90-0.97) | 0.942 (0.91-0.97) | 0.05 | 0.08 |

| Summary | U-Ones | U-Zeros |
|---|---|---|
| Best validation mean AUROC (19k held-out training images) | 0.804 | 0.809 |
| Test mean AUROC, 9 classes with >= 25 positives | 0.844 | 0.851 |
| Test mean AUROC, 5 competition labels | 0.870 | 0.866 |
| Pooled ECE | 0.039 | 0.086 |

Findings:
- **Overall discrimination is the same.** Differences in the summary means are far smaller than the confidence intervals.
- **The 13-class test mean is misleading here** (0.800 vs 0.825): it is driven by Lung Lesion and Pleural Other, which have one test positive each. Use the 9-class mean.
- **Enlarged Cardiomediastinum improved but is not fixed**: 0.502 to 0.569 on test (the interval still includes 0.5) and 0.673 to 0.716 on validation. The label policy
  therefore explains part of the weakness at most; the rest is unresolved (the test labels are radiologist-assigned and this finding is positive in about half the test studies).
- **The best policy differs by label.** Consolidation is better with U-Zeros (+0.053 test, +0.027 validation); Atelectasis is better with U-Ones (-0.050 test, -0.015 validation).
- **U-Zeros is worse calibrated** (pooled ECE 0.086 vs 0.039; Atelectasis 0.08 to 0.23), so it under-states probabilities more.

**Decision:** the deployed model stays U-Ones (equal discrimination, better calibrated). Full per-class numbers: `reports/ablation_uncertainty_policy.csv`.

## Ethical considerations
Public, de-identified data only. The UI keeps a persistent education-only banner and watermark; every AI result requires a human Approve/Reject, and each decision is recorded in a tamper-evident audit log. See `LIMITATIONS.md` for known failure modes and `THREAT_MODEL.md` for security.
