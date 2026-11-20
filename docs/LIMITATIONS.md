# Limitations and bias

**FOR EDUCATION ONLY - NOT FOR CLINICAL USE.**

## Data and labels
- **Noisy labels.** CheXpert labels are extracted from radiology reports by a rule-based labeler, not from pixel-level review. Errors in the labeler become errors the model learns.
- **Uncertain labels.** U-Ones treats "uncertain" as positive, which raises sensitivity and lowers precision for some findings. A different policy (U-Zeros, soft labels) changes behaviour.
- **Single source.** CheXpert comes from one institution; the segmentation data (MSD spleen) is also a narrow, single-center CT cohort. Expect performance to drop on other scanners, protocols, and populations.
- **Resolution.** Images are downsampled (CheXpert-small, then 224x224). Small findings (subtle nodules, thin pneumothoraces) can be lost.
- **Shortcut learning.** Chest X-ray models are known to key on devices, text markers, or view/portable-vs-fixed cues. "Support Devices" is excluded from displayed findings and Grad-CAM selection, but the shortcut can still affect other outputs.

## Bias and fairness
- Subgroup AUROC by sex and age band is reported in `reports/eval/subgroups.csv`, but the test set is small; absence of a measured gap is **not** evidence of fairness.
- Race/ethnicity, body habitus, and scanner type were not analysed.

## Model behaviour
- **Confidence is not calibrated probability.** The percentage shown is a sigmoid score. Check the reliability diagram (`reports/eval/calibration.png`) before reading it as a probability.
- **Independent sigmoids.** Findings are predicted independently, so combinations can be clinically inconsistent (e.g. "No Finding" high alongside a positive finding).
- **Grad-CAM is a coarse hint, not proof of reasoning.** The map comes from a ~7x7 feature grid upsampled to the image; it can highlight non-pathological regions and can look plausible even when the prediction is wrong.
- **One result per study.** For chest X-rays the worker analyses one image per study (preferring an AP/PA view).
- **Segmentation scope.** Spleen only, on CT. Volume (mL) is a voxel count from the predicted mask and depends on spacing/resampling; it is not a validated measurement. The displayed slice is the one with the largest predicted area.

## Pipeline
- CheXpert images are JPEGs wrapped into DICOM, so real-world DICOM variability (compression, window/level, MONOCHROME1, multi-frame) is only partly exercised. 8-bit wrapped images are passed through without re-stretching to match training.
- Compressed DICOM transfer syntaxes need extra decoders (pylibjpeg/gdcm) not installed by default.

## Human oversight
AI output is advisory. The Approve/Reject workflow exists to demonstrate human-in-the-loop design, but reviewers can still over-trust an AI suggestion (automation bias); in a real system this needs study and training.

## Findings from the evaluation run
- **Enlarged Cardiomediastinum is at chance** on the official test set (AUROC 0.50, 95% CI 0.42-0.58, 105 positives) and badly miscalibrated
  (ECE 0.43), although it scored 0.67 on the held-out validation split of the training data. Retraining with uncertain labels mapped to negative (U-Zeros)
  raised it to 0.57 (0.49-0.65), still not distinguishable from chance, so the U-Ones label policy is at most part of the explanation. The remaining
  gap (text-labeler training labels vs radiologist-consensus test labels, or genuine difficulty) is unresolved.
- **Lung Lesion, Pleural Other and Fracture cannot be evaluated** on the official test set (1, 1 and 0 positives). Pneumonia (8) and Pneumothorax (7)
  have very wide confidence intervals. Their point estimates should not be quoted.
- **Confident errors cluster on complex portable films.** In the Pleural Effusion failure gallery (`reports/eval/failures.png`), most of the confident
  false positives and false negatives are portable AP films with lines, tubes, or heavy opacification, and several show "PORTABLE"/"UPRIGHT" text markers. This is
  consistent with the shortcut-learning risk described above, but the gallery does not prove it.

## Informal Grad-CAM review (3 studies, qualitative)
Three chest X-ray overlays were inspected by eye (screenshots in `docs/screenshots/`). This is a small, non-systematic check, not an evaluation of the explanations.
- **Resolution is coarse.** The heat map comes from a 7x7 feature grid upsampled to the image, so hotspots are broad blobs and can look almost rectangular
  (one study showed a near-rectangular hotspot over a lower lung zone). It cannot localise small structures.
- **Pneumothorax example (score 80%).** The hotspot sits over a hazy upper hemithorax region that contains lines and surgical clips, not over the most hyperlucent area.
  Pneumothorax classifiers are known to be able to rely on chest drains and lines as a shortcut; this study was not checked against its ground-truth label, so no conclusion is drawn.
- **Atelectasis example (score 38%).** The hotspot sits over the upper mediastinum / paratracheal region next to central lines, which is plausible but not specific.
- **Text markers.** The films carry markers such as "PORT UPRIGHT" or "AP UPRIGHT"; in these three studies the hotspots did not sit on them.
- **Takeaway:** overlays are a plausibility aid for a human reviewer, not proof of what the model used. A systematic check (many studies, ground-truth comparison,
  occlusion tests of lines/markers) was not done.
