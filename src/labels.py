"""Label definitions shared by training, evaluation and the inference service."""
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
# Not shown to the radiologist as "findings" and excluded from Grad-CAM selection.
NON_FINDINGS = {"No Finding", "Support Devices"}
