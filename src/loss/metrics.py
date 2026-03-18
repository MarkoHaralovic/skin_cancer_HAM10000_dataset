import logging
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

# Indices of malignant classes in the 7-class label scheme:
#   0=akiec, 1=bcc, 4=mel  (pre-malignant / malignant)
# All others (2=bkl, 3=df, 5=nv, 6=vasc) are benign.
MALIGNANT_INDICES = {0, 1, 4}


def get_metrics(y_true, y_pred):
    """
    Compute per-class and aggregate metrics.

    Works for both binary mode (label 1 = malignant, 0 = benign)
    and 7-class mode (HAM10000 diagnosis codes 0-6).

    Returns a dict with:
        accuracy          – overall % correct
        balanced_acc      – mean per-class recall (robust to class imbalance)
        macro_f1          – unweighted average F1 across all classes
        weighted_f1       – F1 weighted by class support (useful for comparison with papers)
        per_class_f1      – {class_idx: f1}
        per_class_recall  – {class_idx: recall}  (clinically critical, esp. for melanoma)
        malignant_acc     – sensitivity: TP / (TP + FN) over malignant classes
        benign_acc        – specificity: TN / (TN + FP) over benign classes
        confusion_matrix  – raw confusion matrix as a numpy array
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Overall accuracy ──────────────────────────────────────────────────────
    accuracy = (y_pred == y_true).mean() * 100
    logging.info(f"Global accuracy: {accuracy:.2f}%")

    # ── Full classification report ────────────────────────────────────────────
    logging.info(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    # ── F1 scores ─────────────────────────────────────────────────────────────
    macro_f1    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_f1 = {int(i): float(f)
                    for i, f in enumerate(f1_score(y_true, y_pred, average=None, zero_division=0))}
    logging.info(f"Macro F1:    {macro_f1:.4f}")
    logging.info(f"Weighted F1: {weighted_f1:.4f}")
    logging.info(f"Per-class F1: {per_class_f1}")

    # ── Per-class recall (sensitivity per diagnosis) ──────────────────────────
    per_class_recall = {int(i): float(r)
                        for i, r in enumerate(recall_score(y_true, y_pred, average=None, zero_division=0))}
    logging.info(f"Per-class recall: {per_class_recall}")

    # ── Balanced accuracy ─────────────────────────────────────────────────────
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    logging.info(f"Balanced accuracy: {balanced_acc:.4f}")

    # ── Malignant vs benign sensitivity / specificity ─────────────────────────
    # Collapse to binary regardless of whether we're in 7-class or binary mode.
    # MALIGNANT_INDICES covers both: in binary mode only label 1 exists there.
    y_true_bin = np.isin(y_true, list(MALIGNANT_INDICES)).astype(int)
    y_pred_bin = np.isin(y_pred, list(MALIGNANT_INDICES)).astype(int)

    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

    malignant_acc = tp / (tp + fn + 1e-10)   # sensitivity: how often malignant cases are caught
    benign_acc    = tn / (tn + fp + 1e-10)   # specificity: how often benign cases are left alone

    logging.info(f"Malignant sensitivity: {malignant_acc:.4f}  [TP={tp}, FN={fn}]")
    logging.info(f"Benign specificity:    {benign_acc:.4f}  [TN={tn}, FP={fp}]")

    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"Confusion matrix:\n{cm}")

    return {
        'accuracy':         accuracy,
        'balanced_acc':     balanced_acc,
        'macro_f1':         macro_f1,
        'weighted_f1':      weighted_f1,
        'per_class_f1':     per_class_f1,
        'per_class_recall': per_class_recall,
        'malignant_acc':    malignant_acc,
        'benign_acc':       benign_acc,
        'confusion_matrix': cm,
    }
