
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
)

def compute_metrics(y_true, y_pred, y_proba=None, label_names=None):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    if y_proba is not None:
        try:
            metrics["macro_roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["macro_roc_auc_ovr"] = None

    return metrics
