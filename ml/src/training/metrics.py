import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

def compute_auc(y_true, y_pred):
    """Compute AUC, returning 0.5 if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)

def compute_accuracy(y_true, y_pred, threshold=0.5):
    preds = (np.array(y_pred) > threshold).astype(int)
    return accuracy_score(y_true, preds)

def compute_eer(y_true, y_pred):
    """Compute Equal Error Rate. Returns 1.0 (worst) if data is invalid."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 1.0 
        
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    
    # Find intersection of FPR and FNR
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def compute_f1_score(y_true, y_pred, threshold=0.5):
    preds = (np.array(y_pred) > threshold).astype(int)
    return float(f1_score(y_true, preds, zero_division=0))

def plot_roc_curve(y_true, y_pred, save_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

def plot_confusion_matrix(y_true, y_pred, save_path, threshold=0.5):
    try:
        preds = (np.array(y_pred) > threshold).astype(int)
        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        plt.figure(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception:
        pass
