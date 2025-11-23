"""
Metrics calculation utilities for model evaluation.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)


def calculate_metrics(y_true, y_pred, y_probs=None, num_classes=2):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (shape: [n_samples, n_classes]), optional
        num_classes: Number of classes
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # For binary classification
    if num_classes == 2:
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # ROC-AUC and PR-AUC (if probabilities provided)
        if y_probs is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
            except Exception as e:
                metrics['roc_auc'] = None
            
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_probs[:, 1])
            except Exception as e:
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    
    # For multi-class classification
    else:
        # Macro averaging (unweighted mean)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted averaging (by support)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_probs is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovo')
            except Exception as e:
                metrics['roc_auc_ovr'] = None
                metrics['roc_auc_ovo'] = None
        else:
            metrics['roc_auc_ovr'] = None
            metrics['roc_auc_ovo'] = None
    
    # Per-class metrics
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def print_metrics(metrics, num_classes=2):
    """
    Pretty print evaluation metrics.
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    
    if num_classes == 2:
        print(f"\nBinary Classification Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        if metrics.get('pr_auc') is not None:
            print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    else:
        print(f"\nMulti-class Classification Metrics (Macro):")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
        
        print(f"\nMulti-class Classification Metrics (Weighted):")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
        
        if metrics.get('roc_auc_ovr') is not None:
            print(f"\nROC-AUC (One-vs-Rest): {metrics['roc_auc_ovr']:.4f}")
        if metrics.get('roc_auc_ovo') is not None:
            print(f"ROC-AUC (One-vs-One):  {metrics['roc_auc_ovo']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i in range(len(metrics['per_class_precision'])):
        print(f"  Class {i}:")
        print(f"    Precision: {metrics['per_class_precision'][i]:.4f}")
        print(f"    Recall:    {metrics['per_class_recall'][i]:.4f}")
        print(f"    F1 Score:  {metrics['per_class_f1'][i]:.4f}")
    
    print("\n" + "="*60)

