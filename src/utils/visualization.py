"""
Visualization utilities for training history and evaluation results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from pathlib import Path


def save_training_history_plots(history, model_name, task, output_dir=None):
    """
    Save training history visualizations (loss and accuracy plots).
    
    Args:
        history: Dictionary with 'train_acc', 'train_loss', 'val_acc', 'val_loss' lists
        model_name: Name of the model (for file naming)
        task: Task name (for file naming)
        output_dir: Optional output directory (defaults to model_name)
    """
    if len(history['train_loss']) <= 1:
        return False
    
    if output_dir is None:
        output_dir = model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = plt.get_cmap('tab10').colors

    epochs = np.arange(1, len(history['train_loss']) + 1)
    xnew = np.linspace(epochs[0], epochs[-1], 300)
    
    # Loss plot
    train_loss_smooth = interpolate.interp1d(epochs, history['train_loss'], kind='linear')(xnew)
    val_loss_smooth = interpolate.interp1d(epochs, history['val_loss'], kind='linear')(xnew)

    fig, ax = plt.subplots()
    ax.plot(xnew, train_loss_smooth, color=colors[0], linewidth=3, label='Training Loss')
    ax.plot(xnew, val_loss_smooth, color=colors[1], linewidth=3, label='Validation Loss')
    ax.set_title('Training and Validation Loss', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10, labelpad=10)
    ax.set_ylabel('Loss', fontsize=10, labelpad=10)
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(axis='y')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, prop={'size': 10})
    
    file_name = output_dir / f"{model_name}_loss_visualization_{task}.png"
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Accuracy plot
    train_acc_smooth = interpolate.interp1d(epochs, history['train_acc'], kind='linear')(xnew)
    val_acc_smooth = interpolate.interp1d(epochs, history['val_acc'], kind='linear')(xnew)

    fig, ax = plt.subplots()
    ax.plot(xnew, train_acc_smooth, color=colors[2], linewidth=3, label='Training Accuracy')
    ax.plot(xnew, val_acc_smooth, color=colors[4], linewidth=3, label='Validation Accuracy')
    ax.set_title('Training and Validation Accuracy', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10, labelpad=10)
    ax.set_ylabel('Accuracy', fontsize=10, labelpad=10)
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(axis='y')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, prop={'size': 10})
    
    file_name = output_dir / f"{model_name}_accuracy_visualization_{task}.png"
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return True


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_probs, num_classes, save_path):
    """
    Plot and save ROC curve(s).
    """
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        # Multi-class classification (One-vs-Rest)
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        # Plot ROC curve for each class
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(y_true, y_probs, num_classes, save_path):
    """
    Plot and save Precision-Recall curve(s).
    """
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        pr_auc = average_precision_score(y_true, y_probs[:, 1])
        
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
    else:
        # Multi-class classification
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        # Plot PR curve for each class
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            pr_auc = average_precision_score(y_true_bin[:, i], y_probs[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {pr_auc:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")


def plot_class_distribution(y_true, y_pred, class_names, save_path):
    """
    Plot distribution comparison between true and predicted labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True labels distribution
    unique, counts = np.unique(y_true, return_counts=True)
    axes[0].bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticks(unique)
    axes[0].set_xticklabels(class_names)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, count in zip(unique, counts):
        axes[0].text(i, count, str(count), ha='center', va='bottom')
    
    # Predicted labels distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    axes[1].bar(unique, counts, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xticks(unique)
    axes[1].set_xticklabels(class_names)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, count in zip(unique, counts):
        axes[1].text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {save_path}")

