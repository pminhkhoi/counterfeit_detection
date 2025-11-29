"""
Testing script for Hybrid Model (PhoBERT + CNN + BiLSTM).
"""
import os
import gc
import json
import torch
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

# Add src directory to path for imports
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from dataset.dataset import CSVDataset
from models.hybrid_model import HybridModel
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution
)


def test_step(model, dataloader, device='cpu'):
    """
    Run inference on test data and collect predictions with probabilities.

    Args:
        model: The Hybrid model to evaluate
        dataloader: Test data loader
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Tuple of (trues, predicts, probabilities)
        - trues: array of true labels
        - predicts: array of predicted labels
        - probabilities: array of prediction probabilities (for ROC-AUC)
    """
    model.eval()
    model.to(device)

    trues = []
    predicts = []
    probabilities = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Testing", leave=False):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            # Forward pass through Hybrid model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            pred = torch.max(outputs, dim=1)[1]

            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)

            # Store results
            trues.extend(labels.cpu().numpy())
            predicts.extend(pred.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return np.array(trues), np.array(predicts), np.array(probabilities)


def save_predictions(test_df, y_true, y_pred, y_probs, save_path, text_col='segmented_comment'):
    """
    Save predictions with original text to CSV for error analysis.

    Args:
        test_df: Original test dataframe
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        save_path: Path to save predictions CSV
        text_col: Column name containing text
    """
    results_df = test_df.copy()
    results_df['true_label'] = y_true
    results_df['predicted_label'] = y_pred
    results_df['correct'] = (y_true == y_pred)

    # Add probability columns
    for i in range(y_probs.shape[1]):
        results_df[f'prob_class_{i}'] = y_probs[:, i]

    # Add prediction confidence (max probability)
    results_df['confidence'] = y_probs.max(axis=1)

    # Sort by confidence (ascending) to see uncertain predictions first
    results_df = results_df.sort_values('confidence', ascending=True)

    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {save_path}")

    # Show some misclassified examples
    misclassified = results_df[results_df['correct'] == False]
    if len(misclassified) > 0:
        print(f"\nFound {len(misclassified)} misclassified samples")

        # Check if text_col exists before trying to display it
        if text_col in results_df.columns:
            print(f"Showing 5 most confident misclassifications:\n")
            for idx, row in misclassified.sort_values('confidence', ascending=False).head(5).iterrows():
                # Safely handle text display
                text = row[text_col]
                if pd.notna(text):  # Check if not NaN
                    text_str = str(text)[:100]
                else:
                    text_str = "[No text available]"
                print(f"Text: {text_str}...")
                print(
                    f"True: {row['true_label']}, Predicted: {row['predicted_label']}, Confidence: {row['confidence']:.4f}\n")
        else:
            print(f"Text column '{text_col}' not found in dataframe. Skipping text display.")
            print(f"Showing 5 most confident misclassifications (without text):\n")
            for idx, row in misclassified.sort_values('confidence', ascending=False).head(5).iterrows():
                print(
                    f"True: {row['true_label']}, Predicted: {row['predicted_label']}, Confidence: {row['confidence']:.4f}\n")


def generate_classification_report(y_true, y_pred, class_names, save_path):
    """
    Generate and save detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save report
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\nClassification Report:")
    print(report)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    print(f"Classification report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Hybrid model for spam classification')

    # Data arguments
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--text_col', type=str, default='segmented_comment', help='Column name for text input')
    parser.add_argument('--label_col', type=str, default='label', help='Column name for labels')

    # Model arguments
    parser.add_argument('--phobert_model', type=str, default='vinai/phobert-base',
                        help='PhoBERT model name (default: vinai/phobert-base)')
    parser.add_argument('--cnn_out_channels', type=int, default=128,
                        help='Number of output channels for each CNN layer (default: 128)')
    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                        help='Hidden size for BiLSTM (default: 128)')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='Number of BiLSTM layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')

    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='results/test_results_hybrid',
                        help='Directory to save results')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes to predict')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Normal', 'Spam'],
                        help='Names of classes (in order)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create timestamped output directory
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir).resolve() / folder_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.exists():
            raise RuntimeError(f"Failed to create output directory: {output_dir}")
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        raise

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    # Load test data
    print(f"\nLoading test data from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(test_df)}")

    # Show class distribution
    if args.label_col in test_df.columns:
        class_dist = test_df[args.label_col].value_counts().sort_index()
        print(f"\nTest set class distribution:")
        print(class_dist)

    # Initialize tokenizer
    print(f"\nInitializing tokenizer ({args.phobert_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.phobert_model)

    # Create dataset and dataloader
    test_dataset = CSVDataset(
        test_df,
        args.text_col,
        args.label_col,
        tokenizer,
        args.max_len
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initialize Hybrid model
    print("\nInitializing Hybrid model...")
    print(f"  - PhoBERT: {args.phobert_model}")
    print(f"  - CNN Output Channels: {args.cnn_out_channels}")
    print(f"  - LSTM Hidden Size: {args.lstm_hidden_size}")
    print(f"  - LSTM Layers: {args.lstm_layers}")
    print(f"  - Dropout: {args.dropout}")

    model = HybridModel(
        phobert_model_name=args.phobert_model,
        cnn_out_channels=args.cnn_out_channels,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    )

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Model trained for {checkpoint['epoch']} epochs")
            if 'best_f1' in checkpoint:
                print(f"Best validation F1: {checkpoint['best_f1']:.4f}")
        else:
            # State dict only
            model.load_state_dict(checkpoint)

        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Run testing
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    y_true, y_pred, y_probs = test_step(model, test_loader, device=device)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_probs=y_probs, num_classes=args.num_classes)

    # Print metrics
    print_metrics(metrics, num_classes=args.num_classes)

    # Save metrics to JSON
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate classification report
    report_path = output_dir / 'classification_report.txt'
    generate_classification_report(y_true, y_pred, args.class_names, report_path)

    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, args.class_names, cm_path)

    # Plot ROC curve
    try:
        roc_path = output_dir / 'roc_curve.png'
        plot_roc_curve(y_true, y_probs, args.num_classes, roc_path)
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

    # Plot Precision-Recall curve
    try:
        pr_path = output_dir / 'precision_recall_curve.png'
        plot_precision_recall_curve(y_true, y_probs, args.num_classes, pr_path)
    except Exception as e:
        print(f"Could not plot PR curve: {e}")

    # Plot class distribution
    dist_path = output_dir / 'class_distribution.png'
    plot_class_distribution(y_true, y_pred, args.class_names, dist_path)

    # Save predictions for error analysis
    predictions_path = output_dir / 'predictions.csv'
    save_predictions(test_df, y_true, y_pred, y_probs, predictions_path, args.text_col)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()