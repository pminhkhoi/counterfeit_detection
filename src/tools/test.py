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
src_dir = current_dir.parent  # Go up from src/tools to src
sys.path.insert(0, str(src_dir))

from dataset.dataset import CSVDataset
from models.PhoBert import ViSpam_Classifier
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution
)
from preprocessing.preprocessing import preprocessing


def test_step(model, dataloader, device='cpu'):
    """
    Run inference on test data and collect predictions with probabilities.
    
    Args:
        model: The model to evaluate
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
            
            # Handle optional category_id
            category_id = data.get('category_id')
            if category_id is not None:
                category_id = category_id.to(device)
            
            # Forward pass
            if category_id is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_id=category_id)
            else:
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
        print(f"Showing 5 most confident misclassifications:\n")
        for idx, row in misclassified.sort_values('confidence', ascending=False).head(5).iterrows():
            print(f"Text: {row[text_col][:100]}...")
            print(f"True: {row['true_label']}, Predicted: {row['predicted_label']}, Confidence: {row['confidence']:.4f}\n")


def generate_classification_report(y_true, y_pred, class_names, save_path):
    """
    Generate and save detailed classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    print("\nClassification Report:")
    print(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test script for spam classification model')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--text_col', type=str, default='segmented_comment', help='Column name for text input')
    parser.add_argument('--label_col', type=str, default='label', help='Column name for labels')
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='../results/test_results', help='Directory to save results')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Normal', 'Spam'], 
                       help='Names of classes (in order)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--preprocess", action="store_true", 
                       help="Apply preprocessing (typo mapping + segmentation) before prediction")
    parser.add_argument("--mapping_path", type=str, default="src/mapping.json",
                       help="Path to typo mapping JSON file (required if --preprocess is used)")
    parser.add_argument("--vncorenlp_dir", type=str, default="notebooks/vncorenlp",
                       help="Path to VnCoreNLP models directory (required if --preprocess is used)")
    
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
    
    # Apply preprocessing if requested
    if args.preprocess:
        print("\nApplying preprocessing (typo mapping + segmentation)...")
        print(f"Mapping file: {args.mapping_path}")
        print(f"VnCoreNLP directory: {args.vncorenlp_dir}")
        
        # Check if 'comment' column exists (required for preprocessing)
        if 'comment' not in test_df.columns:
            print("Warning: 'comment' column not found. Using existing text column for preprocessing.")
            # Create a temporary 'comment' column from the text column
            if args.text_col in test_df.columns:
                test_df['comment'] = test_df[args.text_col]
            else:
                raise ValueError(f"Neither 'comment' nor '{args.text_col}' column found for preprocessing")
        
        # Apply preprocessing
        test_df = preprocessing(
            test_df,
            mapper_path=Path(args.mapping_path),
            vncorenlp_dir=Path(args.vncorenlp_dir),
            save_csv_path=None  # Don't save intermediate result
        )
        
        # Update text column to use segmented_comment if it exists
        if 'segmented_comment' in test_df.columns:
            print("Using 'segmented_comment' column for evaluation")
            args.text_col = 'segmented_comment'
        elif 'preprocessed_review' in test_df.columns:
            print("Using 'preprocessed_review' column for evaluation")
            args.text_col = 'preprocessed_review'
        else:
            print("Warning: Preprocessing did not create expected columns. Using original text column.")
    
    # Show class distribution
    if args.label_col in test_df.columns:
        class_dist = test_df[args.label_col].value_counts().sort_index()
        print(f"\nTest set class distribution:")
        print(class_dist)
    
    # Initialize tokenizer
    print(f"\nInitializing tokenizer ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create dataset and dataloader
    # Use DataFrame directly if preprocessing was applied, otherwise use CSV path
    test_dataset = CSVDataset(
        test_df if args.preprocess else args.test_csv,
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
    
    # Initialize model
    print("\nInitializing model...")
    model = ViSpam_Classifier(model_name=args.model_name)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Model trained for {checkpoint['epoch']} epochs")
            if 'best_f1' in checkpoint:
                print(f"Best validation F1: {checkpoint['best_f1']:.4f}")
        else:
            # Old format: just state dict
            model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    model.to(device)
    
    # Run testing
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()