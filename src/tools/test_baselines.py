import os
import gc
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import sys

# Add src directory to path for imports
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from models.baselines import Baseline
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution
)
from preprocess.preprocessing import Preprocessor


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

        # Check if text_col exists before trying to display it
        if text_col in results_df.columns:
            print(f"Showing 5 most confident misclassifications:\n")
            for idx, row in misclassified.sort_values('confidence', ascending=False).head(5).iterrows():
                # Safely handle text display
                text = row[text_col]
                if pd.notna(text):
                    text_str = str(text)[:100]
                else:
                    text_str = "[No text available]"
                print(f"Text: {text_str}...")
                print(
                    f"True: {row['true_label']}, Predicted: {row['predicted_label']}, Confidence: {row['confidence']:.4f}\n")
        else:
            print(f"Text column '{text_col}' not found in dataframe. Skipping text display.")


def main():
    parser = argparse.ArgumentParser(description='Test baseline models for spam classification')

    # Data arguments
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model (.pkl file)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model config JSON (optional, will try to infer from model_path)')
    parser.add_argument('--text_col', type=str, default='segmented_comment', help='Column name for text input')
    parser.add_argument('--label_col', type=str, default='label', help='Column name for labels')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/baseline_test_results',
                        help='Directory to save results')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Normal', 'Spam'],
                        help='Names of classes (in order)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Preprocessing arguments
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply preprocessing (typo mapping + segmentation)")
    parser.add_argument("--mapping_path", type=str, default="src/mapping.json",
                        help="Path to typo mapping JSON file")
    parser.add_argument("--vncorenlp_dir", type=str, default="../notebooks/vncorenlp",
                        help="Path to VnCoreNLP models directory")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Try to load config if not provided
    if args.config_path is None:
        model_dir = Path(args.model_path).parent
        potential_config = model_dir / 'config.json'
        if potential_config.exists():
            args.config_path = str(potential_config)
            print(f"Found config file: {args.config_path}")

    # Load config if available
    if args.config_path and Path(args.config_path).exists():
        print(f"Loading config from {args.config_path}...")
        with open(args.config_path, 'r') as f:
            config = json.load(f)

        # Update args with config values if not explicitly provided
        if args.num_classes == 2 and 'num_classes' in config:
            args.num_classes = config['num_classes']
        if args.class_names == ['Normal', 'Spam'] and 'class_names' in config:
            args.class_names = config['class_names']
        if args.text_col == 'segmented_comment' and 'text_col' in config:
            args.text_col = config['text_col']

        print(f"Loaded configuration: {config}")

    # Create timestamped output directory
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir).resolve() / folder_name

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        raise

    # Load test data
    print(f"\nLoading test data from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(test_df)}")

    # Apply preprocessing if requested
    if args.preprocess:
        processor = Preprocessor(
            mapper_path=args.mapping_path,
            vncorenlp_dir=args.vncorenlp_dir,
        )

        print("\nApplying preprocessing (typo mapping + segmentation)...")
        print(f"Mapping file: {args.mapping_path}")
        print(f"VnCoreNLP directory: {args.vncorenlp_dir}")

        if args.text_col not in test_df.columns:
            raise ValueError(f"Column {args.text_col} not found for preprocessing")

        test_df = processor.preprocessing_pipeline(test_df, input_col=args.text_col)

        if 'segmented_comment' in test_df.columns:
            print("Using 'segmented_comment' column for evaluation")
            args.text_col = 'segmented_comment'
        else:
            print("Warning: Preprocessing did not create expected columns. Using original text column.")

    # Show class distribution
    if args.label_col in test_df.columns:
        class_dist = test_df[args.label_col].value_counts().sort_index()
        print(f"\nTest set class distribution:")
        print(class_dist)

    # Prepare data
    X_test = test_df[args.text_col].values
    y_test = test_df[args.label_col].values

    # Initialize and load model
    print(f"\nLoading model from {args.model_path}...")
    model = Baseline()  # Initialize with default params
    model.load(args.model_path)

    print(f"Model type: {model.model_type}")
    print(f"Vectorizer: {model.vectorizer_type}")

    # Run testing
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_probs=y_probs, num_classes=args.num_classes)

    # Print metrics
    print_metrics(metrics, num_classes=args.num_classes)

    # Save metrics to JSON
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=args.class_names, digits=4)

    print("\nClassification Report:")
    print(report)

    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, args.class_names, cm_path)

    # Plot ROC curve
    try:
        roc_path = output_dir / 'roc_curve.png'
        plot_roc_curve(y_test, y_probs, args.num_classes, roc_path)
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

    # Plot Precision-Recall curve
    try:
        pr_path = output_dir / 'precision_recall_curve.png'
        plot_precision_recall_curve(y_test, y_probs, args.num_classes, pr_path)
    except Exception as e:
        print(f"Could not plot PR curve: {e}")

    # Plot class distribution
    dist_path = output_dir / 'class_distribution.png'
    plot_class_distribution(y_test, y_pred, args.class_names, dist_path)

    # Save predictions for error analysis
    predictions_path = output_dir / 'predictions.csv'
    save_predictions(test_df, y_test, y_pred, y_probs, predictions_path, args.text_col)

    # Get feature importance if applicable
    if model.model_type in ['logistic', 'linear_svm']:
        print("\nExtracting feature importance...")
        importance = model.get_feature_importance(top_n=20)

        if importance:
            print("\nTop 10 positive features:")
            for feat, coef in importance['top_positive'][:10]:
                print(f"  {feat}: {coef:.4f}")

            print("\nTop 10 negative features:")
            for feat, coef in importance['top_negative'][:10]:
                print(f"  {feat}: {coef:.4f}")

            # Save feature importance
            importance_path = output_dir / 'feature_importance.json'
            importance_serializable = {
                'top_positive': [(str(f), float(c)) for f, c in importance['top_positive']],
                'top_negative': [(str(f), float(c)) for f, c in importance['top_negative']]
            }
            with open(importance_path, 'w', encoding='utf-8') as f:
                json.dump(importance_serializable, f, indent=2, ensure_ascii=False)
            print(f"Feature importance saved to {importance_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)
    print(f"\nTest F1-Score: {metrics['f1']:.4f}")
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()