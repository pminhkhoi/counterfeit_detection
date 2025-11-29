import os
import gc
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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


def train_and_evaluate(model, X_train, y_train, X_val, y_val, class_names, num_classes):
    """
    Train model and evaluate on validation set.

    Args:
        model: Baseline model instance
        X_train: Training texts
        y_train: Training labels
        X_val: Validation texts
        y_val: Validation labels
        class_names: List of class names
        num_classes: Number of classes

    Returns:
        dict: Training and validation metrics
    """
    # Train model
    model.fit(X_train, y_train)

    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_pred = model.predict(X_train)
    train_probs = model.predict_proba(X_train)
    train_metrics = calculate_metrics(y_train, train_pred, y_probs=train_probs, num_classes=num_classes)

    print("\nTraining Metrics:")
    print_metrics(train_metrics, num_classes=num_classes)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    val_metrics = calculate_metrics(y_val, val_pred, y_probs=val_probs, num_classes=num_classes)

    print("\nValidation Metrics:")
    print_metrics(val_metrics, num_classes=num_classes)

    return {
        'train': train_metrics,
        'validation': val_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train baseline models for spam classification')

    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None, help='Path to validation CSV file (optional)')
    parser.add_argument('--text_col', type=str, default='segmented_comment', help='Column name for text input')
    parser.add_argument('--label_col', type=str, default='label', help='Column name for labels')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split if no val_csv provided')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='logistic',
                        choices=['logistic', 'svm', 'linear_svm'],
                        help='Type of baseline model')
    parser.add_argument('--vectorizer', type=str, default='tfidf',
                        choices=['tfidf', 'count'],
                        help='Vectorizer type')
    parser.add_argument('--max_features', type=int, default=5000,
                        help='Maximum number of features')
    parser.add_argument('--ngram_range', type=int, nargs=2, default=[1, 2],
                        help='N-gram range (e.g., 1 2 for unigram+bigram)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/baseline_results',
                        help='Directory to save results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
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

    # Create timestamped output directory
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"{args.model_type}_{args.vectorizer}"
    output_dir = Path(args.output_dir).resolve() / model_name / folder_name

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        raise

    # Load training data
    print(f"\nLoading training data from {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Training samples: {len(train_df)}")

    # Apply preprocessing if requested
    if args.preprocess:
        processor = Preprocessor(
            mapper_path=args.mapping_path,
            vncorenlp_dir=args.vncorenlp_dir,
        )

        print("\nApplying preprocessing (typo mapping + segmentation)...")
        print(f"Mapping file: {args.mapping_path}")
        print(f"VnCoreNLP directory: {args.vncorenlp_dir}")

        if args.text_col not in train_df.columns:
            raise ValueError(f"Column {args.text_col} not found for preprocessing")

        train_df = processor.preprocessing_pipeline(train_df, input_col=args.text_col)

        if 'segmented_comment' in train_df.columns:
            print("Using 'segmented_comment' column for training")
            args.text_col = 'segmented_comment'

    # Load or split validation data
    if args.val_csv:
        print(f"\nLoading validation data from {args.val_csv}...")
        val_df = pd.read_csv(args.val_csv)

        if args.preprocess:
            print("Applying preprocessing to validation data...")
            val_df = processor.preprocessing_pipeline(val_df, input_col=args.text_col)

        print(f"Validation samples: {len(val_df)}")
    else:
        print(f"\nSplitting data: {1 - args.val_split:.0%} train, {args.val_split:.0%} validation")
        train_df, val_df = train_test_split(
            train_df,
            test_size=args.val_split,
            random_state=args.seed,
            stratify=train_df[args.label_col] if args.label_col in train_df.columns else None
        )

    # Prepare data
    X_train = train_df[args.text_col].values
    y_train = train_df[args.label_col].values
    X_val = val_df[args.text_col].values
    y_val = val_df[args.label_col].values

    # Show class distribution
    print(f"\nTraining set class distribution:")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    print(train_dist)

    print(f"\nValidation set class distribution:")
    val_dist = pd.Series(y_val).value_counts().sort_index()
    print(val_dist)

    # Initialize model
    print("\n" + "=" * 60)
    print(f"TRAINING {args.model_type.upper()} MODEL")
    print("=" * 60)

    model = Baseline(
        model_type=args.model_type,
        vectorizer=args.vectorizer,
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.seed
    )

    print(f"\nModel configuration:")
    print(model)

    # Train and evaluate
    results = train_and_evaluate(
        model, X_train, y_train, X_val, y_val,
        args.class_names, args.num_classes
    )

    # Save results
    print("\nSaving results...")

    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    # Save model configuration
    config = {
        'model_type': args.model_type,
        'vectorizer': args.vectorizer,
        'max_features': args.max_features,
        'ngram_range': args.ngram_range,
        'C': args.C,
        'max_iter': args.max_iter,
        'num_classes': args.num_classes,
        'class_names': args.class_names,
        'text_col': args.text_col,
        'label_col': args.label_col
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Generate classification report
    val_pred = model.predict(X_val)
    report = classification_report(y_val, val_pred, target_names=args.class_names, digits=4)

    print("\nValidation Classification Report:")
    print(report)

    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("VALIDATION CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Plot confusion matrix
    cm = np.array(results['validation']['confusion_matrix'])
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, args.class_names, cm_path)

    # Plot ROC curve
    try:
        val_probs = model.predict_proba(X_val)
        roc_path = output_dir / 'roc_curve.png'
        plot_roc_curve(y_val, val_probs, args.num_classes, roc_path)
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

    # Plot Precision-Recall curve
    try:
        pr_path = output_dir / 'precision_recall_curve.png'
        plot_precision_recall_curve(y_val, val_probs, args.num_classes, pr_path)
    except Exception as e:
        print(f"Could not plot PR curve: {e}")

    # Plot class distribution
    dist_path = output_dir / 'class_distribution.png'
    plot_class_distribution(y_val, val_pred, args.class_names, dist_path)

    # Get feature importance
    if args.model_type in ['logistic', 'linear_svm']:
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

    # Save model
    if args.save_model:
        model_path = output_dir / f'{args.model_type}_{args.vectorizer}_model.pkl'
        model.save(model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"\nValidation F1-Score: {results['validation']['f1']:.4f}")
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()