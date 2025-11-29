# Counterfeit Reviews Classification

A deep learning project for classifying counterfeit/fake reviews using PhoBERT (Vietnamese BERT) model. This project includes data preprocessing, augmentation, training, and evaluation pipelines.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setting Up](#setting-up)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Testing/Evaluation](#testingevaluation)
- [Project Structure Details](#project-structure-details)

## Introduction

This project implements a counterfeit reviews classification system using:

- **PhoBERT** (vinai/phobert-base) as the base model
- **VnCoreNLP** for Vietnamese text segmentation
- **Back-translation** and **random oversampling** for data augmentation
- **Typo mapping** for text normalization

The model classifies Vietnamese reviews as either legitimate or counterfeit (spam).

## Project Structure

```
fake-review/
│
├── data/
│
├── notebooks/
│   ├── EDA.ipynb           # Exploratory data analysis
│   └── vncorenlp/          # VnCoreNLP models directory
│       ├── VnCoreNLP-1.2.jar
│       └── models/
│
├── src/
│   ├── dataset/
│   │   ├── dataset.py      # Custom Dataset class
│   │   └── data/           # Processed datasets
│   │       ├── imbalance      # Dataset split without imbalance addressing method
│   │       └── back_translated # Dataset split with back-translation method
│   │
│   ├── models/
│   │   ├── baselines.py    # Baseline models (SVM, logistic regression)
│   │   ├── hybrid_model.py    # Hybrid models (CNN-LSTM + PhoBert embeddings)
│   │   └── PhoBert.py      # PhoBERT classifier model  
│   │
│   ├── preprocessing/
│   │   ├── preprocessing.py      # Text preprocessing pipeline
│   │   └── imbalance_handler.py # Data augmentation methods
│   │
│   ├── tools/
│   │   ├── train_baselines.py  # Training baseline models script
│   │   ├── test_baselines.py  # Evaluation baseline models script
│   │   ├── train_hybrid.py # Training hybrid model script
│   │   ├── test_hybrid.py # Evaluation hybrid models script
│   │   ├── train.py        # Training script
│   │   └── test.py         # Evaluation script
│   │
│   ├── utils/
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── visualization.py     # Plotting utilities
│   │   └── early_stopping.py    # Early stopping callback
│   │
│   ├── results/
│   │   ├── baseline_results # Save baseline models
│   │   ├── baseline_test_results # Evaluation results baseline models
│   │   ├── checkpoints/    # Saved PhoBert model checkpoints
│   │   └── test_results/   # Evaluation results PhoBert results
│   │
│   └── mapping.json        # Typo mapping dictionary
│
├── requirements.txt
└── README.md
```

## Setting Up

### 1. Prerequisites

- Python 3.12
- CUDA-capable GPU (recommended) or CPU
- Java Runtime Environment (JRE) for VnCoreNLP

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

After dependency installation move to the `scr` directory:
```bash
cd src
```

### 3. Download VnCoreNLP Models

VnCoreNLP models are required for text segmentation. The models should be placed in `notebooks/vncorenlp/`.

If models are not present, the preprocessing script will attempt to download them automatically.

```bash
# The models will be auto-downloaded on first use, or you can:
python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='notebooks/vncorenlp')"
```

Alternatively, you can download manually at https://github.com/vncorenlp/VnCoreNLP. Place `VnCoreNLP-1.2.jar` and folder `models` in the same directory, here in the code is `fake-review/data/`.

### 4. Prepare Data

Install ViSpamDetection V2 dataset from https://github.com/sonlam1102/vispamdetection or https://www.kaggle.com/datasets/cinhvn/vispamdataset-v2.
Extract the dataset at `fake-review/data/` or any directory (but will need to change path to the according directory).

Your CSV files should contain at least:

- A text column (e.g., `comment` or `segmented_comment`)
- A label column (e.g., `label`) with binary values (0/1 or Normal/Spam)

Example CSV structure:

```csv
comment, segmented_comment, label
"Đây là một sản phẩm tuyệt vời", "Đây là một sản_phẩm tuyệt_vời", 0
"Spam review fake", "Spam review fake", 1
```

## Data Preprocessing

The preprocessing pipeline includes:

1. **Typo mapping**: Normalizes common typos and teen-code

The mapping is stored in `src/mapping/json`. Manually add more mapping to further improve the performance of the model.

2. **Word segmentation**: Uses VnCoreNLP to segment Vietnamese text

### Standalone Preprocessing

You can preprocess data separately (without imbalance addressing method) using:

```bash
python -m preprocess.preprocessing \
    --csv dataset/data/raw.csv \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --data_col comment \
    --label_col label \
    --stratify_col label \
    --out_dir dataset/data/imbalance \
    --device
```

You can apply back-translation to address imbalance dataset:

```bash
python -m preprocess.preprocessing \
    --csv dataset/data/raw.csv \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --data_col comment \
    --label_col label \
    --stratify_col label \
    --out_dir dataset/data/back_translated \
    --balance_method back_translation \
    --augmentation_factor 2.0 \
    --device
```

Options:

- `--csv`: Input CSV file (must contain 'comment' column)
- `--train_ratio`: Train set ratio
- `--dev_ratio`: Dev/Val set ratio
- `--test_ratio`: Test set ratio
- `--data_col`: The initial text column (column contains raw reviews)
- `--label_col`: The label column
- `--stratify_col`:  Column for stratified splitting (default: 'label')
- `--mapper_path`: Path to typo mapping JSON file
- `--vncorenlp_dir`: VnCoreNLP models directory
- `--out_dir`: Output directory for processed files
- `--random_state`: Random seed
- `--model`: model name (default: vinai/phobert-base)
- `--balance_method`: Data balancing method: `none`, `back_translation`, or `random`
- `--augmentation_factor`: Augmentation factor for back-translation (default: 2.0)
- `--device`: Choosing device for augmentation (cuda/cpu)

# Training

This project supports two approaches: fine-tuning PhoBERT (transformer-based) and training baseline models (TF-IDF + SVM/Logistic Regression).

## PhoBERT Fine-tuning

### Basic Training (Preprocessed Data)

If your data is already preprocessed:

```bash
python tools/train.py \
  --train_csv dataset/data/train.csv \
  --dev_csv dataset/data/dev.csv \
  --text_col segmented_comment \
  --label_col label \
  --out_dir results/checkpoints \
  --epochs 10 \
  --batch_size 32
```

### Training with Preprocessing

Apply preprocessing (typo mapping + word segmentation) during training:

```bash
python tools/train.py \
  --train_csv dataset/data/train.csv \
  --dev_csv dataset/data/dev.csv \
  --text_col comment \
  --label_col label \
  --out_dir results/checkpoints \
  --preprocess \
  --vncorenlp_dir ../notebooks/vncorenlp \
  --mapper_path mapping.json
```

### Training with Data Augmentation

Apply back-translation augmentation to the training set only:

```bash
python tools/train.py \
  --train_csv dataset/data/train.csv \
  --dev_csv dataset/data/dev.csv \
  --text_col comment \
  --label_col label \
  --out_dir results/checkpoints \
  --preprocess \
  --balance_method back_translation \
  --augmentation_factor 2.0
```

### Full PhoBERT Training Example

```bash
python tools/train.py \
    --train_csv dataset/data/train.csv \
    --dev_csv dataset/data/dev.csv \
    --text_col segmented_comment \
    --label_col label \
    --out_dir results/checkpoints \
    --preprocess \
    --balance_method back_translation \
    --augmentation_factor 2.0 \
    --mapper_path mapping.json \
    --vncorenlp_dir ../notebooks/vncorenlp \
    --freeze_bert \
    --dropout 0.3 \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --early_stopping \
    --patience 7 \
    --device cuda
```

### PhoBERT Training Arguments

**Data Arguments:**

- `--train_csv`: Path to training CSV file (required)
- `--dev_csv`: Path to validation/dev CSV file (required)
- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)
- `--out_dir`: Output directory for model checkpoints (default: `results/checkpoints`)

**Preprocessing Arguments:**

- `--preprocess`: Apply preprocessing (typo mapping + segmentation)
- `--mapper_path`: Path to typo mapping JSON (default: `mapping.json`)
- `--vncorenlp_dir`: Path to VnCoreNLP models (default: `../notebooks/vncorenlp`)

**Data Augmentation Arguments (training set only):**

- `--balance_method`: Balancing method: `none` or `back_translation` (default: `none`)
- `--augmentation_factor`: Augmentation factor for back-translation (default: 2.0)

**Model Arguments:**

- `--model_name`: Pretrained model name (default: `vinai/phobert-base`)
- `--freeze_bert`: Freeze BERT parameters (flag)
- `--dropout`: Dropout rate (default: 0.3)

**Training Arguments:**

- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--max_len`: Maximum sequence length (default: 256)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--warmup_steps`: Number of warmup steps for LR scheduler (default: 0)

**Early Stopping:**

- `--early_stopping`: Enable early stopping (flag)
- `--patience`: Patience for early stopping (default: 7)
- `--early_stopping_delta`: Minimum change to qualify as improvement (default: 0)

**Other:**

- `--seed`: Random seed (default: 42)
- `--device`: Device to use (`cuda`/`cpu`, auto-detected if not specified)

### PhoBERT Training Outputs

After training, the following files are saved in `{out_dir}/{timestamp}/` directory:

- `{model_name}_best_model_(counterfeit_reviews_classification).pth`: Best model checkpoint
- `{model_name}_last_model_(counterfeit_reviews_classification).pth`: Final epoch model
- `{model_name}_checkpoint_(counterfeit_reviews_classification).pth`: Early stopping checkpoint (if enabled)
- `{model_name}_history_(counterfeit_reviews_classification).json`: Training history
- `{model_name}_loss_visualization_(counterfeit_reviews_classification).png`: Loss plots
- `{model_name}_accuracy_visualization_(counterfeit_reviews_classification).png`: Accuracy plots

---
## Hybrid Model (PhoBERT + CNN + BiLSTM)

The Hybrid model combines PhoBERT embeddings with CNN and BiLSTM layers for enhanced feature extraction.

### Basic Training

Train a hybrid model with default configuration:

```bash
python tools/train_hybrid.py \
  --train_csv dataset/data/train.csv \
  --dev_csv dataset/data/dev.csv \
  --text_col segmented_comment \
  --label_col label \
  --out_dir results/checkpoints \
  --epochs 10 \
  --batch_size 32
```

### Training with Custom Architecture

Customize the CNN and LSTM layers:

```bash
python tools/train_hybrid.py \
  --train_csv dataset/data/train.csv \
  --dev_csv dataset/data/dev.csv \
  --text_col segmented_comment \
  --label_col label \
  --out_dir results/checkpoints \
  --cnn_out_channels 256 \
  --lstm_hidden_size 256 \
  --lstm_layers 2 \
  --dropout 0.5 \
  --epochs 10
```

### Training with Early Stopping

Enable early stopping to prevent overfitting:

```bash
python tools/train_hybrid.py \
  --train_csv dataset/data/train.csv \
  --dev_csv dataset/data/dev.csv \
  --text_col segmented_comment \
  --label_col label \
  --out_dir results/checkpoints \
  --epochs 20 \
  --early_stopping \
  --patience 5
```

### Full Hybrid Model Training Example

```bash
python tools/train_hybrid.py \
    --train_csv dataset/data/train.csv \
    --dev_csv dataset/data/dev.csv \
    --text_col segmented_comment \
    --label_col label \
    --out_dir results/checkpoints \
    --phobert_model vinai/phobert-base \
    --cnn_out_channels 128 \
    --lstm_hidden_size 128 \
    --lstm_layers 1 \
    --dropout 0.3 \
    --epochs 10 \
    --batch_size 32 \
    --max_len 256 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --early_stopping \
    --patience 7 \
    --early_stopping_delta 0.001 \
    --seed 42
```

### Hybrid Model Training Arguments

**Data Arguments:**

- `--train_csv`: Path to training CSV file (required)
- `--dev_csv`: Path to validation/dev CSV file (required)
- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)
- `--out_dir`: Output directory for model checkpoints (default: `results/checkpoints`)

**PhoBERT Arguments:**

- `--phobert_model`: PhoBERT model name (default: `vinai/phobert-base`)

**Hybrid Model Architecture Arguments:**

- `--cnn_out_channels`: Number of output channels for each CNN layer (default: 128)
- `--lstm_hidden_size`: Hidden size for BiLSTM (default: 128)
- `--lstm_layers`: Number of BiLSTM layers (default: 1)
- `--dropout`: Dropout rate (default: 0.3)

**Training Arguments:**

- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--max_len`: Maximum sequence length (default: 256)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--warmup_steps`: Number of warmup steps for LR scheduler (default: 0)

**Early Stopping:**

- `--early_stopping`: Enable early stopping (flag)
- `--patience`: Patience for early stopping (default: 7)
- `--early_stopping_delta`: Minimum change to qualify as improvement (default: 0)

**Other:**

- `--seed`: Random seed (default: 42)
- `--device`: Device to use (`cuda`/`cpu`, auto-detected if not specified)

### Hybrid Model Training Outputs

After training, the following files are saved in `{out_dir}/{timestamp}/` directory:

- `hybrid_model_best_hybrid_counterfeit_reviews_classification.pth`: Best model checkpoint based on validation F1
- `hybrid_model_last_hybrid_counterfeit_reviews_classification.pth`: Final epoch model
- `hybrid_model_checkpoint_(hybrid_counterfeit_reviews_classification).pth`: Early stopping checkpoint (if enabled)
- `hybrid_model_history_hybrid_counterfeit_reviews_classification.json`: Training history with metrics
- `HybridModel_loss_visualization_hybrid_counterfeit_reviews_classification.png`: Loss plots
- `HybridModel_accuracy_visualization_hybrid_counterfeit_reviews_classification.png`: Accuracy plots
- `HybridModel_f1_visualization_hybrid_counterfeit_reviews_classification.png`: F1 score plots

## Baseline Models (TF-IDF + SVM/Logistic Regression)

### Basic Training

Train a baseline model with default configuration:

```bash
python tools/train_baselines.py \
  --train_csv dataset/data/train.csv \
  --model_type logistic \
  --vectorizer tfidf \
  --output_dir results/baseline_results
```

### Training with Validation Split

Automatically split training data for validation:

```bash
python tools/train_baselines.py \
  --train_csv dataset/data/train.csv \
  --val_split 0.2 \
  --model_type svm \
  --vectorizer tfidf \
  --max_features 5000 \
  --ngram_range 1 2 \
  --output_dir results/baseline_results \
  --save_model
```

### Training with External Validation Set

Use a separate validation CSV file:

```bash
python tools/train_baselines.py \
  --train_csv dataset/data/train.csv \
  --val_csv dataset/data/dev.csv \
  --model_type linear_svm \
  --vectorizer tfidf \
  --output_dir results/baseline_results \
  --save_model
```

### Training with Preprocessing

Apply preprocessing before training:

```bash
python tools/train_baselines.py \
  --train_csv dataset/data/train.csv \
  --val_csv dataset/data/dev.csv \
  --model_type logistic \
  --vectorizer tfidf \
  --preprocess \
  --mapping_path mapping.json \
  --vncorenlp_dir ../notebooks/vncorenlp \
  --output_dir results/baseline_results \
  --save_model
```

### Full Baseline Training Example

```bash
python tools/train_baselines.py \
    --train_csv dataset/data/train.csv \
    --val_csv dataset/data/dev.csv \
    --model_type logistic \
    --vectorizer tfidf \
    --max_features 10000 \
    --ngram_range 1 2 \
    --C 1.0 \
    --max_iter 1000 \
    --output_dir results/baseline_results \
    --preprocess \
    --mapping_path mapping.json \
    --vncorenlp_dir ../notebooks/vncorenlp \
    --class_names Normal Spam \
    --save_model \
    --seed 42
```

### Baseline Training Arguments

**Data Arguments:**

- `--train_csv`: Path to training CSV file (required)
- `--val_csv`: Path to validation CSV file (optional)
- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)
- `--val_split`: Validation split ratio if `--val_csv` not provided (default: 0.2)

**Model Arguments:**

- `--model_type`: Type of baseline model: `logistic`, `svm`, or `linear_svm` (default: `logistic`)
- `--vectorizer`: Vectorizer type: `tfidf` or `count` (default: `tfidf`)
- `--max_features`: Maximum number of features (default: 5000)
- `--ngram_range`: N-gram range as two integers, e.g., `1 2` for unigram+bigram (default: `1 2`)
- `--C`: Regularization parameter (default: 1.0)
- `--max_iter`: Maximum iterations (default: 1000)

**Output Arguments:**

- `--output_dir`: Directory to save results (default: `results/baseline_results`)
- `--save_model`: Save trained model as pickle file (flag)
- `--num_classes`: Number of classes (default: 2)
- `--class_names`: Class names in order (default: `Normal Spam`)

**Preprocessing Arguments:**

- `--preprocess`: Apply preprocessing (typo mapping + segmentation)
- `--mapping_path`: Path to typo mapping JSON (default: `src/mapping.json`)
- `--vncorenlp_dir`: Path to VnCoreNLP models (default: `../notebooks/vncorenlp`)

**Other:**

- `--seed`: Random seed (default: 42)

### Baseline Training Outputs

After training, the following files are saved in `{output_dir}/{model_type}_{vectorizer}/{timestamp}/` directory:

- `{model_type}_{vectorizer}_model.pkl`: Saved model (if `--save_model` enabled)
- `metrics.json`: Training and validation metrics
- `config.json`: Model configuration
- `classification_report.txt`: Detailed classification report
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `precision_recall_curve.png`: Precision-Recall curve plot
- `class_distribution.png`: Class distribution comparison
- `feature_importance.json`: Top positive and negative features (for logistic/linear_svm)

---

# Testing/Evaluation

## PhoBERT Model Evaluation

### Basic Evaluation

Evaluate a trained PhoBERT model on test data:

```bash
python tools/test.py \
    --test_csv dataset/data/test.csv \
    --text_col segmented_comment \
    --label_col label \
    --checkpoint results/checkpoints/2025-01-15_10-30-45/best_model_spam_classification.pth \
    --output_dir results/test_results
```

### Evaluation with Preprocessing

Apply preprocessing before evaluation:

```bash
python tools/test.py \
    --test_csv dataset/data/test.csv \
    --checkpoint results/checkpoints/2025-01-15_10-30-45/best_model_spam_classification.pth \
    --output_dir results/test_results \
    --preprocess \
    --mapping_path mapping.json \
    --vncorenlp_dir ../notebooks/vncorenlp
```

### Full PhoBERT Evaluation Example

```bash
python tools/test.py \
    --test_csv dataset/data/test.csv \
    --checkpoint results/checkpoints/2025-01-15_10-30-45/best_model_spam_classification.pth \
    --output_dir results/test_results \
    --text_col segmented_comment \
    --label_col label \
    --model_name vinai/phobert-base \
    --batch_size 32 \
    --max_len 256 \
    --num_classes 2 \
    --class_names Normal Spam \
    --preprocess \
    --mapping_path mapping.json \
    --vncorenlp_dir ../notebooks/vncorenlp \
    --seed 42
```

### PhoBERT Test Arguments

**Required:**

- `--test_csv`: Path to test CSV file
- `--checkpoint`: Path to model checkpoint (.pth file)

**Data Arguments:**

- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)

**Preprocessing Arguments:**

- `--preprocess`: Apply preprocessing before evaluation
- `--mapping_path`: Path to typo mapping JSON (default: `src/mapping.json`)
- `--vncorenlp_dir`: Path to VnCoreNLP models (default: `../notebooks/vncorenlp`)

**Model Arguments:**

- `--model_name`: Pretrained model name (default: `vinai/phobert-base`)
- `--batch_size`: Batch size for testing (default: 32)
- `--max_len`: Maximum sequence length (default: 256)
- `--num_classes`: Number of classes (default: 2)
- `--class_names`: Class names in order (default: `Normal Spam`)

**Other:**

- `--output_dir`: Directory to save results (default: `results/test_results`)
- `--seed`: Random seed (default: 42)

### PhoBERT Evaluation Outputs

Results are saved in a timestamped directory under `{output_dir}/{timestamp}/`:

- `test_metrics.json`: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
- `classification_report.txt`: Detailed classification report
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `precision_recall_curve.png`: Precision-Recall curve plot
- `class_distribution.png`: Class distribution comparison
- `predictions.csv`: Predictions with probabilities for error analysis

---
## Hybrid Model Evaluation

### Basic Evaluation

Evaluate a trained Hybrid model on test data:

```bash
python tools/test_hybrid.py \
    --test_csv dataset/data/test.csv \
    --checkpoint results/checkpoints/2025-01-15_10-30-45/hybrid_model_best_hybrid_counterfeit_reviews_classification.pth \
    --text_col segmented_comment \
    --label_col label \
    --output_dir results/test_results_hybrid
```

### Evaluation with Custom Architecture

Ensure the architecture matches the trained model:

```bash
python tools/test_hybrid.py \
    --test_csv dataset/data/test.csv \
    --checkpoint results/checkpoints/2025-01-15_10-30-45/hybrid_model_best_hybrid_counterfeit_reviews_classification.pth \
    --phobert_model vinai/phobert-base \
    --cnn_out_channels 128 \
    --lstm_hidden_size 128 \
    --lstm_layers 1 \
    --dropout 0.3 \
    --output_dir results/test_results_hybrid
```

### Full Hybrid Model Evaluation Example

```bash
python tools/test_hybrid.py \
    --test_csv dataset/data/test.csv \
    --checkpoint results/checkpoints/2025-01-15_10-30-45/hybrid_model_best_hybrid_counterfeit_reviews_classification.pth \
    --text_col segmented_comment \
    --label_col label \
    --phobert_model vinai/phobert-base \
    --cnn_out_channels 128 \
    --lstm_hidden_size 128 \
    --lstm_layers 1 \
    --dropout 0.3 \
    --batch_size 32 \
    --max_len 256 \
    --output_dir results/test_results_hybrid \
    --num_classes 2 \
    --class_names Normal Spam \
    --seed 42
```

### Hybrid Model Test Arguments

**Required:**

- `--test_csv`: Path to test CSV file
- `--checkpoint`: Path to model checkpoint (.pth file)

**Data Arguments:**

- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)

**Model Arguments:**

- `--phobert_model`: PhoBERT model name (default: `vinai/phobert-base`)
- `--cnn_out_channels`: Number of output channels for each CNN layer (default: 128)
- `--lstm_hidden_size`: Hidden size for BiLSTM (default: 128)
- `--lstm_layers`: Number of BiLSTM layers (default: 1)
- `--dropout`: Dropout rate (default: 0.3)

**Testing Arguments:**

- `--batch_size`: Batch size for testing (default: 32)
- `--max_len`: Maximum sequence length (default: 256)
- `--num_classes`: Number of classes to predict (default: 2)
- `--class_names`: Names of classes in order (default: `Normal Spam`)

**Other:**

- `--output_dir`: Directory to save results (default: `results/test_results_hybrid`)
- `--seed`: Random seed (default: 42)

### Hybrid Model Evaluation Outputs

Results are saved in a timestamped directory under `{output_dir}/{timestamp}/`:

- `test_metrics.json`: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
- `classification_report.txt`: Detailed classification report with per-class metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot (for binary and multiclass)
- `precision_recall_curve.png`: Precision-Recall curve plot
- `class_distribution.png`: Class distribution comparison (true vs predicted)
- `predictions.csv`: Predictions with probabilities and confidence scores for error analysis

The predictions CSV includes:

- Original text and labels
- True and predicted labels
- Probability for each class
- Prediction confidence (max probability)
- Correctness indicator
- Sorted by confidence (ascending) to highlight uncertain predictions
- Sample misclassified examples with confidence scores

## Baseline Model Evaluation

### Basic Evaluation

Evaluate a trained baseline model on test data:

```bash
python tools/test_baselines.py \
    --test_csv dataset/data/test.csv \
    --model_path results/baseline_results/logistic_tfidf/2025-01-15_10-30-45/logistic_tfidf_model.pkl \
    --output_dir results/baseline_test_results
```

### Evaluation with Preprocessing

Apply preprocessing before evaluation:

```bash
python tools/test_baselines.py \
    --test_csv dataset/data/test.csv \
    --model_path results/baseline_results/logistic_tfidf/2025-01-15_10-30-45/logistic_tfidf_model.pkl \
    --output_dir results/baseline_test_results \
    --preprocess \
    --mapping_path mapping.json \
    --vncorenlp_dir ../notebooks/vncorenlp
```

### Full Baseline Evaluation Example

```bash
python tools/test_baselines.py \
    --test_csv dataset/data/test.csv \
    --model_path results/baseline_results/logistic_tfidf/2025-01-15_10-30-45/logistic_tfidf_model.pkl \
    --output_dir results/baseline_test_results \
    --text_col segmented_comment \
    --label_col label \
    --num_classes 2 \
    --class_names Normal Spam \
    --preprocess \
    --mapping_path mapping.json \
    --vncorenlp_dir ../notebooks/vncorenlp \
    --seed 42
```

### Baseline Test Arguments

**Required:**

- `--test_csv`: Path to test CSV file
- `--model_path`: Path to saved model (.pkl file)

**Data Arguments:**

- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)
- `--config_path`: Path to model config JSON (optional, auto-detected from model directory)

**Output Arguments:**

- `--output_dir`: Directory to save results (default: `results/baseline_test_results`)
- `--num_classes`: Number of classes (default: 2)
- `--class_names`: Class names in order (default: `Normal Spam`)

**Preprocessing Arguments:**

- `--preprocess`: Apply preprocessing before evaluation
- `--mapping_path`: Path to typo mapping JSON (default: `src/mapping.json`)
- `--vncorenlp_dir`: Path to VnCoreNLP models (default: `../notebooks/vncorenlp`)

**Other:**

- `--seed`: Random seed (default: 42)

### Baseline Evaluation Outputs

Results are saved in a timestamped directory under `{output_dir}/{timestamp}/`:

- `test_metrics.json`: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
- `classification_report.txt`: Detailed classification report
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `precision_recall_curve.png`: Precision-Recall curve plot
- `class_distribution.png`: Class distribution comparison
- `predictions.csv`: Predictions with probabilities for error analysis
- `feature_importance.json`: Top features used by the model (for logistic/linear_svm)


## Project Structure Details

### Key Modules

- **`src/dataset/dataset.py`**: Custom PyTorch Dataset class for loading CSV data
- **`src/models/PhoBert.py`**: PhoBERT-based classifier model
- **`src/models/hybrid_model.py`**: Hybrid model combining PhoBERT, CNN, and BiLSTM
- **`src/preprocessing/preprocessing.py`**: Text preprocessing pipeline (typo mapping, segmentation)
- **`src/preprocessing/imbalance_handler.py`**: Data augmentation methods (back-translation, random oversampling)
- **`src/tools/train.py`**: Training script for PhoBERT with preprocessing and augmentation support
- **`src/tools/train_hybrid.py`**: Training script for Hybrid model
- **`src/tools/train_baselines.py`**: Training script for baseline models (SVM rbf, linear kernal, and logistic regression)
- **`src/tools/test.py`**: Evaluation script for PhoBERT with comprehensive metrics
- **`src/tools/test_hybrid.py`**: Evaluation script for Hybrid model
- **`src/tools/test_baselines.py`**: Evaluation script for baseline models (SVM rbf, linear kernal, and logistic regression)
- **`src/utils/metrics.py`**: Evaluation metrics calculation
- **`src/utils/visualization.py`**: Plotting utilities for training history and evaluation results
- **`src/utils/early_stopping.py`**: Early stopping callback for training

### Data Flow

1. **Raw Data** → Preprocessing (typo mapping + segmentation) → **Preprocessed Data**
2. **Preprocessed Data** → Augmentation (optional, train set only) → **Augmented Data**
3. **Augmented Data** → Training → **Model Checkpoints**
4. **Test Data** → Preprocessing (optional) → **Evaluation** → **Results**

## Notes

- Data augmentation is **only applied to the training set**, never to validation or test sets
- Preprocessing can be applied during training/testing or done separately
- The model uses a fixed task name: `counterfeit_reviews_classification`
- The Hybrid model combines PhoBERT embeddings with CNN (for local features) and BiLSTM (for sequential dependencies)
- Model architecture parameters (CNN channels, LSTM hidden size, layers) must match between training and testing
- All paths are resolved relative to the project root for portability
- VnCoreNLP models are automatically downloaded if not found

## Troubleshooting

### VnCoreNLP Issues

- Ensure Java is installed and accessible
- Check that VnCoreNLP models are in the correct directory
- Models will auto-download on first use if missing

### Path Issues

- Use forward slashes (`/`) in paths for cross-platform compatibility
- Relative paths are resolved from the project root
- Use absolute paths if you encounter path resolution issues

### CUDA/GPU Issues

- The script auto-detects CUDA availability
- Use `--device cpu` to force CPU usage
- Ensure PyTorch is installed with CUDA support if using GPU

## License
