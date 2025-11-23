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
│   │
│   ├── models/
│   │   └── PhoBert.py      # PhoBERT classifier model
│   │
│   ├── preprocessing/
│   │   ├── preprocessing.py      # Text preprocessing pipeline
│   │   └── imbalance_handler.py # Data augmentation methods
│   │
│   ├── tools/
│   │   ├── train.py        # Training script
│   │   └── test.py         # Evaluation script
│   │
│   ├── utils/
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── visualization.py     # Plotting utilities
│   │   └── early_stopping.py    # Early stopping callback
│   │
│   ├── results/
│   │   ├── checkpoints/    # Saved model checkpoints
│   │   └── test_results/   # Evaluation results
│   │
│   └── mapping.json        # Typo mapping dictionary
│
├── requirements.txt
└── README.md
```

## Setting Up

### 1. Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- Java Runtime Environment (JRE) for VnCoreNLP

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download VnCoreNLP Models

VnCoreNLP models are required for text segmentation. The models should be placed in `notebooks/vncorenlp/`.

If models are not present, the preprocessing script will attempt to download them automatically. Alternatively, you can download manually:

```bash
# The models will be auto-downloaded on first use, or you can:
python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='notebooks/vncorenlp')"
```

### 4. Prepare Data

Your CSV files should contain at least:

- A text column (e.g., `comment` or `segmented_comment`)
- A label column (e.g., `label`) with binary values (0/1 or Normal/Spam)

Example CSV structure:

```csv
comment,label
"Đây là một sản phẩm tuyệt vời",0
"Spam review fake",1
```

## Data Preprocessing

The preprocessing pipeline includes:

1. **Typo mapping**: Normalizes common typos and teen-code
2. **Word segmentation**: Uses VnCoreNLP to segment Vietnamese text

### Standalone Preprocessing

You can preprocess data separately using:

```bash
python src/preprocessing/preprocessing.py \
    --csv data/train.csv \
    --mapping src/mapping.json \
    --vncorenlp_dir notebooks/vncorenlp \
    --out_dir src/dataset/data \
    --train_ratio 0.8 \
    --stratify_col label
```

Options:

- `--csv`: Input CSV file (must contain 'comment' column)
- `--mapping`: Path to typo mapping JSON file
- `--vncorenlp_dir`: VnCoreNLP models directory
- `--out_dir`: Output directory for processed files
- `--train_ratio`: Train/validation split ratio (default: 0.8)
- `--stratify_col`: Column for stratified splitting (default: 'label')
- `--balance_method`: Data balancing method: `none`, `back_translation`, or `random`
- `--augmentation_factor`: Augmentation factor for back-translation (default: 2.0)

## Training

### Basic Training (Preprocessed Data)

If your data is already preprocessed:

```bash
python src/tools/train.py \
    --train_csv src/train.csv \
    --dev_csv src/dev.csv \
    --text_col segmented_comment \
    --label_col label
```

### Training with Preprocessing

Apply preprocessing during training:

```bash
python src/tools/train.py \
    --train_csv data/train.csv \
    --dev_csv data/dev.csv \
    --preprocess \
    --mapping_path src/mapping.json \
    --vncorenlp_dir notebooks/vncorenlp
```

### Training with Data Augmentation

Apply augmentation to the training set only:

**Back-translation augmentation:**

```bash
python src/tools/train.py \
    --train_csv data/train.csv \
    --dev_csv data/dev.csv \
    --preprocess \
    --augment back_translation \
    --augmentation_factor 2.0
```

**Random oversampling:**

```bash
python src/tools/train.py \
    --train_csv data/train.csv \
    --dev_csv data/dev.csv \
    --preprocess \
    --augment random \
    --sampling_strategy auto
```

### Full Training Example with All Options

```bash
python src/tools/train.py \
    --train_csv data/train.csv \
    --dev_csv data/dev.csv \
    --preprocess \
    --augment back_translation \
    --augmentation_factor 2.0 \
    --model_name vinai/phobert-base \
    --epochs 10 \
    --batch_size 32 \
    --max_len 256 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --early_stopping \
    --patience 7 \
    --seed 42
```

### Training Arguments

**Data Arguments:**

- `--train_csv`: Path to training CSV file (required)
- `--dev_csv`: Path to validation/dev CSV file (required)
- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)

**Preprocessing Arguments:**

- `--preprocess`: Apply preprocessing (typo mapping + segmentation)
- `--mapping_path`: Path to typo mapping JSON (auto-detected if not provided)
- `--vncorenlp_dir`: Path to VnCoreNLP models (default: `notebooks/vncorenlp`)

**Augmentation Arguments (train set only):**

- `--augment`: Augmentation method: `none`, `back_translation`, or `random` (default: `none`)
- `--augmentation_factor`: Augmentation factor for back-translation (default: 2.0)
- `--sampling_strategy`: Sampling strategy for random oversampling: `auto` or float (default: `auto`)

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

### Training Outputs

After training, the following files are saved in `{model_name}/` directory:

- `{model_name}_best_model_(counterfeit_reviews_classification).pth`: Best model checkpoint
- `{model_name}_last_model_(counterfeit_reviews_classification).pth`: Final epoch model
- `{model_name}_checkpoint_(counterfeit_reviews_classification).pth`: Early stopping checkpoint (if enabled)
- `{model_name}_history_(counterfeit_reviews_classification).json`: Training history
- `{model_name}_loss_visualization_(counterfeit_reviews_classification).png`: Loss plots
- `{model_name}_accuracy_visualization_(counterfeit_reviews_classification).png`: Accuracy plots

## Testing/Evaluation

### Basic Evaluation

Evaluate a trained model on test data:

```bash
python src/tools/test.py \
    --test_csv data/test.csv \
    --checkpoint src/results/checkpoints/2025-11-23_12-48-39/best_model_spam_classification.pth \
    --output_dir src/results/test_results
```

### Evaluation with Preprocessing

Apply preprocessing before evaluation:

```bash
python src/tools/test.py \
    --test_csv data/test.csv \
    --checkpoint src/results/checkpoints/2025-11-23_12-48-39/best_model_spam_classification.pth \
    --output_dir src/results/test_results \
    --preprocess \
    --mapping_path src/mapping.json \
    --vncorenlp_dir notebooks/vncorenlp
```

### Full Evaluation Example

```bash
python src/tools/test.py \
    --test_csv data/test.csv \
    --checkpoint src/results/checkpoints/2025-11-23_12-48-39/best_model_spam_classification.pth \
    --output_dir src/results/test_results \
    --text_col segmented_comment \
    --label_col label \
    --model_name vinai/phobert-base \
    --batch_size 32 \
    --max_len 256 \
    --num_classes 2 \
    --class_names Normal Spam \
    --preprocess \
    --seed 42
```

### Test Arguments

**Required:**

- `--test_csv`: Path to test CSV file
- `--checkpoint`: Path to model checkpoint (.pth file)

**Data Arguments:**

- `--text_col`: Column name for text input (default: `segmented_comment`)
- `--label_col`: Column name for labels (default: `label`)

**Preprocessing Arguments:**

- `--preprocess`: Apply preprocessing before evaluation
- `--mapping_path`: Path to typo mapping JSON (default: `src/preprocessing/mapping.json`)
- `--vncorenlp_dir`: Path to VnCoreNLP models (default: `notebooks/vncorenlp`)

**Model Arguments:**

- `--model_name`: Pretrained model name (default: `vinai/phobert-base`)
- `--batch_size`: Batch size for testing (default: 32)
- `--max_len`: Maximum sequence length (default: 256)
- `--num_classes`: Number of classes (default: 2)
- `--class_names`: Class names in order (default: `Normal Spam`)

**Other:**

- `--output_dir`: Directory to save results (default: `../results/test_results`)
- `--seed`: Random seed (default: 42)

### Evaluation Outputs

Results are saved in a timestamped directory under `output_dir/`:

- `test_metrics.json`: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
- `classification_report.txt`: Detailed classification report
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `precision_recall_curve.png`: Precision-Recall curve plot
- `class_distribution.png`: Class distribution comparison
- `predictions.csv`: Predictions with probabilities for error analysis

## Project Structure Details

### Key Modules

- **`src/dataset/dataset.py`**: Custom PyTorch Dataset class for loading CSV data
- **`src/models/PhoBert.py`**: PhoBERT-based classifier model
- **`src/preprocessing/preprocessing.py`**: Text preprocessing pipeline (typo mapping, segmentation)
- **`src/preprocessing/imbalance_handler.py`**: Data augmentation methods (back-translation, random oversampling)
- **`src/tools/train.py`**: Training script with preprocessing and augmentation support
- **`src/tools/test.py`**: Evaluation script with comprehensive metrics
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
