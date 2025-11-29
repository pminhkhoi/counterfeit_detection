"""
Training utilities for Hybrid Model (PhoBERT + CNN + BiLSTM).
"""
import datetime
import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
import logging
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from models.hybrid_model import HybridModel
from dataset.dataset import CSVDataset
from utils.early_stopping import EarlyStopping
from utils.visualization import save_training_history_plots

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")


def train_step(model, criterion, optimizer, lr_scheduler, train_dataloader, device='cpu'):
    """
    Perform one training epoch.

    Args:
        model: The Hybrid model to train
        criterion: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        train_dataloader: Training data loader
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Tuple of (accuracy, f1_score, loss)
    """
    model.train()
    model.to(device)

    losses = []
    correct = 0
    trues = []
    predicts = []

    for data in tqdm(train_dataloader, desc="Training"):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        pred = torch.max(outputs, dim=1)[1]

        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Accumulate metrics
        correct += torch.sum(torch.eq(pred, labels)).item()
        losses.append(loss.item())
        trues.extend(labels.cpu().detach().numpy())
        predicts.extend(pred.cpu().detach().numpy())

    accuracy = correct / len(train_dataloader.dataset)
    avg_loss = np.mean(losses)
    f1 = f1_score(trues, predicts, average='macro')

    return accuracy, f1, avg_loss


def validation_step(model, criterion, dataloader, device='cpu'):
    """
    Perform validation on a dataset.

    Args:
        model: The Hybrid model to evaluate
        criterion: Loss function
        dataloader: Validation data loader
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Tuple of (accuracy, f1_score, loss)
    """
    model.eval()
    model.to(device)

    losses = []
    correct = 0
    trues = []
    predicts = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating"):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.max(outputs, dim=1)[1]
            loss = criterion(outputs, labels)

            # Accumulate metrics
            correct += torch.sum(torch.eq(pred, labels)).item()
            losses.append(loss.item())
            trues.extend(labels.cpu().detach().numpy())
            predicts.extend(pred.cpu().detach().numpy())

    accuracy = correct / len(dataloader.dataset)
    avg_loss = np.mean(losses)
    f1 = f1_score(trues, predicts, average='macro')

    return accuracy, f1, avg_loss


def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs,
          early_stopping=None, output_dir=Path('results/checkpoints'), task='hybrid_counterfeit_classification',
          lr_scheduler=None, device='cpu'):
    """
    Main training loop for Hybrid model.

    Args:
        model: The Hybrid model to train
        criterion: Loss function
        optimizer: Optimizer
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        epochs: Number of epochs to train
        early_stopping: EarlyStopping instance (optional)
        output_dir: Output directory for checkpoints
        task: Task name for saving files
        lr_scheduler: Learning rate scheduler (optional)
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary containing training history
    """
    torch.cuda.empty_cache()
    gc.collect()

    # Create output directory
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup early stopping checkpoint path
    if early_stopping:
        path_checkpoint = output_dir / f"hybrid_model_checkpoint_({task}).pth"
        early_stopping.path = str(path_checkpoint)

    # Initialize tracking variables
    best_f1 = 0
    best_model_path = output_dir / f"hybrid_model_best_{task}.pth"
    last_model_path = output_dir / f"hybrid_model_last_{task}.pth"
    history = {'train_acc': [], 'train_f1': [], 'train_loss': [],
               'val_acc': [], 'val_f1': [], 'val_loss': []}

    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("-"*60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*60)

        # Training step
        train_accuracy, train_f1, train_loss = train_step(
            model, criterion, optimizer, lr_scheduler, train_dataloader, device
        )

        # Validation step
        val_accuracy, val_f1, val_loss = validation_step(
            model, criterion, val_dataloader, device
        )

        # Update history
        history['train_acc'].append(float(train_accuracy))
        history['train_f1'].append(float(train_f1))
        history['train_loss'].append(float(train_loss))
        history['val_acc'].append(float(val_accuracy))
        history['val_f1'].append(float(val_f1))
        history['val_loss'].append(float(val_loss))

        # Print metrics
        print(f"Train - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Loss: {train_loss:.4f}")
        print(f"Valid - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Loss: {val_loss:.4f}")

        # Save best model
        if val_f1 > best_f1:
            torch.save(model.state_dict(), best_model_path)
            best_f1 = val_f1
            print(f"✓ New best model saved! (F1: {best_f1:.4f})")

        # Save last model at final epoch
        if epoch + 1 == epochs:
            torch.save(model.state_dict(), last_model_path)
            print(f"Final model saved to {last_model_path}")

        # Early stopping check
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                torch.save(model.state_dict(), last_model_path)
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Save training history
    history_path = output_dir / f"hybrid_model_history_{task}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Generate visualization plots
    try:
        save_training_history_plots(history, "HybridModel", task, output_dir)
        print(f"Training visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not generate training visualizations: {e}")

    return history


def main():
    """
    Main entry point for Hybrid model training script.
    """
    parser = argparse.ArgumentParser(description='Train Hybrid model for counterfeit reviews classification')

    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--dev_csv', type=str, required=True,
                       help='Path to validation/dev CSV file')
    parser.add_argument('--text_col', type=str, default='segmented_comment',
                       help='Column name for text input (default: segmented_comment)')
    parser.add_argument('--label_col', type=str, default='label',
                       help='Column name for labels (default: label)')
    parser.add_argument('--out_dir', type=str, default='results/checkpoints',
                       help='Output directory for checkpoints')

    # PhoBERT arguments
    parser.add_argument('--phobert_model', type=str, default='vinai/phobert-base',
                       help='PhoBERT model name (default: vinai/phobert-base)')

    # Hybrid model architecture arguments
    parser.add_argument('--cnn_out_channels', type=int, default=128,
                       help='Number of output channels for each CNN layer (default: 128)')
    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                       help='Hidden size for BiLSTM (default: 128)')
    parser.add_argument('--lstm_layers', type=int, default=1,
                       help='Number of BiLSTM layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--max_len', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of warmup steps for learning rate scheduler (default: 0)')

    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=7,
                       help='Patience for early stopping (default: 7)')
    parser.add_argument('--early_stopping_delta', type=float, default=0,
                       help='Minimum change to qualify as improvement (default: 0)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*60)
    print("HYBRID MODEL TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"PhoBERT Model: {args.phobert_model}")
    print(f"CNN Output Channels: {args.cnn_out_channels}")
    print(f"LSTM Hidden Size: {args.lstm_hidden_size}")
    print(f"LSTM Layers: {args.lstm_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_len}")
    print("-"*60)

    # Load data
    print(f"\nLoading training data from {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Training samples: {len(train_df)}")

    print(f"Loading validation data from {args.dev_csv}...")
    dev_df = pd.read_csv(args.dev_csv)
    print(f"Validation samples: {len(dev_df)}")

    # Show initial class distribution
    if args.label_col in train_df.columns:
        train_dist = train_df[args.label_col].value_counts().sort_index()
        print(f"\nInitial training set class distribution:\n{train_dist}")

    # Initialize tokenizer
    print(f"\nInitializing tokenizer ({args.phobert_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.phobert_model)

    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = CSVDataset(
        train_df,
        args.text_col,
        args.label_col,
        tokenizer,
        args.max_len
    )

    dev_dataset = CSVDataset(
        dev_df,
        args.text_col,
        args.label_col,
        tokenizer,
        args.max_len
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(dev_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize Hybrid model
    print(f"\nInitializing Hybrid model...")
    model = HybridModel(
        phobert_model_name=args.phobert_model,
        cnn_out_channels=args.cnn_out_channels,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        num_classes=2,
        dropout=args.dropout
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Initialize learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    lr_scheduler = None
    if args.warmup_steps > 0:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Learning rate scheduler enabled with {args.warmup_steps} warmup steps")

    # Initialize early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            verbose=True,
            delta=args.early_stopping_delta
        )
        print(f"Early stopping enabled with patience={args.patience}")

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) / folder_name

    history = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=dev_loader,
        epochs=args.epochs,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        output_dir=out_dir,
        device=device,
        task='hybrid_counterfeit_reviews_classification'
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"\nTraining history saved to: {out_dir}")


if __name__ == '__main__':
    main()