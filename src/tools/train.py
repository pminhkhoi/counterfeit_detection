"""
Training utilities for model training and validation.
"""
import os
import gc
import json
import torch
from torch import nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from utils.visualization import save_training_history_plots
from utils.early_stopping import EarlyStopping
from dataset.dataset import CSVDataset
from models.PhoBert import ViSpam_Classifier
from preprocess.preprocessing import preprocessing
from preprocess.imbalance_handler import apply_back_translation, apply_random_oversampling


def train_step(model, criterion, optimizer, lr_scheduler, train_dataloader, device='cpu'):
    """
    Perform one training epoch.
    
    Args:
        model: The model to train
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
        
        # Handle optional category_id
        category_id = data.get('category_id')
        if category_id is not None:
            category_id = category_id.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        if category_id is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_id=category_id)
        else:
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
        correct += torch.sum(pred == labels).item()
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
        model: The model to evaluate
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
            
            # Handle optional category_id
            category_id = data.get('category_id')
            if category_id is not None:
                category_id = category_id.to(device)
            
            # Forward pass
            if category_id is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_id=category_id)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            pred = torch.max(outputs, dim=1)[1]
            loss = criterion(outputs, labels)
            
            # Accumulate metrics
            correct += torch.sum(pred == labels).item()
            losses.append(loss.item())
            trues.extend(labels.cpu().detach().numpy())
            predicts.extend(pred.cpu().detach().numpy())
            
    accuracy = correct / len(dataloader.dataset)
    avg_loss = np.mean(losses)
    f1 = f1_score(trues, predicts, average='macro')

    return accuracy, f1, avg_loss


def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs, 
          early_stopping=None, task='counterfeit_reviews_classification', lr_scheduler=None, device='cpu'):
    """
    Main training loop.
    
    Args:
        model: The model to train
        criterion: Loss function
        optimizer: Optimizer
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        epochs: Number of epochs to train
        early_stopping: EarlyStopping instance (optional)
        task: Task name for saving files
        lr_scheduler: Learning rate scheduler (optional)
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing training history
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create output directory
    output_dir = Path(model.model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup early stopping checkpoint path
    if early_stopping:
        path_checkpoint = output_dir / f"{model.model_name}_checkpoint_({task}).pth"
        early_stopping.path = str(path_checkpoint)
    
    # Initialize tracking variables
    best_f1 = 0
    best_model_path = output_dir / f"{model.model_name}_best_model_({task}).pth"
    last_model_path = output_dir / f"{model.model_name}_last_model_({task}).pth"
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
    history_path = output_dir / f"{model.model_name}_history_({task}).json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Generate visualization plots
    try:
        save_training_history_plots(history, model.model_name, task, output_dir)
        print(f"Training visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not generate training visualizations: {e}")
    
    return history


def main():
    """
    Main entry point for training script.
    """
    parser = argparse.ArgumentParser(description='Train counterfeit reviews classification model')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--dev_csv', type=str, required=True,
                       help='Path to validation/dev CSV file')
    parser.add_argument('--text_col', type=str, default='segmented_comment',
                       help='Column name for text input (default: segmented_comment)')
    parser.add_argument('--label_col', type=str, default='label',
                       help='Column name for labels (default: label)')
    
    # Preprocessing arguments
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing (typo mapping + segmentation) before training')
    parser.add_argument('--mapping_path', type=str, default=None,
                       help='Path to typo mapping JSON file (default: src/preprocessing/mapping.json)')
    parser.add_argument('--vncorenlp_dir', type=str, default=None,
                       help='Path to VnCoreNLP models directory (default: notebooks/vncorenlp)')
    
    # Data augmentation arguments (only applied to train set)
    parser.add_argument('--augment', type=str, choices=['none', 'back_translation', 'random'],
                       default='none',
                       help='Data augmentation method for training set only (default: none)')
    parser.add_argument('--augmentation_factor', type=float, default=2.0,
                       help='Augmentation factor for back-translation (default: 2.0)')
    parser.add_argument('--sampling_strategy', type=str, default='auto',
                       help='Sampling strategy for random oversampling: "auto" for full balance, or float for partial (default: auto)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base',
                       help='Pretrained model name (default: vinai/phobert-base)')
    parser.add_argument('--freeze_bert', action='store_true',
                       help='Freeze BERT parameters')
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
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_len}")
    print(f"Preprocessing: {'Yes' if args.preprocess else 'No'}")
    print(f"Data augmentation: {args.augment}")
    print("-"*60)
    
    # Load data
    import pandas as pd
    
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
    
    # Apply preprocessing if requested
    if args.preprocess:
        print("\n" + "="*60)
        print("APPLYING PREPROCESSING")
        print("="*60)
        
        # Resolve paths relative to project root
        # Get project root (parent of src directory)
        script_dir = Path(__file__).resolve().parent  # src/tools
        project_root = script_dir.parent.parent  # src/tools -> src -> project_root
        
        # Set default paths if not provided
        if args.mapping_path is None:
            # Try common locations
            possible_mapping_paths = [
                project_root / 'src' / 'preprocessing' / 'mapping.json',
                project_root / 'src' / 'mapping.json',
                project_root / 'preprocessing' / 'mapping.json'
            ]
            mapping_path = None
            for path in possible_mapping_paths:
                if path.exists():
                    mapping_path = path
                    break
            if mapping_path is None:
                # Default to src/preprocessing/mapping.json
                mapping_path = project_root / 'src' / 'preprocessing' / 'mapping.json'
        else:
            # Resolve user-provided path
            mapping_path = Path(args.mapping_path)
            if not mapping_path.is_absolute():
                mapping_path = (project_root / mapping_path).resolve()
        
        if args.vncorenlp_dir is None:
            vncorenlp_dir = project_root / 'notebooks' / 'vncorenlp'
        else:
            vncorenlp_dir = Path(args.vncorenlp_dir)
            if not vncorenlp_dir.is_absolute():
                vncorenlp_dir = (project_root / vncorenlp_dir).resolve()
        
        print(f"Mapping file: {mapping_path}")
        print(f"VnCoreNLP directory: {vncorenlp_dir}")
        
        # Verify mapping file exists
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Typo mapping file not found at: {mapping_path}\n"
                f"Please provide the correct path using --mapping_path"
            )
        
        # Check if 'comment' column exists (required for preprocessing)
        if 'comment' not in train_df.columns:
            print("Warning: 'comment' column not found in train set. Using existing text column for preprocessing.")
            if args.text_col in train_df.columns:
                train_df['comment'] = train_df[args.text_col]
            else:
                raise ValueError(f"Neither 'comment' nor '{args.text_col}' column found in train set for preprocessing")
        
        if 'comment' not in dev_df.columns:
            if args.text_col in dev_df.columns:
                dev_df['comment'] = dev_df[args.text_col]
            else:
                raise ValueError(f"Neither 'comment' nor '{args.text_col}' column found in dev set for preprocessing")
        
        # Apply preprocessing to both train and dev sets
        print("\nPreprocessing training set...")
        train_df = preprocessing(
            train_df,
            mapper_path=mapping_path,
            vncorenlp_dir=vncorenlp_dir,
            save_csv_path=None
        )
        
        print("Preprocessing validation set...")
        dev_df = preprocessing(
            dev_df,
            mapper_path=mapping_path,
            vncorenlp_dir=vncorenlp_dir,
            save_csv_path=None
        )
        
        # Update text column to use segmented_comment if it exists
        if 'segmented_comment' in train_df.columns:
            print("Using 'segmented_comment' column for training")
            args.text_col = 'segmented_comment'
        elif 'preprocessed_review' in train_df.columns:
            print("Using 'preprocessed_review' column for training")
            args.text_col = 'preprocessed_review'
    
    # Apply data augmentation to TRAIN SET ONLY
    if args.augment != 'none':
        print("\n" + "="*60)
        print(f"APPLYING DATA AUGMENTATION ({args.augment.upper()}) TO TRAIN SET")
        print("="*60)
        print("Note: Augmentation is only applied to training set, not validation set.")
        
        # Determine text column for augmentation
        aug_text_col = args.text_col
        if args.augment == 'back_translation':
            # Back-translation expects segmented_comment or preprocessed_review
            if 'segmented_comment' in train_df.columns:
                aug_text_col = 'segmented_comment'
            elif 'preprocessed_review' in train_df.columns:
                aug_text_col = 'preprocessed_review'
            else:
                aug_text_col = args.text_col
            
            print(f"Applying back-translation augmentation to train set...")
            print(f"Using text column: {aug_text_col}")
            print(f"Augmentation factor: {args.augmentation_factor}")
            
            # apply_back_translation can accept DataFrame directly
            train_df = apply_back_translation(
                train_csv=train_df,  # Pass DataFrame directly
                text_col=aug_text_col,
                label_col=args.label_col,
                output_csv=None,
                augmentation_factor=args.augmentation_factor,
                random_state=args.seed,
                device=device
            )
        
        elif args.augment == 'random':
            print(f"Applying random oversampling to train set...")
            print(f"Sampling strategy: {args.sampling_strategy}")
            
            # apply_random_oversampling only accepts Path, so save temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig') as f:
                temp_path = f.name
                train_df.to_csv(temp_path, index=False, encoding='utf-8-sig')
            
            try:
                train_df = apply_random_oversampling(
                    train_csv=Path(temp_path),
                    label_col=args.label_col,
                    output_csv=None,
                    sampling_strategy=args.sampling_strategy,
                    random_state=args.seed
                )
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Show updated class distribution
        if args.label_col in train_df.columns:
            train_dist = train_df[args.label_col].value_counts().sort_index()
            print(f"\nUpdated training set class distribution:\n{train_dist}")
            print(f"Training samples: {len(train_df)} (augmented)")
    
    # Initialize tokenizer
    print(f"\nInitializing tokenizer ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
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
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = ViSpam_Classifier(
        model_name=args.model_name,
        freeze_bert=args.freeze_bert,
        drop=args.dropout
    )
    model.to(device)
    
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
    
    history = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=dev_loader,
        epochs=args.epochs,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        device=device
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"\nTraining history saved. Best model saved in: {model.model_name}/")


if __name__ == '__main__':
    main()