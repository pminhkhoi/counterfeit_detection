
import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

class BackTranslator:
    """
    Back-translation augmentation using MarianMT models.
    Translates Vietnamese -> English -> Vietnamese to create paraphrases.
    """

    def __init__(self, device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"):
        self.device = device

        # Model names for Vietnamese <-> English translation
        self.vi_to_en_model_name = "vinai/vinai-translate-vi2en-v2"
        self.en_to_vi_model_name = "vinai/vinai-translate-en2vi-v2"

        logger.info("Loading translation models (this may take a while)...")

        # Vietnamese to English
        self.vi_to_en_tokenizer = AutoTokenizer.from_pretrained(self.vi_to_en_model_name)
        self.vi_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(self.vi_to_en_model_name).to(self.device)

        # English to Vietnamese
        self.en_to_vi_tokenizer = AutoTokenizer.from_pretrained(self.en_to_vi_model_name)
        self.en_to_vi_model = AutoModelForSeq2SeqLM.from_pretrained(self.en_to_vi_model_name).to(self.device)

        logger.info("Translation models loaded successfully!")

    def translate_vi_to_en(self, text: str, max_length: int = 1024) -> str:
        """Translate Vietnamese text to English."""
        if not text or not text.strip():
            return text

        try:
            # Tokenize Vietnamese input
            input_ids = self.vi_to_en_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Generate English translation
            output_ids = self.vi_to_en_model.generate(
                **input_ids, 
                decoder_start_token_id=self.vi_to_en_tokenizer.lang_code_to_id["en_XX"], 
                num_return_sequences=1, 
                num_beams=5, 
                early_stopping=True,
                max_length=max_length
            )
            en_texts = self.vi_to_en_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return en_texts[0] if isinstance(en_texts, list) else en_texts
        except Exception as e:
            logger.warning(f"Vietnamese to English translation failed: {e}")
            return text

    def translate_en_to_vi(self, text: str, max_length: int = 1024) -> str:
        """Translate English text to Vietnamese."""
        if not text or not text.strip():
            return text

        try:
            # Tokenize English input
            input_ids = self.en_to_vi_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Generate Vietnamese translation
            output_ids = self.en_to_vi_model.generate(
                **input_ids, 
                decoder_start_token_id=self.en_to_vi_tokenizer.lang_code_to_id["vi_VN"], 
                num_return_sequences=1, 
                num_beams=5, 
                early_stopping=True,
                max_length=max_length
            )
            vi_texts = self.en_to_vi_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return vi_texts[0] if isinstance(vi_texts, list) else vi_texts
        except Exception as e:
            logger.warning(f"English to Vietnamese translation failed: {e}")
            return text

    def back_translate(self, text: str, max_length: int = 1024) -> str:
        """
        Perform back-translation: Vietnamese -> English -> Vietnamese
        This creates a paraphrased version of the input text.
        """
        if not text or not text.strip():
            return text

        try:
            # Step 1: Vietnamese -> English
            en_text = self.translate_vi_to_en(text, max_length)
            
            # Step 2: English -> Vietnamese
            vi_text = self.translate_en_to_vi(en_text, max_length)
            
            return vi_text
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text

    def augment_text(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Generate multiple augmented versions of the text using back-translation.
        """
        augmented = []
        for _ in range(num_augmentations):
            aug_text = self.back_translate(text)
            if aug_text and aug_text != text:  # Only add if different from original
                augmented.append(aug_text)
        return augmented


def apply_back_translation(
        train_csv: Union[Path, pd.DataFrame],
        text_col: str = "preprocessed_review",
        label_col: str = "label",
        output_csv: Optional[Path] = None,
        minority_class: Optional[int] = None,
        augmentation_factor: float = 2.0,
        random_state: int = 42,
        device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Apply back-translation to augment minority class samples.

    Args:
        train_csv: Path to training CSV file
        text_col: Column containing text data
        label_col: Column containing labels
        output_csv: Optional path to save augmented dataset
        minority_class: Specific class to augment (if None, auto-detect minority)
        augmentation_factor: Target ratio for minority class (e.g., 2.0 means double the samples)
        random_state: Random seed
        device: 'cuda' or 'cpu'

    Returns:
        Augmented DataFrame
    """
    if isinstance(train_csv, Path):
        logger.info(f"Loading data from {train_csv}...")
        df = pd.read_csv(train_csv)
    elif isinstance(train_csv, pd.DataFrame):
        df = train_csv
    else:
        raise TypeError("Input must be Path or DataFrame type")

    # Check class distribution
    class_dist = df[label_col].value_counts().sort_index()
    logger.info(f"Original class distribution:\n{class_dist}")

    # Identify minority class
    if minority_class is None:
        minority_class = class_dist.idxmin()
        logger.info(f"Auto-detected minority class: {minority_class}")

    majority_count = class_dist.max()
    minority_count = class_dist[minority_class]

    # Calculate how many samples to generate
    target_count = int(minority_count * augmentation_factor)
    num_to_generate = min(target_count - minority_count, int(majority_count - minority_count))

    if num_to_generate <= 0:
        logger.warning("No augmentation needed. Classes are already balanced.")
        return df

    logger.info(f"Will generate {num_to_generate} synthetic samples for class {minority_class}")

    # Initialize back-translator
    translator = BackTranslator(device=device)

    # Get minority samples
    minority_samples = df[df[label_col] == minority_class].copy()

    # Randomly select samples to augment (with replacement if needed)
    random.seed(random_state)
    np.random.seed(random_state)

    samples_to_augment = minority_samples.sample(n=num_to_generate, replace=True, random_state=random_state)

    # Generate augmented samples
    augmented_rows = []
    logger.info("Generating augmented samples via back-translation...")

    for idx, row in tqdm(samples_to_augment.iterrows(), total=len(samples_to_augment), desc="Back-translating"):
        original_text = row[text_col]

        # Perform back-translation
        augmented_text = translator.back_translate(original_text)

        # Create new row
        new_row = row.to_dict()
        new_row[text_col] = augmented_text
        new_row['is_augmented'] = True
        new_row['augmentation_method'] = 'back_translation'

        augmented_rows.append(new_row)

    # Combine original and augmented data
    augmented_df = pd.DataFrame(augmented_rows)
    df['is_augmented'] = False
    df['augmentation_method'] = 'original'

    combined_df = pd.concat([df, augmented_df], ignore_index=True)

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Check new distribution
    new_dist = combined_df[label_col].value_counts().sort_index()
    logger.info(f"Augmented class distribution:\n{new_dist}")
    logger.info(f"Total samples: {len(df)} -> {len(combined_df)}")
    logger.info(f"Augmented samples created: {len(augmented_rows)}")

    # Save if requested
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"Augmented dataset saved to {output_csv}")

    return combined_df


# ==========================================
# 3. Simple Random Oversampling (Baseline)
# ==========================================

def apply_random_oversampling(
        train_csv: Path,
        label_col: str = "label",
        output_csv: Optional[Path] = None,
        sampling_strategy: str = "auto",
        random_state: int = 42
) -> pd.DataFrame:
    """
    Simple random oversampling by duplicating minority class samples.

    Args:
        train_csv: Path to training CSV file
        label_col: Column containing labels
        output_csv: Optional path to save balanced dataset
        sampling_strategy: 'auto' for full balance, or float for partial balance
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    logger.info(f"Loading data from {train_csv}...")
    df = pd.read_csv(train_csv)

    # Check class distribution
    class_dist = df[label_col].value_counts().sort_index()
    logger.info(f"Original class distribution:\n{class_dist}")

    majority_count = class_dist.max()

    balanced_dfs = []
    for class_label, count in class_dist.items():
        class_df = df[df[label_col] == class_label]

        if sampling_strategy == "auto":
            target_count = majority_count
        else:
            target_count = int(majority_count * float(sampling_strategy))

        if count < target_count:
            # Oversample
            num_to_add = target_count - count
            oversampled = class_df.sample(n=num_to_add, replace=True, random_state=random_state)
            oversampled['is_oversampled'] = True
            class_df = class_df.copy()
            class_df['is_oversampled'] = False
            balanced_dfs.append(pd.concat([class_df, oversampled], ignore_index=True))
        else:
            class_df = class_df.copy()
            class_df['is_oversampled'] = False
            balanced_dfs.append(class_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Check new distribution
    new_dist = balanced_df[label_col].value_counts().sort_index()
    logger.info(f"Balanced class distribution:\n{new_dist}")
    logger.info(f"Total samples: {len(df)} -> {len(balanced_df)}")

    # Save if requested
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        balanced_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"Balanced dataset saved to {output_csv}")

    return balanced_df


# ==========================================
# 4. Main Function
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Handle imbalanced data using SMOTE or Back-Translation")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV file")
    parser.add_argument("--method", choices=["back_translation", "random"],
                        default="back_translation", help="Balancing method")
    parser.add_argument("--text_col", default="preprocessed_review", help="Text column name")
    parser.add_argument("--label_col", default="label", help="Label column name")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--sampling_strategy", default="auto",
                        help="Sampling strategy: 'auto' for full balance, float for partial")
    parser.add_argument("--augmentation_factor", type=float, default=2.0,
                        help="Augmentation factor for back-translation (only for back_translation method)")
    parser.add_argument("--minority_class", type=int, default=None,
                        help="Minority class to augment (auto-detect if None)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
                        help="Device for back-translation: 'cuda' or 'cpu'")

    args = parser.parse_args()

    logger.info(f"Starting imbalance handling with method: {args.method}")
    if args.method == "back_translation":
        balanced_df = apply_back_translation(
            train_csv=Path(args.train_csv),
            text_col=args.text_col,
            label_col=args.label_col,
            output_csv=Path(args.output_csv),
            minority_class=args.minority_class,
            augmentation_factor=args.augmentation_factor,
            random_state=args.random_state,
            device=args.device
        )
    elif args.method == "random":
        balanced_df = apply_random_oversampling(
            train_csv=Path(args.train_csv),
            label_col=args.label_col,
            output_csv=Path(args.output_csv),
            sampling_strategy=args.sampling_strategy,
            random_state=args.random_state
        )

    logger.info("Imbalance handling completed successfully!")
    logger.info(f"Balanced dataset shape: {balanced_df.shape}")


if __name__ == "__main__":
    main()

    # Back-translator
    # python imbalance_handler.py \
    #     --train_csv data/processed/train.csv \
    #     --method back_translation \
    #     --text_col segmented_comment \
    #     --label_col label \
    #     --output_csv data/processed/train_balanced_bt.csv \
    #     --augmentation_factor 2.0 \
    #     --device cuda
    
    # Random Oversampling
    # python imbalance_handler.py \
    #     --train_csv data/processed/train.csv \
    #     --method random \
    #     --output_csv data/processed/train_balanced_random.csv