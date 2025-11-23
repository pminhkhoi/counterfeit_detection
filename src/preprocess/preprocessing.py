# preprocessing.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from tqdm import tqdm
import py_vncorenlp
from .imbalance_handler import apply_back_translation, apply_random_oversampling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

# Global segmenter instance to avoid JVM conflicts
_SEGMENTER_INSTANCE = None


def load_typo_mapper(mapper_path: Path) -> Dict[str, str]:
    """Load JSON mapping file from disk."""
    mapper_path = Path(mapper_path)
    if not mapper_path.exists():
        raise FileNotFoundError(f"Typo mapping file not found at: {mapper_path}")
    with open(mapper_path, "r", encoding="utf-8") as fh:
        mapper = json.load(fh)
    if not isinstance(mapper, dict):
        raise ValueError("Mapping JSON must be an object (dict).")
    return mapper


def tokenize_words(review: str) -> list:
    """Simple whitespace tokenizer (lowercases first)."""
    if review is None:
        return []
    return str(review).lower().strip().split()


def mapping_typo_token(token: str, mapper: Dict[str, str]) -> str:
    """Map a single token using the mapper; return token if no mapping exists."""
    return mapper.get(token, token)


def segment_review(segmenter, review: str) -> str:
    """Use VnCoreNLP segmenter to segment a single review."""
    if review is None:
        return ""
    
    review = str(review).strip()
    if not review:
        return ""
    
    review = review.lower()
    try:
        segmented = segmenter.word_segment(review)
        if isinstance(segmented, list) and len(segmented) > 0:
            first = segmented[0]
            return " ".join(first) if isinstance(first, list) else first
        return review
    except Exception:
        logger.exception("VnCoreNLP segmentation failed; returning original review.")
        return review


def ensure_vncorenlp_models(vncorenlp_dir: Path, required_files: Optional[Iterable[str]] = None) -> bool:
    """
    Ensure the VnCoreNLP jar and model files exist in vncorenlp_dir.
    If missing, attempt automatic download (py_vncorenlp.download_model) if available.
    Returns True if models are present, False otherwise.
    """
    vncorenlp_dir = Path(vncorenlp_dir)
    vncorenlp_dir.mkdir(parents=True, exist_ok=True)

    if required_files is None:
        required_files = [
            "VnCoreNLP-1.2.jar",
            "models/wordsegmenter/vi-vocab",
            "models/wordsegmenter/wordsegmenter.rdr",
        ]

    missing = [f for f in required_files if not (vncorenlp_dir / f).exists()]

    if not missing:
        logger.info("VnCoreNLP models already present.")
        return True

    logger.info("VnCoreNLP models missing: %s", missing)
    if py_vncorenlp is None:
        logger.error("py_vncorenlp package is not installed.")
        return False

    try:
        logger.info("Attempting automatic download of VnCoreNLP models to %s ...", vncorenlp_dir)
        py_vncorenlp.download_model(save_dir=str(vncorenlp_dir))
    except Exception:
        logger.exception("Automatic download failed.")
        return False

    missing_after = [f for f in required_files if not (vncorenlp_dir / f).exists()]
    if missing_after:
        logger.error("Models still missing after download attempt: %s", missing_after)
        return False

    logger.info("VnCoreNLP download completed and models present.")
    return True


def build_segmenter(vncorenlp_dir: Path):
    """
    Instantiate the VnCoreNLP segmenter (singleton pattern).
    Uses a global instance to avoid JVM conflicts.
    """
    global _SEGMENTER_INSTANCE
    
    if _SEGMENTER_INSTANCE is not None:
        logger.info("Reusing existing VnCoreNLP segmenter instance.")
        return _SEGMENTER_INSTANCE
    
    if py_vncorenlp is None:
        raise ImportError("py_vncorenlp is required. Install with: pip install py_vncorenlp")
    
    # Convert to absolute path - THIS IS CRITICAL
    vncorenlp_dir = Path(vncorenlp_dir).resolve()
    
    # Verify the JAR file exists
    jar_path = vncorenlp_dir / "VnCoreNLP-1.2.jar"
    if not jar_path.exists():
        raise FileNotFoundError(
            f"VnCoreNLP JAR not found at {jar_path}. "
            f"Please ensure VnCoreNLP models are properly installed in {vncorenlp_dir}"
        )
    
    logger.info(f"Creating new VnCoreNLP segmenter instance with directory: {vncorenlp_dir}")
    
    try:
        _SEGMENTER_INSTANCE = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], 
            save_dir=str(vncorenlp_dir)
        )
        logger.info("VnCoreNLP segmenter created successfully.")
    except Exception as e:
        logger.error(f"Failed to create VnCoreNLP segmenter: {e}")
        logger.error(f"VnCoreNLP directory: {vncorenlp_dir}")
        logger.error(f"JAR exists: {jar_path.exists()}")
        raise
    
    return _SEGMENTER_INSTANCE


def segment_reviews(df: pd.DataFrame, vncorenlp_dir: Path, segmenter=None) -> pd.DataFrame:
    """
    Segment `preprocessed_review` column and store result in `segmented_comment`.
    Expects df to already contain 'preprocessed_review' column.
    
    Args:
        df: DataFrame with 'preprocessed_review' column
        vncorenlp_dir: Path to VnCoreNLP models
        segmenter: Optional existing segmenter instance (to avoid recreating)
    """
    if "preprocessed_review" not in df.columns:
        raise ValueError("DataFrame must contain 'preprocessed_review' column before segmentation.")

    vncorenlp_dir = Path(vncorenlp_dir)
    ok = ensure_vncorenlp_models(vncorenlp_dir)
    if not ok:
        raise RuntimeError("VnCoreNLP models are not available. See logs for instructions.")

    # Reuse existing segmenter or create new one
    if segmenter is None:
        segmenter = build_segmenter(vncorenlp_dir)

    segmented_reviews = []
    for review in tqdm(df["preprocessed_review"].tolist(), desc="Segmenting reviews"):
        try:
            seg = segment_review(segmenter, review)
        except Exception:
            logger.exception("Failed to segment a review; using original preprocessed text.")
            seg = review or ""
        segmented_reviews.append(seg)

    df = df.copy()
    df["segmented_comment"] = segmented_reviews
    return df


def preprocessing(df: pd.DataFrame,
                  mapper_path: Path = Path("src/mapping.json"),
                  vncorenlp_dir: Path = Path("notebooks/vncorenlp"),
                  save_csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Full preprocessing:
      - load typo mapping
      - fix token-level typos in `comment` -> new column `preprocessed_review`
      - segment reviews using VnCoreNLP -> new column `segmented_comment`
    Returns processed DataFrame.
    """
    df = df.copy()
    mapper = load_typo_mapper(mapper_path)

    # Build preprocessed_review by tokenizing and mapping typos
    preprocessed_list = []
    for comment in tqdm(df["comment"].tolist(), desc="Mapping typos"):
        tokens = tokenize_words(comment)
        mapped = [mapping_typo_token(t, mapper) for t in tokens]
        preprocessed_list.append(" ".join(mapped))
    df["preprocessed_review"] = preprocessed_list

    # Segment
    df = segment_reviews(df, vncorenlp_dir=vncorenlp_dir)

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        logger.info("Saved processed DataFrame to %s", save_csv_path)

    return df


def build_corpus(df: pd.DataFrame, text_col: str = "segmented_comment") -> Dict[str, int]:
    """
    Build frequency dictionary (word -> count) from df[text_col].
    Accepts both string entries or list-like (token lists).
    """
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col} not present in DataFrame.")

    corpus: Dict[str, int] = {}
    for value in tqdm(df[text_col].tolist(), desc=f"Building corpus from {text_col}"):
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            tokens = [str(x).lower() for x in value]
        else:
            tokens = tokenize_words(str(value))
        for w in tokens:
            if not w:
                continue
            corpus[w] = corpus.get(w, 0) + 1
    return corpus


def split_train_dev(df: pd.DataFrame,
                    train_ratio: float = 0.8,
                    random_state: int = 42,
                    stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and dev sets.

    Args:
        df: Input DataFrame
        train_ratio: Ratio of training data (default: 0.8 for 80/20 split)
        random_state: Random seed for reproducibility
        stratify_col: Column name for stratified splitting (e.g., 'label')

    Returns:
        Tuple of (train_df, dev_df)
    """
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    train_df, dev_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify,
        shuffle=True
    )

    logger.info(f"Split dataset: train={len(train_df)}, dev={len(dev_df)}")
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


# ---- main usage example ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="input csv path (must contain 'comment' column)")
    parser.add_argument("--mapping", default="mapping.json", help="typo mapping json")
    parser.add_argument("--vncorenlp_dir", default="notebooks/vncorenlp",
                        help="vncorenlp models directory (will be converted to absolute path)")
    parser.add_argument("--out_dir", default="dataset/data",
                        help="output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train split ratio (default: 0.8)")
    parser.add_argument("--stratify_col", default="label", help="column name for stratified split (e.g., 'label')")
    parser.add_argument("--random_state", type=int, default=42, help="random seed")
    parser.add_argument("--balance_method", choices=["back_translation", "random", "none"],
                       default="none", help="method to balance dataset")
    parser.add_argument("--augmentation_factor", type=float, default=2.0,
                       help="augmentation factor for back-translation")
    args = parser.parse_args()

    # Convert paths to absolute paths
    vncorenlp_dir = Path(args.vncorenlp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using VnCoreNLP directory: {vncorenlp_dir}")
    logger.info(f"Output directory: {out_dir}")

    # Load and preprocess full dataset
    logger.info("Loading and preprocessing data...")
    df_in = pd.read_csv(args.csv)
    processed = preprocessing(
        df_in,
        mapper_path=Path(args.mapping).resolve(),
        vncorenlp_dir=vncorenlp_dir,
        save_csv_path=out_dir / "processed_full.csv"
    )

    # Build corpus BEFORE splitting (important for vocabulary consistency)
    logger.info("Building corpus from full dataset...")
    corpus = build_corpus(processed, text_col="segmented_comment")
    corpus_path = out_dir / "corpus_freq.json"
    with open(corpus_path, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    logger.info(f"Corpus saved to {corpus_path} with {len(corpus)} unique tokens")

    # Split into train and dev
    logger.info("Splitting dataset into train and dev...")
    train_df, dev_df = split_train_dev(
        processed,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        stratify_col=args.stratify_col
    )

    # Apply balancing method if specified
    if args.balance_method != "none":
        # Save initial train split
        initial_train_path = out_dir / "train_before_balance.csv"
        train_df.to_csv(initial_train_path, index=False, encoding="utf-8-sig")
        logger.info(f"Initial train set saved to {initial_train_path}")
        
        logger.info(f"Applying {args.balance_method} to balance the training set...")
        
        if args.balance_method == "back_translation":
            
            # Back-translation needs to be done on the CSV file
            temp_train_path = out_dir / "temp_train_unbalanced.csv"
            train_df.to_csv(temp_train_path, index=False, encoding="utf-8-sig")
            
            # Apply back-translation (it will handle preprocessing and segmentation)
            balanced_train_df = apply_back_translation(
                train_csv=temp_train_path,
                text_col="segmented_comment",
                label_col="label",
                output_csv=None,  # Don't save yet
                augmentation_factor=args.augmentation_factor,
                random_state=args.random_state
            )
            
            # Clean up temp file
            if temp_train_path.exists():
                temp_train_path.unlink()
            
            train_df = balanced_train_df
            
        elif args.balance_method == "random":
            from imbalance_handler import apply_random_oversampling
            
            temp_train_path = out_dir / "temp_train_unbalanced.csv"
            train_df.to_csv(temp_train_path, index=False, encoding="utf-8-sig")
            
            balanced_train_df = apply_random_oversampling(
                train_csv=temp_train_path,
                label_col="label",
                output_csv=None,
                sampling_strategy="auto",
                random_state=args.random_state
            )
            
            # Clean up temp file
            if temp_train_path.exists():
                temp_train_path.unlink()
            
            train_df = balanced_train_df

    # Save final splits
    train_path = out_dir / "train.csv"
    dev_path = out_dir / "dev.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    dev_df.to_csv(dev_path, index=False, encoding="utf-8-sig")

    logger.info(f"Train set saved to {train_path} ({len(train_df)} samples)")
    logger.info(f"Dev set saved to {dev_path} ({len(dev_df)} samples)")
    
    # Print class distribution
    if 'label' in train_df.columns:
        train_dist = train_df['label'].value_counts().sort_index()
        logger.info(f"Final training set distribution:\n{train_dist}")
    
    logger.info("Done!")