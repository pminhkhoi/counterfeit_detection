# preprocessing.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable, List, Union, Any
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import py_vncorenlp
from transformers import AutoTokenizer

from .imbalance_handler import apply_back_translation, apply_random_oversampling
from dataset.dataset import CSVDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")


class Preprocessor:

    def __init__(self, mapper_path: Optional[str] = None, vncorenlp_dir: Optional[str] = None) -> None:
        """
        Initialize Preprocessor.

        Args:
            train: Optional training CSVDataset
            dev: Optional dev CSVDataset
            test: Optional test CSVDataset
            mapper_path: Path to typo mapping JSON file (required for preprocessing operations)
        """
        self.mapper: Optional[Dict[str, str]] = None
        if mapper_path:
            self.mapper = self.load_typo_mapper(mapper_path)

        self.vncorenlp_dir: Optional[Path] = Path(vncorenlp_dir).resolve() if vncorenlp_dir else None
        self.has_vncorenlp: bool = False
        if self.vncorenlp_dir:
            self.has_vncorenlp = self.ensure_vncorenlp_models(self.vncorenlp_dir)

        self._SEGMENTER_INSTANCE = None

    @staticmethod
    def split_train_test(
        dataset: Union["CSVDataset", pd.DataFrame],
        ratios: List[float],
        stratify_col: Optional[str] = None,
        random_state: int = 42
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Split dataset into train/dev/test according to ratios (train, dev, test).
        Always returns a 3-tuple (train_df, dev_df, test_df). If a split has ratio 0,
        the corresponding tuple element is None.

        Args:
            dataset: CSVDataset or pandas DataFrame
            ratios: list/tuple of three floats summing to 1.0 (train, dev, test)
            stratify_col: optional column name used for stratified splitting
            random_state: seed for reproducibility

        Returns:
            (train_df, dev_df, test_df) where elements may be None if their ratio is 0.
        """
        if len(ratios) != 3:
            raise ValueError("Ratios must have length of 3 for train-dev-test.")
        if any(r < 0 for r in ratios):
            raise ValueError("Ratios must be non-negative.")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("The total ratios of train-dev-test must sum to 1.")

        train_ratio, dev_ratio, test_ratio = ratios

        # get base dataframe
        if hasattr(dataset, "data"):
            base_df = dataset.data.copy().reset_index(drop=True)
        elif isinstance(dataset, pd.DataFrame):
            base_df = dataset.copy().reset_index(drop=True)
        else:
            raise TypeError("dataset must be a CSVDataset-like object with .data or a pandas.DataFrame")

        # quick returns for degenerate cases
        if base_df.empty:
            logger.warning("Input dataframe is empty; returning (None, None, None).")
            return None, None, None

        # handle stratify column presence
        stratify_series = None
        if stratify_col:
            if stratify_col not in base_df.columns:
                raise ValueError(f"Stratify column '{stratify_col}' not found in dataset.")
            stratify_series = base_df[stratify_col]

        # If train_ratio == 1.0 just return all as train
        if train_ratio == 1.0:
            return base_df, None, None
        # If dev_ratio == 1.0 or test_ratio == 1.0 (others zero)
        if dev_ratio == 1.0:
            return None, base_df, None
        if test_ratio == 1.0:
            return None, None, base_df

        # Helper to decide whether stratify argument is valid for a given series
        def valid_stratify(s: Optional[pd.Series]) -> Optional[pd.Series]:
            if s is None:
                return None
            # drop NA for stratify decision
            uniq = s.dropna().unique()
            if len(uniq) <= 1:
                # cannot stratify when only one class present
                return None
            return s

        # First split: train vs remaining
        stratify_for_first = valid_stratify(stratify_series)
        train_df, remaining_df = train_test_split(
            base_df,
            train_size=train_ratio,
            stratify=stratify_for_first,
            random_state=random_state,
            shuffle=True
        )

        # if dev and test both zero
        if dev_ratio == 0 and test_ratio == 0:
            return train_df, None, None

        remaining_total = dev_ratio + test_ratio
        if remaining_total == 0 or remaining_df.empty:
            return train_df, None, None

        # If one of dev/test is zero, remaining goes to the other
        if dev_ratio == 0:
            return train_df, None, remaining_df.reset_index(drop=True)
        if test_ratio == 0:
            return train_df, remaining_df.reset_index(drop=True), None

        # both dev and test > 0 -> split remaining
        dev_rel_ratio = dev_ratio / remaining_total
        stratify_remaining = None
        if stratify_col and stratify_col in remaining_df.columns:
            stratify_remaining = valid_stratify(remaining_df[stratify_col])

        dev_df, test_df = train_test_split(
            remaining_df,
            train_size=dev_rel_ratio,
            stratify=stratify_remaining,
            random_state=random_state,
            shuffle=True
        )

        # reset indices before returning
        return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @staticmethod
    def save_dataset(df: pd.DataFrame, out_path: Union[str, Path]) -> None:
        """Save DataFrame to CSV file."""
        if df is None or df.empty:
            raise ValueError("Dataset cannot be empty.")

        if isinstance(out_path, str):
            out_path = Path(out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved dataset at {out_path}")

    def load_typo_mapper(self, mapper_path: Union[str, Path]) -> Dict[str, str]:
        """Load JSON mapping file from disk."""
        try:
            with open(mapper_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Mapper JSON must be an object/dict.")
            return data
        except Exception as e:
            logger.exception("Failed to load mapper JSON")
            raise

    @staticmethod
    def tokenize_words(review: str) -> list:
        """Simple whitespace tokenizer (lowercases first)."""
        if review is None:
            return []
        return str(review).lower().strip().split()

    def mapping_typo_token(self, token: str) -> str:
        """Map a single token using the mapper; return token if no mapping exists."""
        if not self.mapper:
            raise ValueError("Mapper has not been initialized yet.")
        return self.mapper.get(token, token)

    @staticmethod
    def segment_review(segmenter: Any, text: Optional[str]) -> str:
        """
        Wrap the actual call to the segmenter. This default assumes a simple API:
            segmenter.word_segment(text)
        Override this method if your segmenter requires different invocation.
        """
        if text is None:
            return ""
        return segmenter.word_segment(text)[0]


    def ensure_vncorenlp_models(self, vncorenlp_dir: Path) -> bool:
        """
        Check that VnCoreNLP models are present at vncorenlp_dir.
        This is intentionally conservative: return True only if directory exists.
        Replace/enhance with real checks if you know required files.
        """
        try:
            if not vncorenlp_dir.exists():
                logger.warning("vncorenlp_dir does not exist: %s", vncorenlp_dir)
                return False
            return True
        except Exception:
            logger.exception("Error while checking VnCoreNLP models directory")
            return False

    def build_segmenter(self) -> Any:
        """
        Lazy-create and return a singleton segmenter instance.
        Raises ImportError if py_vncorenlp is unavailable.
        """
        if self._SEGMENTER_INSTANCE is not None:
            logger.debug("Reusing existing segmenter instance.")
            return self._SEGMENTER_INSTANCE

        if py_vncorenlp is None:
            raise ImportError("py_vncorenlp is not installed. Install with: pip install py_vncorenlp")

        if not self.vncorenlp_dir or not self.has_vncorenlp:
            raise RuntimeError("VnCoreNLP models are not available. Set vncorenlp_dir and ensure models exist.")

        try:
            # instantiate according to the py_vncorenlp API
            self._SEGMENTER_INSTANCE = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(self.vncorenlp_dir))
            logger.info("VnCoreNLP segmenter created.")
            return self._SEGMENTER_INSTANCE
        except Exception:
            logger.exception("Failed to create VnCoreNLP segmenter")
            raise

    def segment_reviews(
        self,
        df: pd.DataFrame,
        segmenter: Optional[Any] = None,
        text_col: str = "preprocessed_review",
        output_col: str = "segmented_comment",
    ) -> pd.DataFrame:
        """
        Segment `text_col` in `df` and return a copy with `output_col` added.

        - Raises ValueError if `text_col` is missing or df is empty.
        - Raises RuntimeError if VnCoreNLP models are not available.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas.DataFrame")

        if text_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{text_col}' column before segmentation.")

        if df.empty:
            raise ValueError("DataFrame is empty.")

        if not self.has_vncorenlp:
            raise RuntimeError("VnCoreNLP models are not available. Set vncorenlp_dir and ensure models exist.")

        # allow caller to pass a pre-built segmenter (useful for tests)
        if segmenter is None:
            segmenter = self.build_segmenter()

        segmented = []
        # iterate with tqdm for user feedback
        for text in tqdm(df[text_col].astype(str).tolist(), desc=f"Segmenting {text_col}"):
            try:
                seg = self.segment_review(segmenter, text)
            except Exception:
                logger.exception("Segmentation failed for a review; using original text.")
                seg = text or ""
            segmented.append(seg)

        out_df = df.copy()
        out_df[output_col] = segmented
        return out_df

    def preprocess_text(self,
                        df: pd.DataFrame,
                        input_col: str = "comment",
                        output_col: str = "preprocessed_review") -> pd.DataFrame:
        """
        Apply typo mapping to text column.

        Args:
            df: DataFrame containing the input column
            input_col: Name of input column with raw text
            output_col: Name of output column for preprocessed text

        Returns:
            DataFrame with new preprocessed column
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Initialize with mapper_path or call load_typo_mapper().")

        if input_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{input_col}' column.")

        preprocessed_list = []
        for comment in tqdm(df[input_col].tolist(), desc="Mapping typos"):
            tokens = self.tokenize_words(comment)
            mapped = [self.mapping_typo_token(t) for t in tokens]
            preprocessed_list.append(" ".join(mapped))

        out_df = df.copy(deep=True)
        out_df[output_col] = preprocessed_list
        return out_df

    def preprocessing_pipeline(self,
                               dataset: pd.DataFrame,
                               input_col: str = "comment",
                               intermediate_col: str = "preprocessed_review",
                               output_col: str = "segmented_comment") -> pd.DataFrame:
        """
        Full preprocessing pipeline:
        1. Load typo mapping
        2. Fix token-level typos: input_col -> intermediate_col
        3. Segment reviews: intermediate_col -> output_col

        Args:
            mapper_path: Path to typo mapping JSON file
            vncorenlp_dir: Path to VnCoreNLP models directory
            input_col: Name of original text column
            intermediate_col: Name of intermediate preprocessed column
            output_col: Name of final segmented column

        Returns:
            Processed DataFrame with new columns
        """
        if dataset is None:
            raise ValueError("No training dataset available.")

        # Load mapper if provided or ensure it's already loaded
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Provide mapper_path or initialize Preprocessor with mapper_path.")

        logger.info("Starting preprocessing pipeline...")

        # Step 1: Typo mapping
        logger.info(f"Step 1: Mapping typos from '{input_col}' -> '{intermediate_col}'")
        out_df = self.preprocess_text(
            dataset,
            input_col=input_col,
            output_col=intermediate_col
        )

        # Step 2: Segmentation
        logger.info(f"Step 2: Segmenting text from '{intermediate_col}' -> '{output_col}'")
        out_df = self.segment_reviews(
            out_df,
            text_col=intermediate_col,
            output_col=output_col
        )

        logger.info("Preprocessing pipeline completed successfully.")
        return out_df

    def apply_balancing(self,
                        df: pd.DataFrame,
                        method: str = "none",
                        text_col: str = "preprocessed_review",
                        label_col: str = "label",
                        augmentation_factor: float = 2.0,
                        random_state: int = 42,
                        device: str = "cuda") -> pd.DataFrame:
        """
        Apply data balancing/augmentation to handle imbalanced classes.

        Args:
            df: DataFrame to balance
            method: Balancing method ("none", "back_translation", "random_oversampling")
            text_col: Column containing text data
            label_col: Column containing labels
            augmentation_factor: Factor for augmentation (e.g., 2.0 for doubling minority class)
            random_state: Random seed
            device: Device for back-translation ("cuda" or "cpu")

        Returns:
            Balanced DataFrame
        """
        if method == "none":
            logger.info("No balancing applied.")
            return df

        elif method == "back_translation":
            logger.info("Applying back-translation augmentation...")
            balanced_df = apply_back_translation(
                train_csv=df,
                text_col=text_col,
                label_col=label_col,
                output_csv=None,
                augmentation_factor=augmentation_factor,
                random_state=random_state,
                device=device
            )
            return balanced_df

        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def build_corpus(self,
                     df: pd.DataFrame,
                     data_col: str = "segmented_comment",
                     save: bool = False,
                     save_path: Union[str, Path] = "../dataset/data/") -> Dict[str, int]:
        """
        Build frequency dictionary (word -> count) from df[text_col].
        Accepts both string entries or list-like (token lists).
        """
        if data_col not in df.columns:
            raise ValueError(f"Column {data_col} not present in DataFrame.")

        corpus: Dict[str, int] = {}
        for value in tqdm(df[data_col].tolist(), desc=f"Building corpus from {data_col}"):
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                tokens = [str(x).lower() for x in value]
            else:
                tokens = self.tokenize_words(str(value))
            for w in tokens:
                if not w:
                    continue
                corpus[w] = corpus.get(w, 0) + 1

        if save:
            save_path = Path(save_path) / 'corpus.json'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False, indent=2)
            logger.info(f"Corpus saved to {save_path} with {len(corpus)} unique tokens")

        return corpus


# ---- Main usage example ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data preprocessing and splitting pipeline")
    parser.add_argument("--csv", required=True, help="input csv path (must contain 'comment' column)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="train split ratio")
    parser.add_argument("--dev_ratio", type=float, default=0.15, help="dev split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="test split ratio")
    parser.add_argument("--data_col", default="comment", help="column name for the reviews")
    parser.add_argument("--label_col", default="label", help="column name for the labels")
    parser.add_argument("--stratify_col", default="label", help="column name for stratified split")
    parser.add_argument("--mapper_path", required=True, help="typo mapping json")
    parser.add_argument("--vncorenlp_dir", default="../notebooks/vncorenlp", help="vncorenlp models directory")
    parser.add_argument("--out_dir", default="dataset/data", help="output directory")
    parser.add_argument("--random_state", type=int, default=50, help="random seed")
    parser.add_argument("--model", type=str, default="vinai/phobert-base", help="model name")
    parser.add_argument("--balance_method", choices=["back_translation", "none"],
                        default="none", help="method to balance dataset")
    parser.add_argument("--augmentation_factor", type=float, default=2.0,
                        help="augmentation factor for back-translation")
    parser.add_argument("--device", default="cuda", help="device for augmentation (cuda/cpu)")

    args = parser.parse_args()

    # Convert paths to absolute paths
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Load dataset
    logger.info("Loading data...")
    data = pd.read_csv(args.csv)

    # Initialize preprocessor
    processor = Preprocessor(mapper_path=args.mapper_path, vncorenlp_dir=args.vncorenlp_dir)

    # ============================================================
    # STEP 1: Full preprocessing pipeline (typo mapping + segmentation)
    # ============================================================
    logger.info("Running preprocessing pipeline...")
    preprocessed_df = processor.preprocessing_pipeline(
        dataset=data,
        input_col=args.data_col,
        intermediate_col="preprocessed_review",
        output_col="segmented_comment"
    )

    # Build corpus BEFORE splitting (important for vocabulary consistency)
    logger.info("Building corpus from full dataset...")
    corpus = processor.build_corpus(
        preprocessed_df,
        data_col="segmented_comment",
        save=True,
        save_path=out_dir
    )

    # ============================================================
    # STEP 2: Split into train/dev/test BEFORE balancing
    # ============================================================
    logger.info("Splitting dataset into train/dev/test...")
    ratios = [args.train_ratio, args.dev_ratio, args.test_ratio]
    train_df, dev_df, test_df = processor.split_train_test(preprocessed_df, ratios, args.stratify_col, args.random_state)

    # ============================================================
    # STEP 3: Apply balancing ONLY to training data
    # ============================================================

    if args.balance_method != "none" and train_df is not None:
        logger.info(f"Applying {args.balance_method} balancing to training data...")
        balanced_train_df = processor.apply_balancing(
            train_df,
            method=args.balance_method,
            text_col="preprocessed_review",
            label_col=args.label_col,
            augmentation_factor=args.augmentation_factor,
            random_state=args.random_state,
            device=args.device
        )

        # Re-segment the balanced training data
        logger.info("Segmenting balanced training data...")
        balanced_train_df = processor.segment_reviews(
            balanced_train_df,
            text_col="preprocessed_review",
            output_col="segmented_comment"
        )

        train_df = balanced_train_df
        logger.info(f"Balanced training dataset: {len(balanced_train_df)} samples")


    # ============================================================
    # STEP 4: Save splits
    # ============================================================
    logger.info("Saving datasets...")
    if train_df is not None:
        train_path = out_dir / "train.csv"
        processor.save_dataset(train_df, train_path)
        logger.info(f"Train set saved: {len(train_df)} samples")
        if args.label_col in train_df.columns:
            logger.info(f"Train distribution:\n{train_df[args.label_col].value_counts().sort_index()}")

    if dev_df is not None:
        dev_path = out_dir / "dev.csv"
        processor.save_dataset(dev_df, dev_path)
        logger.info(f"Dev set saved: {len(dev_df)} samples")
        if args.label_col in dev_df.columns:
            logger.info(f"Dev distribution:\n{dev_df[args.label_col].value_counts().sort_index()}")

    if test_df is not None:
        test_path = out_dir / "test.csv"
        processor.save_dataset(test_df, test_path)
        logger.info(f"Test set saved: {len(test_df)} samples")
        if args.label_col in test_df.columns:
            logger.info(f"Test distribution:\n{test_df[args.label_col].value_counts().sort_index()}")

    logger.info("Pipeline completed successfully!")