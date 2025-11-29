import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


class Baseline:
    """
    Baseline classifier supporting Logistic Regression and SVM models.
    Uses TF-IDF or Count vectorization for text features.
    """

    def __init__(
            self,
            model_type='logistic',
            vectorizer='tfidf',
            max_features=5000,
            ngram_range=(1, 2),
            C=1.0,
            max_iter=1000,
            random_state=42,
            **kwargs
    ):
        """
        Initialize baseline classifier.

        Args:
            model_type (str): 'logistic' or 'svm' (or 'linear_svm')
            vectorizer (str): 'tfidf' or 'count'
            max_features (int): Maximum number of features for vectorizer
            ngram_range (tuple): Range of n-grams (default: unigram + bigram)
            C (float): Regularization parameter
            max_iter (int): Maximum iterations for training
            random_state (int): Random seed
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type.lower()
        self.vectorizer_type = vectorizer.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs

        # Initialize vectorizer
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                strip_accents='unicode'
            )
        elif self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer}. Use 'tfidf' or 'count'")

        # Initialize model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=C,
                kernel='rbf',
                max_iter=max_iter,
                random_state=random_state,
                probability=True,  # Enable probability estimates
                **kwargs
            )
        elif self.model_type == 'linear_svm':
            self.model = LinearSVC(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                dual='auto',
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'logistic', 'svm', or 'linear_svm'")

        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])

        self.is_fitted = False

    def fit(self, X, y):
        """
        Train the baseline model.

        Args:
            X (array-like): Text data (list of strings or pandas Series)
            y (array-like): Labels

        Returns:
            self
        """
        print(f"Training {self.model_type.upper()} with {self.vectorizer_type.upper()} vectorization...")
        print(f"Features: {self.max_features}, N-grams: {self.ngram_range}, C: {self.C}")

        self.pipeline.fit(X, y)
        self.is_fitted = True

        print("Training completed!")
        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X (array-like): Text data

        Returns:
            array: Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Text data

        Returns:
            array: Predicted probabilities for each class
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # LinearSVC doesn't support predict_proba
        if self.model_type == 'linear_svm':
            # Use decision function as a proxy
            decision = self.pipeline.decision_function(X)
            # Convert to pseudo-probabilities using sigmoid
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])

        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y, average='binary'):
        """
        Evaluate model on test data.

        Args:
            X (array-like): Text data
            y (array-like): True labels
            average (str): Averaging method for metrics

        Returns:
            dict: Dictionary of metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1': f1_score(y, y_pred, average=average, zero_division=0)
        }

        return metrics

    def save(self, save_path):
        """
        Save the trained model.

        Args:
            save_path (str): Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the entire object state, not just the pipeline
        save_dict = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'C': self.C,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'kwargs': self.kwargs
        }

        joblib.dump(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        """
        Load a trained model.

        Args:
            load_path (str): Path to the saved model
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        loaded_data = joblib.load(load_path)

        # Handle both old format (just pipeline) and new format (dict)
        if isinstance(loaded_data, dict):
            # New format with all attributes
            self.pipeline = loaded_data['pipeline']
            self.model_type = loaded_data.get('model_type', 'logistic')
            self.vectorizer_type = loaded_data.get('vectorizer_type', 'tfidf')
            self.max_features = loaded_data.get('max_features', 5000)
            self.ngram_range = loaded_data.get('ngram_range', (1, 2))
            self.C = loaded_data.get('C', 1.0)
            self.max_iter = loaded_data.get('max_iter', 1000)
            self.random_state = loaded_data.get('random_state', 42)
            self.kwargs = loaded_data.get('kwargs', {})
        else:
            # Old format (just pipeline) - for backward compatibility
            self.pipeline = loaded_data
            print("Warning: Loaded old format model. Some attributes may not be available.")

        # CRITICAL FIX: Extract vectorizer and model from pipeline
        self.vectorizer = self.pipeline.named_steps['vectorizer']
        self.model = self.pipeline.named_steps['classifier']

        self.is_fitted = True
        print(f"Model loaded from {load_path}")

        # Verify vectorizer is properly fitted
        if hasattr(self.vectorizer, 'vocabulary_'):
            print(f"✓ Vectorizer vocabulary loaded: {len(self.vectorizer.vocabulary_)} features")
        else:
            print("⚠ Warning: Vectorizer vocabulary not found")

    def get_feature_importance(self, top_n=20):
        """
        Get most important features (works for Logistic Regression and Linear SVM).

        Args:
            top_n (int): Number of top features to return

        Returns:
            dict: Top positive and negative features, or None if not available
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Check if vectorizer has vocabulary
        if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
            print("Error: Vectorizer vocabulary not available. Cannot extract feature importance.")
            return None

        # Check if model has coef_ attribute (only linear models have this)
        if not hasattr(self.model, 'coef_'):
            print(f"Feature importance not available for {self.model_type} with non-linear kernel.")
            print("Feature importance only works for Logistic Regression and Linear SVM.")
            return None

        # Additional check for model type
        if self.model_type not in ['logistic', 'linear_svm']:
            print(f"Feature importance not available for model type: {self.model_type}")
            return None

        try:
            # Get feature names and coefficients
            feature_names = self.vectorizer.get_feature_names_out()
            coef = self.model.coef_[0]

            # Get top positive and negative features
            top_positive_idx = np.argsort(coef)[-top_n:][::-1]
            top_negative_idx = np.argsort(coef)[:top_n]

            top_positive = [(feature_names[i], coef[i]) for i in top_positive_idx]
            top_negative = [(feature_names[i], coef[i]) for i in top_negative_idx]

            return {
                'top_positive': top_positive,
                'top_negative': top_negative
            }
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None

    def __repr__(self):
        return (f"Baseline(model_type='{self.model_type}', "
                f"vectorizer='{self.vectorizer_type}', "
                f"max_features={self.max_features}, "
                f"ngram_range={self.ngram_range}, "
                f"C={self.C})")