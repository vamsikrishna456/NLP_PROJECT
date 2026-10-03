"""
============================================
Classical ML Models Module
============================================
Implements: Logistic Regression, Naive Bayes, SVM
With hyperparameter tuning and cross-validation support.
"""

import logging
import numpy as np
import joblib
from typing import Dict, Any, Optional
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSICAL_MODEL_CONFIG, MODEL_DIR, EVALUATION_CONFIG


class ClassicalModelTrainer:
    """Trains and manages classical ML models for sentiment analysis."""

    def __init__(self):
        self.models = {}
        self.best_params = {}

    def train_logistic_regression(self, X_train, y_train, tune=False):
        """Train Logistic Regression with optional hyperparameter tuning."""
        logger.info("Training Logistic Regression...")
        config = CLASSICAL_MODEL_CONFIG["logistic_regression"]

        if tune:
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [1000],
            }
            model = LogisticRegression(class_weight="balanced", random_state=42)
            grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
            grid.fit(X_train, y_train)
            self.models["logistic_regression"] = grid.best_estimator_
            self.best_params["logistic_regression"] = grid.best_params_
            logger.info(f"Best LR params: {grid.best_params_}")
        else:
            model = LogisticRegression(**config, random_state=42)
            model.fit(X_train, y_train)
            self.models["logistic_regression"] = model

        logger.info("Logistic Regression trained ✅")
        return self.models["logistic_regression"]

    def train_naive_bayes(self, X_train, y_train, use_gaussian=False):
        """Train Naive Bayes (Multinomial for sparse, Gaussian for dense)."""
        logger.info("Training Naive Bayes...")
        config = CLASSICAL_MODEL_CONFIG["naive_bayes"]

        if use_gaussian:
            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()
            model = GaussianNB()
        else:
            model = MultinomialNB(alpha=config["alpha"])

        model.fit(X_train, y_train)
        self.models["naive_bayes"] = model
        logger.info("Naive Bayes trained ✅")
        return model

    def train_svm(self, X_train, y_train, tune=False):
        """Train SVM with probability calibration for confidence scores."""
        logger.info("Training SVM...")
        config = CLASSICAL_MODEL_CONFIG["svm"]

        if tune:
            param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
            base = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
            grid = GridSearchCV(base, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
            grid.fit(X_train, y_train)
            # Calibrate for probability estimates
            model = CalibratedClassifierCV(grid.best_estimator_, cv=3)
            model.fit(X_train, y_train)
            self.best_params["svm"] = grid.best_params_
            logger.info(f"Best SVM params: {grid.best_params_}")
        else:
            base = LinearSVC(**config, random_state=42)
            model = CalibratedClassifierCV(base, cv=3)
            model.fit(X_train, y_train)

        self.models["svm"] = model
        logger.info("SVM trained ✅")
        return model

    def train_all(self, X_train, y_train, tune=False, use_gaussian=False):
        """Train all classical models."""
        logger.info("=" * 50)
        logger.info("Training all classical models...")
        logger.info("=" * 50)
        self.train_logistic_regression(X_train, y_train, tune=tune)
        self.train_naive_bayes(X_train, y_train, use_gaussian=use_gaussian)
        self.train_svm(X_train, y_train, tune=tune)
        logger.info("All classical models trained! ✅")
        return self.models

    def predict(self, model_name, X):
        """Get predictions from a specific model."""
        model = self.models[model_name]
        if hasattr(X, "toarray") and isinstance(model, GaussianNB):
            X = X.toarray()
        return model.predict(X)

    def predict_proba(self, model_name, X):
        """Get prediction probabilities."""
        model = self.models[model_name]
        if hasattr(X, "toarray") and isinstance(model, GaussianNB):
            X = X.toarray()
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def cross_validate(self, model_name, X, y, cv=None):
        """Perform cross-validation for a specific model."""
        cv = cv or EVALUATION_CONFIG["cv_folds"]
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        logger.info(f"{model_name} CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return {"mean": scores.mean(), "std": scores.std(), "scores": scores}

    def save_models(self):
        """Save all trained models."""
        for name, model in self.models.items():
            path = MODEL_DIR / f"{name}.pkl"
            joblib.dump(model, path)
            logger.info(f"Saved {name} to {path}")

    def load_model(self, model_name):
        """Load a specific model."""
        path = MODEL_DIR / f"{model_name}.pkl"
        self.models[model_name] = joblib.load(path)
        logger.info(f"Loaded {model_name} from {path}")
        return self.models[model_name]


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=100, n_classes=3,
                                n_informative=50, random_state=42)
    trainer = ClassicalModelTrainer()
    trainer.train_all(X, y, use_gaussian=True)
    for name in trainer.models:
        preds = trainer.predict(name, X[:5])
        print(f"{name}: {preds}")
    print("✅ Classical models working!")
