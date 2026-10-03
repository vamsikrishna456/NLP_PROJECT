"""
============================================
Evaluation Module
============================================
Comprehensive model evaluation: accuracy, precision, recall, F1,
confusion matrix, cross-validation, and model comparison.
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EVALUATION_CONFIG, SENTIMENT_LABELS, LOG_DIR


class ModelEvaluator:
    """Evaluates and compares sentiment analysis models."""

    def __init__(self, label_names: Dict = None):
        self.label_names = label_names or SENTIMENT_LABELS
        self.results = {}

    def evaluate(self, y_true, y_pred, model_name: str,
                 y_proba=None) -> Dict:
        """
        Comprehensive evaluation of a model's predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary with all metrics
        """
        average = EVALUATION_CONFIG["average"]

        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }

        # Per-class metrics
        target_names = [self.label_names.get(i, str(i)) for i in sorted(set(y_true))]
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        metrics["classification_report"] = report

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # AUC-ROC (if probabilities available)
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                metrics["auc_roc"] = float(auc)
            except Exception:
                metrics["auc_roc"] = None

        self.results[model_name] = metrics

        logger.info(f"\n{'='*50}")
        logger.info(f"📊 {model_name} Evaluation Results")
        logger.info(f"{'='*50}")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics.get("auc_roc"):
            logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")

        return metrics

    def compare_models(self) -> Dict:
        """Compare all evaluated models and rank them."""
        if not self.results:
            logger.warning("No models evaluated yet!")
            return {}

        comparison = []
        for name, metrics in self.results.items():
            comparison.append({
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics.get("auc_roc", 0),
            })

        # Sort by F1 score
        comparison.sort(key=lambda x: x["f1_score"], reverse=True)

        logger.info("\n" + "=" * 70)
        logger.info("📊 MODEL COMPARISON (sorted by F1-Score)")
        logger.info("=" * 70)
        logger.info(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        logger.info("-" * 70)
        for m in comparison:
            logger.info(f"{m['model']:<25} {m['accuracy']:>10.4f} "
                       f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
        logger.info("-" * 70)

        best = comparison[0]
        logger.info(f"\n🏆 Best Model: {best['model']} (F1: {best['f1_score']:.4f})")

        return {"comparison": comparison, "best_model": best["model"]}

    def save_results(self, filepath: str = None):
        """Save evaluation results to JSON."""
        filepath = filepath or str(LOG_DIR / "evaluation_results.json")

        # Convert numpy types for JSON serialization
        serializable = {}
        for name, metrics in self.results.items():
            serializable[name] = {
                k: v for k, v in metrics.items()
                if k != "classification_report"
            }
            serializable[name]["classification_report"] = str(metrics.get("classification_report", ""))

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """Plot confusion matrix for a model."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return

        if model_name not in self.results:
            logger.error(f"Model {model_name} not found")
            return

        cm = np.array(self.results[model_name]["confusion_matrix"])
        labels = [self.label_names.get(i, str(i)) for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")

        save_path = save_path or str(LOG_DIR / f"cm_{model_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {save_path}")

    def plot_comparison_chart(self, save_path: str = None):
        """Plot comparison bar chart of all models."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return

        if not self.results:
            return

        models = list(self.results.keys())
        metrics_names = ["accuracy", "precision", "recall", "f1_score"]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics_names):
            values = [self.results[m][metric] for m in models]
            ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)

        save_path = save_path or str(LOG_DIR / "model_comparison.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Comparison chart saved to {save_path}")
