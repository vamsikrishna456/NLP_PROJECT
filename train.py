import logging
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from config import (
    DATASET_CONFIG, PREPROCESSING_CONFIG, FEATURE_CONFIG,
    CLASSICAL_MODEL_CONFIG, DEEP_LEARNING_CONFIG, TRANSFORMER_CONFIG,
    MODEL_DIR, SENTIMENT_LABELS,
)


def main():
    """Run the complete training pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("🚀 NLP SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("=" * 70)

    # ============================================
    # STEP 1: Load Dataset
    # ============================================
    print("\n📦 STEP 1: Loading Dataset...")
    from data.data_loader import prepare_dataset

    data = prepare_dataset(
        dataset_name=DATASET_CONFIG.get("name", "synthetic"),
        max_samples=DATASET_CONFIG.get("max_samples", 3000),
        test_size=DATASET_CONFIG.get("test_size", 0.2),
        val_size=DATASET_CONFIG.get("val_size", 0.1),
        random_state=DATASET_CONFIG.get("random_state", 42),
    )

    X_train_raw, y_train = data["train"]
    X_val_raw, y_val = data["val"]
    X_test_raw, y_test = data["test"]

    print(f"  Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    # ============================================
    # STEP 2: Preprocess Text
    # ============================================
    print("\n🔧 STEP 2: Preprocessing...")
    from preprocessing.text_cleaner import TextPreprocessor

    preprocessor = TextPreprocessor()

    X_train_processed = preprocessor.preprocess_batch(X_train_raw)
    X_val_processed = preprocessor.preprocess_batch(X_val_raw)
    X_test_processed = preprocessor.preprocess_batch(X_test_raw)

    # Also create tokenized versions for Word2Vec
    X_train_tokens = [text.split() for text in X_train_processed]
    X_test_tokens = [text.split() for text in X_test_processed]

    # Show lemmatization vs stemming comparison
    print("\n📊 Lemmatization vs Stemming Comparison:")
    sample = "The products were running beautifully and customers loved improvements"
    comparison = preprocessor.compare_lemmatization_vs_stemming(sample)
    print(f"  Original:    {comparison['original_tokens']}")
    print(f"  Lemmatized:  {comparison['lemmatized']}")
    print(f"  Stemmed:     {comparison['stemmed']}")

    # ============================================
    # STEP 3: Feature Engineering
    # ============================================
    print("\n⚙️ STEP 3: Feature Engineering...")
    from features.feature_extractor import (
        BagOfWordsExtractor, TFIDFExtractor, Word2VecExtractor,
    )

    # 3a: Bag of Words
    print("\n  [BoW] Building Bag of Words features...")
    bow = BagOfWordsExtractor()
    X_train_bow = bow.fit_transform(X_train_processed)
    X_test_bow = bow.transform(X_test_processed)
    bow.save()

    # 3b: TF-IDF
    print("  [TF-IDF] Building TF-IDF features...")
    tfidf = TFIDFExtractor()
    X_train_tfidf = tfidf.fit_transform(X_train_processed)
    X_test_tfidf = tfidf.transform(X_test_processed)
    tfidf.save()

    # 3c: Word2Vec
    print("  [Word2Vec] Training Word2Vec embeddings...")
    w2v = Word2VecExtractor()
    X_train_w2v = w2v.fit_transform(X_train_tokens)
    X_test_w2v = w2v.transform(X_test_tokens)
    w2v.save()

    # ============================================
    # STEP 4: Train Classical ML Models
    # ============================================
    print("\n🤖 STEP 4: Training Classical ML Models...")
    from models.classical_models import ClassicalModelTrainer
    from evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    classical_trainer = ClassicalModelTrainer()

    # Train with TF-IDF features (generally best for classical models)
    print("\n  --- Using TF-IDF Features ---")
    classical_trainer.train_all(X_train_tfidf, y_train, tune=True)
    classical_trainer.save_models()

    # Evaluate classical models
    for model_name in ["logistic_regression", "naive_bayes", "svm"]:
        y_pred = classical_trainer.predict(model_name, X_test_tfidf)
        y_proba = classical_trainer.predict_proba(model_name, X_test_tfidf)
        evaluator.evaluate(y_test, y_pred, f"TF-IDF + {model_name}", y_proba)

    # Also evaluate with BoW features (for comparison)
    print("\n  --- Using BoW Features (comparison) ---")
    from sklearn.linear_model import LogisticRegression as LR
    lr_bow = LR(max_iter=1000, class_weight="balanced", random_state=42)
    lr_bow.fit(X_train_bow, y_train)
    y_pred_bow = lr_bow.predict(X_test_bow)
    y_proba_bow = lr_bow.predict_proba(X_test_bow)
    evaluator.evaluate(y_test, y_pred_bow, "BoW + LogisticRegression", y_proba_bow)

    # ============================================
    # STEP 5: Train Deep Learning Models
    # ============================================
    print("\n🧠 STEP 5: Training Deep Learning Models...")

    try:
        import torch
        from models.deep_models import DeepModelTrainer

        dl_config = DEEP_LEARNING_CONFIG.copy()
        dl_config["epochs"] = 5  # Reduce for faster training
        dl_config["max_len"] = 64

        dl_trainer = DeepModelTrainer(config=dl_config)

        # Train LSTM
        print("\n  [LSTM] Training Bidirectional LSTM...")
        dl_trainer.train_lstm(
            X_train_processed, y_train,
            X_val_processed, y_val,
        )

        # Evaluate LSTM
        lstm_preds, lstm_probs = dl_trainer.predict("lstm", X_test_processed)
        evaluator.evaluate(y_test, lstm_preds, "BiLSTM", lstm_probs)

        # Train GRU
        print("\n  [GRU] Training Bidirectional GRU...")
        dl_trainer.train_gru(
            X_train_processed, y_train,
            X_val_processed, y_val,
        )

        # Evaluate GRU
        gru_preds, gru_probs = dl_trainer.predict("gru", X_test_processed)
        evaluator.evaluate(y_test, gru_preds, "BiGRU", gru_probs)

    except Exception as e:
        logger.warning(f"Deep learning training failed: {e}")
        logger.info("Continuing with classical models only...")

    # ============================================
    # STEP 6: Fine-tune DistilBERT (optional - slow)
    # ============================================
    train_transformer = os.environ.get("TRAIN_TRANSFORMER", "false").lower() == "true"

    if train_transformer:
        print("\n🔬 STEP 6: Fine-tuning DistilBERT...")
        try:
            from models.deep_models import TransformerModelTrainer

            transformer_trainer = TransformerModelTrainer()
            transformer_trainer.train(
                X_train_raw[:2000], y_train[:2000],  # Limit for speed
                X_val_raw[:500], y_val[:500],
            )

            bert_preds, bert_probs = transformer_trainer.predict(X_test_raw[:200])
            evaluator.evaluate(
                y_test[:200], bert_preds, "DistilBERT",
                bert_probs,
            )
        except Exception as e:
            logger.warning(f"Transformer training failed: {e}")
    else:
        print("\n⏭️  STEP 6: Skipping DistilBERT (set TRAIN_TRANSFORMER=true to enable)")

    # ============================================
    # STEP 7: Compare All Models
    # ============================================
    print("\n📊 STEP 7: Model Comparison...")
    comparison = evaluator.compare_models()
    evaluator.save_results()

    # Generate plots
    try:
        evaluator.plot_comparison_chart()
        for model_name in evaluator.results:
            evaluator.plot_confusion_matrix(model_name)
        print("  📈 Plots saved to logs/ directory")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    # ============================================
    # STEP 8: Final Summary
    # ============================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  ⏱️  Total training time: {total_time:.1f}s")
    print(f"  🏆 Best model: {comparison.get('best_model', 'N/A')}")
    print(f"  💾 Models saved to: {MODEL_DIR}")
    print(f"\n  📋 Recommendation for chatbot use:")
    print(f"     • For speed: TF-IDF + Logistic Regression (fast, good accuracy)")
    print(f"     • For accuracy: DistilBERT (best quality, slower inference)")
    print(f"     • Balanced: TF-IDF + SVM (good balance of speed and accuracy)")
    print(f"\n  🚀 Start the chatbot API with: python -m api.app")
    print("=" * 70)


if __name__ == "__main__":
    main()
