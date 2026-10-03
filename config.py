"""
============================================
Configuration Module
============================================
Central configuration for the NLP Sentiment Analysis Pipeline.
All hyperparameters, paths, and settings are defined here.
"""

import os
from pathlib import Path

# ============================================
# Project Paths
# ============================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "datasets"
MODEL_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# Dataset Configuration
# ============================================
DATASET_CONFIG = {
    "name": "synthetic",                 # Options: "synthetic", "twitter_sentiment", "imdb", "custom"
    "max_samples": 3000,                # Smaller for fast training on CPU
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
}

# ============================================
# Preprocessing Configuration
# ============================================
PREPROCESSING_CONFIG = {
    "lowercase": True,
    "remove_html": True,
    "remove_urls": True,
    "remove_mentions": True,
    "remove_hashtag_symbol": True,       # Keep hashtag text, remove #
    "remove_punctuation": True,
    "remove_numbers": False,
    "remove_stopwords": True,
    "handle_emojis": True,              # Convert emojis to text descriptions
    "handle_slang": True,               # Expand chat slang/abbreviations
    "handle_contractions": True,        # Expand contractions (don't -> do not)
    "lemmatize": True,                  # Use spaCy lemmatization
    "stem": False,                      # Use NLTK stemming (compare with lemmatization)
    "min_token_length": 2,
    "max_sequence_length": 128,
}

# ============================================
# Feature Engineering Configuration
# ============================================
FEATURE_CONFIG = {
    # Bag of Words
    "bow_max_features": 10000,
    "bow_ngram_range": (1, 2),

    # TF-IDF
    "tfidf_max_features": 15000,
    "tfidf_ngram_range": (1, 2),
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.95,

    # Word2Vec
    "w2v_vector_size": 100,
    "w2v_window": 5,
    "w2v_min_count": 2,
    "w2v_epochs": 10,

    # Transformer Embeddings
    "transformer_model": "distilbert-base-uncased",
    "transformer_max_length": 128,
}

# ============================================
# Classical ML Model Configuration
# ============================================
CLASSICAL_MODEL_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    },
    "naive_bayes": {
        "alpha": 1.0,
    },
    "svm": {
        "C": 1.0,
        "kernel": "linear",
        "class_weight": "balanced",
        "max_iter": 5000,
    },
}

# ============================================
# Deep Learning Configuration
# ============================================
DEEP_LEARNING_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.3,
    "bidirectional": True,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 5,
    "patience": 2,                      # Early stopping patience
    "vocab_size": 10000,
}

# ============================================
# Transformer Fine-tuning Configuration
# ============================================
TRANSFORMER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "patience": 2,
}

# ============================================
# Evaluation Configuration
# ============================================
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "average": "weighted",              # For multi-class metrics
}

# ============================================
# API Configuration
# ============================================
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "default_model": "distilbert",      # Best model for production
    "cors_origins": ["*"],
}

# ============================================
# Sentiment Labels
# ============================================
SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

SENTIMENT_TONES = {
    "positive": "happy",
    "neutral": "neutral",
    "negative": "empathetic",
}

# ============================================
# Slang Dictionary (Common Chat Abbreviations)
# ============================================
SLANG_DICT = {
    "brb": "be right back",
    "btw": "by the way",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "smh": "shaking my head",
    "fyi": "for your information",
    "idk": "i do not know",
    "imo": "in my opinion",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "omg": "oh my god",
    "wtf": "what the fuck",
    "nvm": "never mind",
    "ikr": "i know right",
    "thx": "thanks",
    "ty": "thank you",
    "pls": "please",
    "plz": "please",
    "bc": "because",
    "cuz": "because",
    "ur": "your",
    "u": "you",
    "r": "are",
    "2": "to",
    "4": "for",
    "b4": "before",
    "gr8": "great",
    "h8": "hate",
    "l8r": "later",
    "np": "no problem",
    "nw": "no worries",
    "gg": "good game",
    "gl": "good luck",
    "hmu": "hit me up",
    "dm": "direct message",
    "af": "as fuck",
    "irl": "in real life",
    "tbf": "to be fair",
    "rn": "right now",
    "w/": "with",
    "w/o": "without",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "ain't": "is not",
    "y'all": "you all",
    "sup": "what is up",
    "wassup": "what is up",
    "nah": "no",
    "yep": "yes",
    "yup": "yes",
    "nope": "no",
    "dope": "cool",
    "lit": "amazing",
    "salty": "upset",
    "slay": "doing great",
    "vibe": "feeling",
    "goat": "greatest of all time",
    "fomo": "fear of missing out",
    "lowkey": "somewhat",
    "highkey": "very much",
    "fire": "amazing",
    "cap": "lie",
    "no cap": "no lie",
    "bet": "okay sure",
    "sus": "suspicious",
    "simp": "someone who does too much",
    "stan": "obsessive fan",
}
