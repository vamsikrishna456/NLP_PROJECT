"""
============================================
Feature Engineering Module
============================================
Implements: BoW, TF-IDF, Word2Vec, Transformer Embeddings
"""

import logging
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURE_CONFIG, MODEL_DIR


class BagOfWordsExtractor:
    """BoW feature extractor - converts text to word frequency vectors."""

    def __init__(self, max_features=None, ngram_range=None):
        self.max_features = max_features or FEATURE_CONFIG["bow_max_features"]
        self.ngram_range = ngram_range or FEATURE_CONFIG["bow_ngram_range"]
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        logger.info(f"Fitting BoW (max_features={self.max_features})")
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info(f"BoW shape: {features.shape}")
        return features

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, path=None):
        path = path or str(MODEL_DIR / "bow_vectorizer.pkl")
        joblib.dump(self.vectorizer, path)

    def load(self, path=None):
        path = path or str(MODEL_DIR / "bow_vectorizer.pkl")
        self.vectorizer = joblib.load(path)
        self.is_fitted = True


class TFIDFExtractor:
    """TF-IDF feature extractor - weighs words by importance."""

    def __init__(self, max_features=None, ngram_range=None, min_df=None, max_df=None):
        self.max_features = max_features or FEATURE_CONFIG["tfidf_max_features"]
        self.ngram_range = ngram_range or FEATURE_CONFIG["tfidf_ngram_range"]
        self.min_df = min_df or FEATURE_CONFIG["tfidf_min_df"]
        self.max_df = max_df or FEATURE_CONFIG["tfidf_max_df"]
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range,
            min_df=self.min_df, max_df=self.max_df, sublinear_tf=True,
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        logger.info(f"Fitting TF-IDF (max_features={self.max_features})")
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF shape: {features.shape}")
        return features

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, path=None):
        path = path or str(MODEL_DIR / "tfidf_vectorizer.pkl")
        joblib.dump(self.vectorizer, path)

    def load(self, path=None):
        path = path or str(MODEL_DIR / "tfidf_vectorizer.pkl")
        self.vectorizer = joblib.load(path)
        self.is_fitted = True


class Word2VecExtractor:
    """Word2Vec embedding extractor - averages word vectors per document."""

    def __init__(self, vector_size=None, window=None, min_count=None, epochs=None):
        self.vector_size = vector_size or FEATURE_CONFIG["w2v_vector_size"]
        self.window = window or FEATURE_CONFIG["w2v_window"]
        self.min_count = min_count or FEATURE_CONFIG["w2v_min_count"]
        self.epochs = epochs or FEATURE_CONFIG["w2v_epochs"]
        self.model = None

    def fit(self, tokenized_texts):
        from gensim.models import Word2Vec
        logger.info(f"Training Word2Vec (dim={self.vector_size})")
        self.model = Word2Vec(
            sentences=tokenized_texts, vector_size=self.vector_size,
            window=self.window, min_count=self.min_count,
            epochs=self.epochs, workers=4, sg=1,
        )
        logger.info(f"Word2Vec vocab size: {len(self.model.wv)}")

    def get_document_vector(self, tokens):
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

    def transform(self, tokenized_texts):
        return np.array([self.get_document_vector(t) for t in tokenized_texts])

    def fit_transform(self, tokenized_texts):
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)

    def get_similar_words(self, word, topn=10):
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []

    def save(self, path=None):
        path = path or str(MODEL_DIR / "word2vec.model")
        self.model.save(path)

    def load(self, path=None):
        from gensim.models import Word2Vec
        path = path or str(MODEL_DIR / "word2vec.model")
        self.model = Word2Vec.load(path)


class TransformerEmbeddingExtractor:
    """DistilBERT contextual embedding extractor using [CLS] token."""

    def __init__(self, model_name=None, max_length=None):
        self.model_name = model_name or FEATURE_CONFIG["transformer_model"]
        self.max_length = max_length or FEATURE_CONFIG["transformer_max_length"]
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            logger.info(f"Loading transformer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info(f"Transformer on: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load transformer: {e}")
            self.model = None

    def transform(self, texts, batch_size=32):
        import torch
        from tqdm import tqdm
        if self.model is None:
            return np.zeros((len(texts), 768))

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

        return np.vstack(all_embeddings)


class FeatureExtractor:
    """Unified interface for all feature extraction methods."""

    def __init__(self):
        self.extractors = {}

    def create_bow_features(self, train_texts, test_texts=None):
        bow = BagOfWordsExtractor()
        result = {"train": bow.fit_transform(train_texts)}
        self.extractors["bow"] = bow
        if test_texts:
            result["test"] = bow.transform(test_texts)
        return result

    def create_tfidf_features(self, train_texts, test_texts=None):
        tfidf = TFIDFExtractor()
        result = {"train": tfidf.fit_transform(train_texts)}
        self.extractors["tfidf"] = tfidf
        if test_texts:
            result["test"] = tfidf.transform(test_texts)
        return result

    def create_w2v_features(self, train_tokens, test_tokens=None):
        w2v = Word2VecExtractor()
        result = {"train": w2v.fit_transform(train_tokens)}
        self.extractors["word2vec"] = w2v
        if test_tokens:
            result["test"] = w2v.transform(test_tokens)
        return result

    def create_transformer_features(self, train_texts, test_texts=None):
        trans = TransformerEmbeddingExtractor()
        result = {"train": trans.transform(train_texts)}
        self.extractors["transformer"] = trans
        if test_texts:
            result["test"] = trans.transform(test_texts)
        return result

    def save_all(self):
        for name, ext in self.extractors.items():
            if hasattr(ext, "save"):
                ext.save()


if __name__ == "__main__":
    texts = [
        "love this product amazing",
        "terrible worst purchase",
        "okay nothing special",
        "fantastic highly recommend",
        "waste money do not buy",
    ]
    print("=" * 50)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 50)

    bow = BagOfWordsExtractor(max_features=100)
    print(f"\nBoW shape: {bow.fit_transform(texts).shape}")

    tfidf = TFIDFExtractor(max_features=100)
    print(f"TF-IDF shape: {tfidf.fit_transform(texts).shape}")

    w2v = Word2VecExtractor(vector_size=50)
    tokens = [t.split() for t in texts]
    print(f"Word2Vec shape: {w2v.fit_transform(tokens).shape}")

    print("\n✅ All feature extractors working!")
