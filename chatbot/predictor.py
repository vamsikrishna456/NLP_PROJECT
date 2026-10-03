"""
============================================
Real-Time Sentiment Predictor
============================================
Takes user input text and returns sentiment label + confidence score.
Also returns detailed NLP pipeline information showing each technique applied.
"""

import re
import string
import logging
import json
import time
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SENTIMENT_LABELS, MODEL_DIR, SLANG_DICT


class SentimentPredictor:
    """
    Real-time sentiment prediction engine.
    Loads trained models, provides instant predictions,
    and returns detailed NLP pipeline step-by-step breakdown.
    """

    def __init__(self):
        self.preprocessor = None
        self.models = {}
        self.vectorizers = {}
        self.default_model = "tfidf_logistic_regression"

    def load_preprocessor(self):
        """Load the text preprocessor."""
        from preprocessing.text_cleaner import TextPreprocessor
        self.preprocessor = TextPreprocessor()
        logger.info("Preprocessor loaded ✅")

    def load_classical_model(self, model_name: str, vectorizer_name: str = "tfidf"):
        """Load a classical ML model with its vectorizer."""
        import joblib
        model_path = MODEL_DIR / f"{model_name}.pkl"
        vec_path = MODEL_DIR / f"{vectorizer_name}_vectorizer.pkl"
        if model_path.exists() and vec_path.exists():
            self.models[f"{vectorizer_name}_{model_name}"] = {
                "model": joblib.load(model_path),
                "vectorizer": joblib.load(vec_path),
                "type": "classical",
            }
            logger.info(f"Loaded {vectorizer_name}_{model_name} ✅")
        else:
            logger.warning(f"Model files not found: {model_path} or {vec_path}")

    def load_transformer_model(self, path: str = None):
        """Load fine-tuned DistilBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            path = path or str(MODEL_DIR / "distilbert_sentiment")
            self.models["distilbert"] = {
                "tokenizer": AutoTokenizer.from_pretrained(path),
                "model": AutoModelForSequenceClassification.from_pretrained(path),
                "type": "transformer",
            }
            self.models["distilbert"]["model"].eval()
            logger.info("DistilBERT loaded ✅")
        except Exception as e:
            logger.warning(f"Could not load transformer: {e}")

    def _capture_nlp_pipeline(self, text: str) -> Dict:
        """
        Run text through each NLP preprocessing step individually
        and capture the output of each step for display.
        """
        pipeline_steps = []
        current = text

        # Step 1: Original Input
        pipeline_steps.append({
            "step": 1,
            "name": "Original Input",
            "technique": "Raw Text",
            "description": "The raw user input before any processing",
            "output": current,
        })

        # Step 2: HTML Tag Removal
        after_html = re.sub(r"<.*?>", " ", current)
        if after_html != current:
            pipeline_steps.append({
                "step": 2, "name": "HTML Removal", "technique": "Regex Cleaning",
                "description": "Removed HTML tags using regex pattern <.*?>",
                "output": after_html.strip(),
            })
            current = after_html

        # Step 3: URL Removal
        after_url = re.sub(r"https?://\S+|www\.\S+", " ", current)
        if after_url != current:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "URL Removal", "technique": "Regex Cleaning",
                "description": "Removed URLs (http/https/www patterns)",
                "output": after_url.strip(),
            })
            current = after_url

        # Step 4: @Mention Removal
        after_mention = re.sub(r"@\w+", " ", current)
        if after_mention != current:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Mention Removal", "technique": "Regex Cleaning",
                "description": "Removed @mentions (common in Twitter data)",
                "output": after_mention.strip(),
            })
            current = after_mention

        # Step 5: Hashtag Symbol Removal
        after_hash = re.sub(r"#(\w+)", r"\1", current)
        if after_hash != current:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Hashtag Processing", "technique": "Regex",
                "description": "Removed # symbol, kept hashtag text",
                "output": after_hash.strip(),
            })
            current = after_hash

        # Step 6: Emoji Handling
        try:
            import emoji
            after_emoji = emoji.demojize(current, delimiters=(" ", " "))
            if after_emoji != current:
                pipeline_steps.append({
                    "step": len(pipeline_steps) + 1, "name": "Emoji → Text",
                    "technique": "Emoji Demojization",
                    "description": "Converted emojis to text descriptions (e.g. 😍 → smiling_face_with_heart_eyes)",
                    "output": after_emoji.strip(),
                })
                current = after_emoji
        except ImportError:
            pass

        # Step 7: Lowercasing
        after_lower = current.lower()
        if after_lower != current:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Lowercasing",
                "technique": "Case Normalization",
                "description": "Converted all characters to lowercase for uniformity",
                "output": after_lower,
            })
            current = after_lower

        # Step 8: Contraction Expansion
        try:
            import contractions
            after_contract = contractions.fix(current)
        except ImportError:
            contraction_map = {
                "don't": "do not", "doesn't": "does not", "didn't": "did not",
                "won't": "will not", "can't": "cannot", "isn't": "is not",
                "aren't": "are not", "wasn't": "was not", "i'm": "i am",
                "i've": "i have", "i'll": "i will", "it's": "it is",
                "that's": "that is", "there's": "there is", "wouldn't": "would not",
                "couldn't": "could not", "shouldn't": "should not",
                "haven't": "have not", "hasn't": "has not",
            }
            after_contract = current
            for c, e in contraction_map.items():
                after_contract = re.sub(re.escape(c), e, after_contract, flags=re.IGNORECASE)

        if after_contract != current:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Contraction Expansion",
                "technique": "Text Normalization",
                "description": "Expanded contractions (e.g. don't → do not, can't → cannot)",
                "output": after_contract,
            })
            current = after_contract

        # Step 9: Slang Expansion
        words = current.split()
        expanded = []
        slang_found = []
        for word in words:
            lower_w = word.lower().strip(string.punctuation)
            if lower_w in SLANG_DICT:
                expanded.append(SLANG_DICT[lower_w])
                slang_found.append(f"{lower_w} → {SLANG_DICT[lower_w]}")
            else:
                expanded.append(word)
        after_slang = " ".join(expanded)
        if slang_found:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Slang Expansion",
                "technique": "Chat Normalization",
                "description": f"Expanded chat slang: {', '.join(slang_found[:5])}",
                "output": after_slang,
            })
            current = after_slang

        # Step 10: Punctuation Removal
        translator = str.maketrans("", "", string.punctuation)
        after_punct = current.translate(translator)
        if after_punct != current:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Punctuation Removal",
                "technique": "Text Cleaning",
                "description": "Removed all punctuation marks",
                "output": after_punct.strip(),
            })
            current = after_punct

        # Step 11: Extra Whitespace
        current = re.sub(r"\s+", " ", current).strip()

        # Step 12: Tokenization
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(current)
        except Exception:
            tokens = current.split()

        pipeline_steps.append({
            "step": len(pipeline_steps) + 1, "name": "Tokenization",
            "technique": "NLTK word_tokenize",
            "description": f"Split text into {len(tokens)} individual tokens",
            "output": " | ".join(tokens),
        })

        # Step 13: Stopword Removal
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
        sentiment_keepers = {"not", "no", "nor", "never", "very", "really", "most", "more", "too", "so"}
        stop_words -= sentiment_keepers
        filtered = [t for t in tokens if t.lower() not in stop_words]
        removed = [t for t in tokens if t.lower() in stop_words]

        if removed:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Stopword Removal",
                "technique": "NLTK Stopwords (Sentiment-Aware)",
                "description": f"Removed {len(removed)} stopwords: {', '.join(removed[:8])}{'...' if len(removed) > 8 else ''}. Kept sentiment words: not, never, very, etc.",
                "output": " | ".join(filtered),
            })
            tokens = filtered

        # Step 14: Lemmatization
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(" ".join(tokens))
            lemmatized = [token.lemma_ for token in doc]
            changes = [(t, l) for t, l in zip(tokens, lemmatized) if t != l]
            if changes:
                change_str = ", ".join([f"{t}→{l}" for t, l in changes[:6]])
                pipeline_steps.append({
                    "step": len(pipeline_steps) + 1, "name": "Lemmatization",
                    "technique": "spaCy en_core_web_sm",
                    "description": f"Reduced words to base forms: {change_str}",
                    "output": " | ".join(lemmatized),
                })
            else:
                pipeline_steps.append({
                    "step": len(pipeline_steps) + 1, "name": "Lemmatization",
                    "technique": "spaCy en_core_web_sm",
                    "description": "Applied lemmatization (no changes needed for these tokens)",
                    "output": " | ".join(lemmatized),
                })
            tokens = lemmatized
        except Exception:
            pass

        # Step 15: Stemming (comparison)
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(t) for t in tokens]
        stem_changes = [(t, s) for t, s in zip(tokens, stemmed) if t != s]

        # Step 16: Min Token Length Filter
        final_tokens = [t for t in tokens if len(t) >= 2]
        short_removed = [t for t in tokens if len(t) < 2]
        if short_removed:
            pipeline_steps.append({
                "step": len(pipeline_steps) + 1, "name": "Short Token Filter",
                "technique": "Min Length = 2",
                "description": f"Removed {len(short_removed)} single-character tokens: {', '.join(short_removed)}",
                "output": " | ".join(final_tokens),
            })

        # Step 17: Feature Extraction Info
        final_text = " ".join(final_tokens)
        pipeline_steps.append({
            "step": len(pipeline_steps) + 1, "name": "TF-IDF Vectorization",
            "technique": "sklearn TfidfVectorizer",
            "description": f"Converted {len(final_tokens)} tokens to TF-IDF weighted numerical features (15,000 max features, bigrams)",
            "output": f"[{len(final_tokens)}-token vector → sparse TF-IDF matrix]",
        })

        # Step 18: Model Classification
        pipeline_steps.append({
            "step": len(pipeline_steps) + 1, "name": "Model Classification",
            "technique": "Logistic Regression / SVM / Naive Bayes",
            "description": "Classified the TF-IDF vector into sentiment categories using trained ML model",
            "output": "→ Sentiment prediction with confidence score",
        })

        # Stemming comparison data
        stemming_comparison = {
            "lemmatized": tokens[:10],
            "stemmed": stemmed[:10],
            "changes": [{"original": t, "stemmed": s} for t, s in stem_changes[:6]],
        }

        return {
            "steps": pipeline_steps,
            "total_steps": len(pipeline_steps),
            "stemming_vs_lemmatization": stemming_comparison,
        }

    def predict(self, text: str, model_name: str = None) -> Dict:
        """
        Predict sentiment for a single text input.
        Returns sentiment, confidence, AND detailed NLP pipeline breakdown.
        """
        start_time = time.time()
        model_name = model_name or self.default_model

        if model_name not in self.models:
            available = list(self.models.keys())
            if available:
                model_name = available[0]
            else:
                return {
                    "error": "No models loaded",
                    "sentiment": "unknown",
                    "confidence": 0.0,
                }

        # Capture NLP pipeline steps
        nlp_pipeline = self._capture_nlp_pipeline(text)

        # Preprocess
        if self.preprocessor:
            processed_text = self.preprocessor.preprocess(text)
        else:
            processed_text = text.lower().strip()

        model_info = self.models[model_name]

        # Get prediction based on model type
        if model_info["type"] == "classical":
            features = model_info["vectorizer"].transform([processed_text])
            model = model_info["model"]

            prediction = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                all_probs = {SENTIMENT_LABELS[i]: float(p) for i, p in enumerate(probabilities)}
            else:
                confidence = 0.8
                all_probs = {}

        elif model_info["type"] == "transformer":
            import torch
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            encoded = tokenizer(
                processed_text, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            )
            with torch.no_grad():
                outputs = model(**encoded)
                probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            all_probs = {SENTIMENT_LABELS[i]: float(p) for i, p in enumerate(probabilities)}
        else:
            return {"error": f"Unknown model type: {model_info['type']}"}

        processing_time = time.time() - start_time
        sentiment_label = SENTIMENT_LABELS.get(prediction, "unknown")

        return {
            "input_text": text,
            "processed_text": processed_text,
            "sentiment": sentiment_label,
            "confidence": round(confidence, 4),
            "probabilities": all_probs,
            "model_used": model_name,
            "processing_time_ms": round(processing_time * 1000, 2),
            "nlp_pipeline": nlp_pipeline,
        }

    def predict_batch(self, texts: list, model_name: str = None) -> list:
        """Predict sentiment for multiple texts."""
        return [self.predict(text, model_name) for text in texts]

    def get_available_models(self) -> list:
        """List all loaded models."""
        return list(self.models.keys())
