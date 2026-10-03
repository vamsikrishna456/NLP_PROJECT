
import re
import string
import logging
from typing import List, Optional, Dict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Download required NLTK data
# ============================================
NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]
for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# ============================================
# Try loading spaCy model
# ============================================
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy English model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Falling back to NLTK for lemmatization.")

# ============================================
# Try loading emoji library
# ============================================
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    logger.warning("emoji library not available. Emoji handling disabled.")

# ============================================
# Try loading contractions library
# ============================================
try:
    import contractions
    CONTRACTIONS_AVAILABLE = True
except ImportError:
    CONTRACTIONS_AVAILABLE = False
    logger.warning("contractions library not available.")

# ============================================
# Try loading language detection
# ============================================
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREPROCESSING_CONFIG, SLANG_DICT


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for sentiment analysis.
    Handles noisy real-world chat inputs including emojis, slang,
    HTML tags, URLs, and various text artifacts.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the text preprocessor.

        Args:
            config: Optional configuration dictionary. Uses PREPROCESSING_CONFIG defaults.
        """
        self.config = config or PREPROCESSING_CONFIG
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        # Keep sentiment-important words that are usually stopwords
        sentiment_words = {
            "not", "no", "nor", "never", "neither", "nobody", "nothing",
            "nowhere", "hardly", "scarcely", "barely", "but", "however",
            "very", "really", "most", "more", "much", "too", "so",
            "just", "only", "own", "same", "than", "other",
            "won", "wouldn", "shouldn", "couldn", "didn", "doesn",
            "hasn", "haven", "isn", "aren", "wasn", "weren",
            "won't", "wouldn't", "shouldn't", "couldn't", "didn't",
            "doesn't", "hasn't", "haven't", "isn't", "aren't",
            "wasn't", "weren't", "don't",
        }
        self.stop_words -= sentiment_words

        self.slang_dict = SLANG_DICT
        logger.info("TextPreprocessor initialized successfully.")

    # ============================================
    # Individual Cleaning Steps
    # ============================================

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.compile(r"<.*?>")
        return re.sub(clean, " ", text)

    def remove_urls(self, text: str) -> str:
        """Remove URLs (http, https, www) from text."""
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text (common in Twitter data)."""
        return re.sub(r"@\w+", " ", text)

    def remove_hashtag_symbol(self, text: str) -> str:
        """Remove the # symbol but keep the hashtag text."""
        return re.sub(r"#(\w+)", r"\1", text)

    def handle_emojis(self, text: str) -> str:
        """
        Convert emojis to their text descriptions.
        Example: 😍 -> 'smiling_face_with_heart_eyes'
        This preserves the sentiment information carried by emojis.
        """
        if EMOJI_AVAILABLE:
            return emoji.demojize(text, delimiters=(" ", " "))
        return text

    def handle_slang(self, text: str) -> str:
        """
        Expand chat slang and abbreviations to their full forms.
        Example: "lol" -> "laughing out loud", "brb" -> "be right back"
        """
        words = text.split()
        expanded = []
        for word in words:
            lower = word.lower().strip(string.punctuation)
            if lower in self.slang_dict:
                expanded.append(self.slang_dict[lower])
            else:
                expanded.append(word)
        return " ".join(expanded)

    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions: "don't" -> "do not", "can't" -> "cannot"
        Critical for sentiment analysis to capture negation.
        """
        if CONTRACTIONS_AVAILABLE:
            return contractions.fix(text)
        # Manual fallback for common contractions
        contraction_map = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
            "shouldn't": "should not", "can't": "cannot", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "i'd": "i would",
            "you'd": "you would", "he'd": "he would", "she'd": "she would",
            "we'd": "we would", "they'd": "they would",
        }
        for contraction, expansion in contraction_map.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r"\d+", " ", text)

    def remove_extra_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into a single space."""
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words using NLTK."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords while preserving sentiment-critical words."""
        return [t for t in tokens if t.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using spaCy.
        Better for sentiment analysis as it preserves word meaning.
        Example: "better" -> "good", "running" -> "run"
        """
        if SPACY_AVAILABLE:
            text = " ".join(tokens)
            doc = nlp(text)
            return [token.lemma_ for token in doc]
        # Fallback to NLTK WordNetLemmatizer
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(t) for t in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens using Porter Stemmer.
        Faster but less accurate than lemmatization.
        Example: "running" -> "run", "happiness" -> "happi"
        """
        return [self.stemmer.stem(t) for t in tokens]

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens by minimum length."""
        min_len = self.config.get("min_token_length", 2)
        return [t for t in tokens if len(t) >= min_len]

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        Returns ISO 639-1 language code (e.g., 'en', 'es', 'fr').
        """
        if LANGDETECT_AVAILABLE:
            try:
                return detect(text)
            except Exception:
                return "en"
        return "en"

    # ============================================
    # Main Preprocessing Pipeline
    # ============================================

    def preprocess(self, text: str, return_tokens: bool = False) -> str:
        """
        Apply the full preprocessing pipeline to a single text.

        Args:
            text: Raw input text
            return_tokens: If True, return list of tokens instead of string

        Returns:
            Preprocessed text string or list of tokens
        """
        if not isinstance(text, str) or not text.strip():
            return [] if return_tokens else ""

        # Step 1: HTML removal
        if self.config.get("remove_html", True):
            text = self.remove_html_tags(text)

        # Step 2: URL removal
        if self.config.get("remove_urls", True):
            text = self.remove_urls(text)

        # Step 3: Mention removal
        if self.config.get("remove_mentions", True):
            text = self.remove_mentions(text)

        # Step 4: Hashtag symbol removal
        if self.config.get("remove_hashtag_symbol", True):
            text = self.remove_hashtag_symbol(text)

        # Step 5: Handle emojis (BEFORE lowercasing to preserve emoji format)
        if self.config.get("handle_emojis", True):
            text = self.handle_emojis(text)

        # Step 6: Lowercase
        if self.config.get("lowercase", True):
            text = text.lower()

        # Step 7: Expand contractions (BEFORE removing punctuation)
        if self.config.get("handle_contractions", True):
            text = self.expand_contractions(text)

        # Step 8: Handle slang
        if self.config.get("handle_slang", True):
            text = self.handle_slang(text)

        # Step 9: Remove punctuation
        if self.config.get("remove_punctuation", True):
            text = self.remove_punctuation(text)

        # Step 10: Remove numbers
        if self.config.get("remove_numbers", False):
            text = self.remove_numbers(text)

        # Step 11: Remove extra whitespace
        text = self.remove_extra_whitespace(text)

        # Step 12: Tokenize
        tokens = self.tokenize(text)

        # Step 13: Remove stopwords
        if self.config.get("remove_stopwords", True):
            tokens = self.remove_stopwords(tokens)

        # Step 14: Lemmatize or Stem
        if self.config.get("lemmatize", True):
            tokens = self.lemmatize(tokens)
        elif self.config.get("stem", False):
            tokens = self.stem(tokens)

        # Step 15: Filter short tokens
        tokens = self.filter_tokens(tokens)

        if return_tokens:
            return tokens
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str], return_tokens: bool = False) -> List:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw text strings
            return_tokens: If True, return list of token lists

        Returns:
            List of preprocessed texts or token lists
        """
        from tqdm import tqdm
        return [
            self.preprocess(text, return_tokens=return_tokens)
            for text in tqdm(texts, desc="Preprocessing", unit="text")
        ]

    def compare_lemmatization_vs_stemming(self, text: str) -> Dict[str, List[str]]:
        """
        Compare lemmatization and stemming outputs for the same text.
        Useful for understanding the trade-offs between both approaches.

        Args:
            text: Input text to compare

        Returns:
            Dictionary with 'original', 'lemmatized', and 'stemmed' token lists
        """
        # Basic cleaning without lemmatization/stemming
        config_backup = self.config.copy()
        self.config["lemmatize"] = False
        self.config["stem"] = False
        base_tokens = self.preprocess(text, return_tokens=True)
        self.config = config_backup

        # Lemmatize
        lemmatized = self.lemmatize(base_tokens.copy())

        # Stem
        stemmed = self.stem(base_tokens.copy())

        return {
            "original_tokens": base_tokens,
            "lemmatized": lemmatized,
            "stemmed": stemmed,
        }


# ============================================
# Quick Test
# ============================================
if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    test_texts = [
        "I LOVE this product!!! 😍😍😍 Best thing ever!!! <b>Amazing</b>",
        "@user Check out https://example.com #awesome #NLP lol bruh this is lit 🔥",
        "I don't think this is good. It's terrible tbh... smh 😤",
        "It's okay I guess 🤷 nothing special, kinda meh ngl",
        "The product isn't bad but it could've been better. Wouldn't recommend tho.",
    ]

    print("=" * 70)
    print("TEXT PREPROCESSING PIPELINE DEMO")
    print("=" * 70)

    for text in test_texts:
        processed = preprocessor.preprocess(text)
        print(f"\n📝 Original:    {text}")
        print(f"✅ Processed:   {processed}")

    # Compare lemmatization vs stemming
    print("\n" + "=" * 70)
    print("LEMMATIZATION vs STEMMING COMPARISON")
    print("=" * 70)

    comparison_text = "The products were running beautifully and the customers loved the improvements"
    comparison = preprocessor.compare_lemmatization_vs_stemming(comparison_text)

    print(f"\n📝 Original text: {comparison_text}")
    print(f"🔤 Base tokens:   {comparison['original_tokens']}")
    print(f"📗 Lemmatized:    {comparison['lemmatized']}")
    print(f"📕 Stemmed:       {comparison['stemmed']}")
