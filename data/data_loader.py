"""
============================================
Data Loader Module
============================================
Handles downloading, loading, and splitting datasets for sentiment analysis.
Supports: Twitter Sentiment140, IMDB, and custom chat data.
"""

import os
import re
import csv
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Custom Chat Dataset Generator
# ============================================
# Since downloading large datasets requires API keys / manual download,
# we provide a rich synthetic chat dataset + support for CSV loading.

POSITIVE_TEMPLATES = [
    "I absolutely love this product! It's amazing! 😍",
    "Best experience ever, highly recommend to everyone!",
    "This made my day so much better, thank you! ❤️",
    "Wow, this is incredible! I'm so happy with the results!",
    "Great job team! Everything works perfectly 👏",
    "I'm so grateful for this, it changed my life!",
    "Amazing quality and fantastic customer service!",
    "This is exactly what I needed, love it! 🎉",
    "Couldn't be happier with my purchase, 10/10!",
    "Outstanding work, you guys are the best!",
    "So glad I found this, it's a game changer!",
    "The update is fire 🔥 loving every feature!",
    "OMG this is perfect! Ty so much! 💯",
    "Bruh this app is lit, no cap",
    "Ngl this is the best thing I've seen all week",
    "Super impressed with the quality, keep it up!",
    "My experience has been nothing short of wonderful",
    "Finally something that actually works! Love it!",
    "You guys nailed it, this is perfection 👌",
    "Haha I'm obsessed with this, can't stop using it lol",
    "This product exceeded all my expectations!",
    "Just bought it and I'm already in love 😊",
    "Such a pleasant surprise, really well made!",
    "Five stars all the way, absolutely brilliant!",
    "This is hands down the best purchase I've made this year",
    "Everything about this screams quality, impressed!",
    "My friends are jealous, this is so cool!",
    "The design is gorgeous and it works like a charm",
    "I've tried many alternatives but this is THE one",
    "Can't believe how good this is for the price!",
    "Totally worth every penny, would buy again!",
    "This literally made me smile, so wholesome ❤️",
    "The customer support was incredibly helpful and friendly",
    "I recommended this to all my friends and family!",
    "Shipping was fast and the product is even better in person",
    "Such attention to detail, really appreciate the effort",
    "This solved my problem instantly, lifesaver!",
    "Honestly blown away by the performance",
    "The new features are absolutely fantastic, well done!",
    "I'm a loyal customer now, you've earned my trust",
    "gr8 product tbh, would def recommend",
    "yooo this slaps fr fr 🙌",
    "lowkey the best thing ever ngl",
    "just vibing with this amazing product rn",
    "deadass the goat of all products no cap 🐐",
]

NEGATIVE_TEMPLATES = [
    "This is terrible, worst purchase ever 😡",
    "Completely disappointed, waste of money!",
    "I hate this, it doesn't work at all!",
    "Awful experience, never buying from here again",
    "The quality is so bad, broke after one day 💔",
    "Customer service was rude and unhelpful",
    "Total scam, don't waste your money on this!",
    "I want a refund, this is unacceptable!",
    "Nothing works as advertised, so frustrating!",
    "The worst product I've ever used, period.",
    "This ruined my entire day, thanks for nothing 😤",
    "Smh this is trash, can't believe I paid for this",
    "Bruh this is so bad it's not even funny",
    "Ngl I'm really disappointed with this garbage",
    "wtf is this? Complete waste of time",
    "Don't buy this, it's a total ripoff!",
    "I regret every penny spent on this junk",
    "How did this even get approved for sale?",
    "My expectations were low and I'm still disappointed",
    "The app crashes every 5 minutes, fix your bugs!",
    "Horrible quality, falling apart already",
    "Save yourself the trouble, look elsewhere",
    "I've never been so frustrated with a product",
    "This is an embarrassment, seriously pathetic",
    "The design is ugly and the functionality is worse",
    "Support team ignores all complaints, terrible company",
    "Waited 3 weeks for delivery and got a broken item",
    "False advertising at its finest, nothing matches the description",
    "One star is too generous for this disaster",
    "Return process is a nightmare, adding insult to injury",
    "h8 this so much, worst ever fr",
    "nah this ain't it chief, big L 👎",
    "lowkey wanna cry, this is so bad",
    "bro this is sus af, doesn't work at all",
    "deadass the worst thing I've ever bought no cap",
    "I'm beyond furious with this purchase!",
    "Toxic customer support, never again!",
    "The quality has gone downhill dramatically",
    "Bugs everywhere, did anyone even test this?",
    "Misleading reviews, the product is nothing like shown",
    "Complete failure from start to finish",
    "I feel cheated and disrespected as a customer",
    "This belongs in the trash, not on the shelf",
    "Glitchy, slow, and unreliable — the trifecta of terrible",
    "Would rate zero stars if I could",
]

NEUTRAL_TEMPLATES = [
    "It's okay, nothing special but gets the job done",
    "Average product, meets basic expectations",
    "Not bad, not great. Just somewhere in the middle",
    "It works fine I guess, could be better though",
    "Decent quality for the price, fair enough",
    "The product is alright, nothing to write home about",
    "Standard features, does what it says",
    "Meh, it's just okay tbh 🤷",
    "Could use some improvements but it's functional",
    "Neither impressed nor disappointed, it's adequate",
    "It serves its purpose, that's about it",
    "Ordered the item, arrived on time, works okay",
    "Not sure how I feel about this yet",
    "It's a product, it exists, it works sometimes",
    "idk how to feel about this one tbh",
    "Some features are good, others need work",
    "The product does exactly what's described, no more no less",
    "It's fine for basic use, don't expect anything fancy",
    "Mediocre at best, but it does work",
    "I have mixed feelings about this purchase",
    "Can someone explain how to use this feature?",
    "Just received my order, will update after testing",
    "Does anyone know if this comes in other colors?",
    "The size chart was accurate, fits as expected",
    "Reasonable product for everyday use",
    "It's on par with similar products in this range",
    "Nothing wrong with it, but nothing exciting either",
    "Functional and practical, just not very exciting",
    "Does the job adequately, no complaints really",
    "Standard quality, standard price, standard experience",
    "idk it's kinda mid ngl",
    "eh it's whatever ig",
    "not bad not good just vibes",
    "it exists I guess lol",
    "its aight, could be worse could be better",
    "The instructions were clear, setup was straightforward",
    "Comparable to other options on the market",
    "Neither love it nor hate it, it's serviceable",
    "It fulfilled my basic needs, that's sufficient",
    "Fair product, reasonable expectations met",
    "I suppose it does what it's supposed to do",
    "Took a while to arrive but the product is fine",
    "It's a middle-of-the-road option, nothing more",
    "Passable quality, wouldn't rave about it",
    "I don't have strong feelings about this either way",
]


def generate_synthetic_chat_data(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate a balanced synthetic chat dataset for sentiment analysis.
    Includes realistic chat language, emojis, slang, and varied expressions.

    Args:
        n_samples: Total number of samples to generate

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    logger.info(f"Generating {n_samples} synthetic chat samples...")

    samples_per_class = n_samples // 3
    data = []

    # Add noise/variation to templates
    def add_variation(text: str) -> str:
        """Add random variations to make data more realistic."""
        variations = [
            lambda t: t,                                    # Keep original
            lambda t: t.upper(),                            # ALL CAPS
            lambda t: t.lower(),                            # all lowercase
            lambda t: t + "!!!",                            # Extra punctuation
            lambda t: t + "...",                            # Trailing dots
            lambda t: "honestly, " + t.lower(),             # Add filler
            lambda t: "I think " + t[0].lower() + t[1:],   # Add prefix
            lambda t: t + " lol",                           # Add chat suffix
            lambda t: t.replace("!", "!!!"),                # Amplify
            lambda t: "tbh " + t.lower(),                   # Add slang prefix
        ]
        return random.choice(variations)(text)

    # Generate positive samples
    for _ in range(samples_per_class):
        text = random.choice(POSITIVE_TEMPLATES)
        data.append({"text": add_variation(text), "label": 2})  # positive = 2

    # Generate negative samples
    for _ in range(samples_per_class):
        text = random.choice(NEGATIVE_TEMPLATES)
        data.append({"text": add_variation(text), "label": 0})  # negative = 0

    # Generate neutral samples
    for _ in range(samples_per_class):
        text = random.choice(NEUTRAL_TEMPLATES)
        data.append({"text": add_variation(text), "label": 1})  # neutral = 1

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    logger.info(f"Generated dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

    return df


def load_csv_dataset(filepath: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load a custom CSV dataset.

    Args:
        filepath: Path to the CSV file
        text_col: Name of the text column
        label_col: Name of the label column

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath, encoding="utf-8")

    # Rename columns to standard names
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Ensure required columns exist
    assert "text" in df.columns, f"Column '{text_col}' not found in dataset"
    assert "label" in df.columns, f"Column '{label_col}' not found in dataset"

    # Drop nulls
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    logger.info(f"Loaded dataset shape: {df.shape}")
    return df


def load_imdb_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load IMDB dataset using Hugging Face datasets library.
    Maps: positive -> 2, negative -> 0 (binary, no neutral)

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    try:
        from datasets import load_dataset

        logger.info("Loading IMDB dataset from Hugging Face...")
        dataset = load_dataset("imdb")

        # Combine train and test
        texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
        labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"])

        df = pd.DataFrame({"text": texts, "label": labels})

        # Map: 0 (neg) -> 0, 1 (pos) -> 2
        df["label"] = df["label"].map({0: 0, 1: 2})

        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)

        logger.info(f"IMDB dataset shape: {df.shape}")
        return df

    except Exception as e:
        logger.warning(f"Could not load IMDB dataset: {e}")
        logger.info("Falling back to synthetic data...")
        return generate_synthetic_chat_data(max_samples or 3000)


def load_twitter_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load Twitter Sentiment140 dataset.
    Since the full dataset requires downloading from Kaggle,
    this attempts to use HuggingFace or falls back to synthetic data.

    Maps: 0 (neg) -> 0, 2 (neutral) -> 1, 4 (pos) -> 2

    Args:
        max_samples: Maximum number of samples

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    try:
        from datasets import load_dataset

        logger.info("Loading Twitter sentiment dataset from Hugging Face...")
        dataset = load_dataset("sentiment140")

        texts = list(dataset["train"]["text"])
        labels = list(dataset["train"]["sentiment"])

        df = pd.DataFrame({"text": texts, "label": labels})

        # Map sentiment140 labels: 0->0 (neg), 2->1 (neutral), 4->2 (pos)
        label_map = {0: 0, 2: 1, 4: 2}
        df["label"] = df["label"].map(label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)

        logger.info(f"Twitter dataset shape: {df.shape}")
        return df

    except Exception as e:
        logger.warning(f"Could not load Twitter dataset: {e}")
        logger.info("Falling back to synthetic data...")
        return generate_synthetic_chat_data(max_samples or 3000)


def prepare_dataset(
    dataset_name: str = "synthetic",
    max_samples: Optional[int] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Main function to prepare the dataset for training.
    Loads data, splits into train/val/test sets.

    Args:
        dataset_name: One of "synthetic", "imdb", "twitter", or path to CSV
        max_samples: Maximum total samples to use
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (texts, labels)
    """
    # Load dataset based on name
    if dataset_name == "synthetic":
        df = generate_synthetic_chat_data(max_samples or 3000)
    elif dataset_name == "imdb":
        df = load_imdb_dataset(max_samples)
    elif dataset_name == "twitter" or dataset_name == "twitter_sentiment":
        df = load_twitter_dataset(max_samples)
    elif os.path.exists(dataset_name):
        df = load_csv_dataset(dataset_name)
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=random_state)
    else:
        logger.warning(f"Unknown dataset '{dataset_name}', using synthetic data.")
        df = generate_synthetic_chat_data(max_samples or 3000)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_trainval,
    )

    logger.info(f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "num_classes": len(set(labels)),
        "label_names": {0: "negative", 1: "neutral", 2: "positive"},
    }


# ============================================
# Quick test
# ============================================
if __name__ == "__main__":
    # Test synthetic data generation
    data = prepare_dataset("synthetic", max_samples=1500)
    print(f"\nTrain samples: {len(data['train'][0])}")
    print(f"Val samples:   {len(data['val'][0])}")
    print(f"Test samples:  {len(data['test'][0])}")
    print(f"\nSample texts:")
    for i in range(5):
        label = data["train"][1][i]
        text = data["train"][0][i][:80]
        print(f"  [{data['label_names'][label]:>8}] {text}...")
