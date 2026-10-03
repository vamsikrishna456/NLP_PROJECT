"""
============================================
Deep Learning Models Module
============================================
Implements: BiLSTM, GRU, and DistilBERT fine-tuning for sentiment analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEEP_LEARNING_CONFIG, TRANSFORMER_CONFIG, MODEL_DIR


# ============================================
# PyTorch LSTM/GRU Models
# ============================================

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available.")


if TORCH_AVAILABLE:
    class TextDataset(Dataset):
        """PyTorch dataset for text classification."""
        def __init__(self, texts, labels, vocab, max_len=128):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            tokens = self.texts[idx].split()[:self.max_len]
            indices = [self.vocab.get(t, self.vocab.get("<UNK>", 1)) for t in tokens]
            # Pad
            if len(indices) < self.max_len:
                indices += [0] * (self.max_len - len(indices))
            return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    class SentimentLSTM(nn.Module):
        """Bidirectional LSTM for sentiment classification."""
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                     num_layers=2, dropout=0.3, bidirectional=True):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            direction = 2 if bidirectional else 1
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * direction, num_classes)

        def forward(self, x):
            embedded = self.dropout(self.embedding(x))
            output, (hidden, _) = self.lstm(embedded)
            # Concatenate last hidden states from both directions
            if self.lstm.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            return self.fc(self.dropout(hidden))

    class SentimentGRU(nn.Module):
        """Bidirectional GRU for sentiment classification."""
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                     num_layers=2, dropout=0.3, bidirectional=True):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.gru = nn.GRU(
                embedding_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            direction = 2 if bidirectional else 1
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * direction, num_classes)

        def forward(self, x):
            embedded = self.dropout(self.embedding(x))
            output, hidden = self.gru(embedded)
            if self.gru.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            return self.fc(self.dropout(hidden))


def build_vocab(texts: List[str], max_vocab: int = 20000) -> Dict:
    """Build vocabulary from texts."""
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(sorted_words[:max_vocab - 2]):
        vocab[word] = i + 2
    logger.info(f"Vocabulary size: {len(vocab)}")
    return vocab


class DeepModelTrainer:
    """Trains LSTM and GRU models."""

    def __init__(self, config=None):
        self.config = config or DEEP_LEARNING_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self.models = {}
        self.vocab = None
        logger.info(f"Device: {self.device}")

    def prepare_data(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Prepare DataLoaders."""
        self.vocab = build_vocab(train_texts, self.config["vocab_size"])
        max_len = self.config.get("max_len", 128)
        bs = self.config["batch_size"]

        train_ds = TextDataset(train_texts, train_labels, self.vocab, max_len)
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

        if val_texts and val_labels:
            val_ds = TextDataset(val_texts, val_labels, self.vocab, max_len)
            self.val_loader = DataLoader(val_ds, batch_size=bs)
        else:
            self.val_loader = None

    def _train_model(self, model, model_name):
        """Generic training loop with early stopping."""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

            train_acc = correct / total
            avg_loss = total_loss / len(self.train_loader)

            # Validation
            val_loss = 0
            if self.val_loader:
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()
                        preds = outputs.argmax(dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += len(labels)

                val_loss /= len(self.val_loader)
                val_acc = val_correct / val_total
                logger.info(f"Epoch {epoch+1}/{self.config['epochs']} | "
                           f"Loss: {avg_loss:.4f} | Acc: {train_acc:.4f} | "
                           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), str(MODEL_DIR / f"{model_name}_best.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["patience"]:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.config['epochs']} | "
                           f"Loss: {avg_loss:.4f} | Acc: {train_acc:.4f}")

        self.models[model_name] = model
        return model

    def train_lstm(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Train BiLSTM model."""
        logger.info("Training BiLSTM...")
        self.prepare_data(train_texts, train_labels, val_texts, val_labels)
        model = SentimentLSTM(
            vocab_size=len(self.vocab),
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_classes=3,
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bidirectional=self.config["bidirectional"],
        )
        return self._train_model(model, "lstm")

    def train_gru(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Train BiGRU model."""
        logger.info("Training BiGRU...")
        if self.vocab is None:
            self.prepare_data(train_texts, train_labels, val_texts, val_labels)
        model = SentimentGRU(
            vocab_size=len(self.vocab),
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_classes=3,
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bidirectional=self.config["bidirectional"],
        )
        return self._train_model(model, "gru")

    def predict(self, model_name, texts):
        """Get predictions from a deep learning model."""
        model = self.models[model_name]
        model.eval()
        max_len = self.config.get("max_len", 128)

        all_preds = []
        all_probs = []
        with torch.no_grad():
            for text in texts:
                tokens = text.split()[:max_len]
                indices = [self.vocab.get(t, 1) for t in tokens]
                if len(indices) < max_len:
                    indices += [0] * (max_len - len(indices))
                x = torch.tensor([indices], dtype=torch.long).to(self.device)
                output = model(x)
                probs = torch.softmax(output, dim=1)
                all_preds.append(output.argmax(dim=1).item())
                all_probs.append(probs.cpu().numpy()[0])

        return np.array(all_preds), np.array(all_probs)


# ============================================
# DistilBERT Fine-tuning
# ============================================

class TransformerModelTrainer:
    """Fine-tunes DistilBERT for sentiment classification."""

    def __init__(self, config=None):
        self.config = config or TRANSFORMER_CONFIG
        self.model = None
        self.tokenizer = None

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Fine-tune DistilBERT."""
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                TrainingArguments, Trainer
            )
            from datasets import Dataset as HFDataset
        except ImportError:
            logger.error("transformers/datasets not available")
            return None

        logger.info(f"Fine-tuning {self.config['model_name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"], num_labels=3
        )

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"], padding="max_length",
                truncation=True, max_length=self.config["max_length"],
            )

        train_ds = HFDataset.from_dict({"text": train_texts, "label": train_labels})
        train_ds = train_ds.map(tokenize_fn, batched=True)

        eval_ds = None
        if val_texts and val_labels:
            eval_ds = HFDataset.from_dict({"text": val_texts, "label": val_labels})
            eval_ds = eval_ds.map(tokenize_fn, batched=True)

        output_dir = str(MODEL_DIR / "distilbert_sentiment")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_steps=self.config["warmup_steps"],
            eval_strategy="epoch" if eval_ds else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_ds else False,
            logging_steps=50,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score
            predictions = np.argmax(eval_pred.predictions, axis=-1)
            labels = eval_pred.label_ids
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted"),
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"DistilBERT saved to {output_dir} ✅")
        return self.model

    def predict(self, texts):
        """Predict sentiment with DistilBERT."""
        import torch
        if self.model is None or self.tokenizer is None:
            logger.error("Model not trained/loaded")
            return None, None

        self.model.eval()
        device = next(self.model.parameters()).device

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.config["max_length"], return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        return preds, probs

    def load(self, path=None):
        """Load a fine-tuned model."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = path or str(MODEL_DIR / "distilbert_sentiment")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        logger.info(f"Loaded DistilBERT from {path}")
