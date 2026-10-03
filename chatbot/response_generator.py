"""
============================================
Response Generator Module
============================================
Generates tone-aware chatbot responses based on sentiment analysis.
Outputs clean JSON with sentiment + suggested response tone.
"""

import json
import random
import logging
from typing import Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SENTIMENT_TONES


# Response templates organized by detected sentiment
RESPONSE_TEMPLATES = {
    "positive": {
        "tone": "happy",
        "responses": [
            "That's wonderful to hear! 😊 I'm glad you're having a great experience!",
            "Awesome! Your positive energy is contagious! Keep it up! 🎉",
            "So happy to hear that! Is there anything else I can help you with? 💛",
            "That's fantastic! We love hearing positive feedback! ✨",
            "Yay! It makes our day to know you're satisfied! 🌟",
        ],
        "follow_ups": [
            "Would you like to share your experience with others?",
            "Is there anything else we can help make even better?",
            "Feel free to explore more of our features!",
        ],
    },
    "negative": {
        "tone": "empathetic",
        "responses": [
            "I'm really sorry to hear that. 😔 Let me help you resolve this.",
            "I understand your frustration, and I want to make this right for you.",
            "That's not the experience we want you to have. Let's fix this together.",
            "I sincerely apologize for the inconvenience. Your concerns are valid.",
            "I hear you, and I'm here to help. Let's work through this step by step.",
        ],
        "follow_ups": [
            "Could you tell me more about what went wrong so I can assist better?",
            "Would you like me to connect you with a specialist?",
            "Can I help escalate this issue for you?",
        ],
    },
    "neutral": {
        "tone": "neutral",
        "responses": [
            "Thanks for sharing your thoughts! Let me know if you need anything.",
            "I appreciate your feedback. Is there something specific I can help with?",
            "Got it! Feel free to ask me anything you'd like to know more about.",
            "Thanks for reaching out! How can I assist you further?",
            "I understand. Would you like more information about anything?",
        ],
        "follow_ups": [
            "Is there something specific you'd like to know more about?",
            "Can I help you explore more options?",
            "Would you like to try any of our features?",
        ],
    },
}

# Sarcasm indicators (basic heuristic detection)
SARCASM_INDICATORS = [
    "oh great", "yeah right", "sure thing", "as if",
    "oh wonderful", "how lovely", "oh perfect", "just great",
    "oh fantastic", "wow so amazing", "totally not",
    "oh joy", "what a surprise", "who would have thought",
    "shocking", "oh really", "no way", "you don't say",
]


class ResponseGenerator:
    """
    Generates context-aware chatbot responses based on sentiment analysis.
    Includes sarcasm detection and tone-matched responses.
    """

    def __init__(self):
        self.response_templates = RESPONSE_TEMPLATES

    def detect_sarcasm(self, text: str) -> bool:
        """
        Basic sarcasm detection using heuristic patterns.
        Checks for common sarcastic phrases and contradictory signals.
        """
        text_lower = text.lower()

        # Check for known sarcastic phrases
        for indicator in SARCASM_INDICATORS:
            if indicator in text_lower:
                return True

        # Check for contradictory signals (positive words + negative context)
        positive_words = {"great", "amazing", "wonderful", "fantastic", "perfect", "love"}
        negative_signals = {"not", "but", "however", "though", "...", "smh", "🙄"}

        words = set(text_lower.split())
        has_positive = bool(words & positive_words)
        has_negative = bool(words & negative_signals)

        if has_positive and has_negative:
            return True

        # Excessive punctuation with short text could indicate sarcasm
        if text.count("!") > 3 and len(text.split()) < 10:
            return True

        return False

    def generate_response(self, sentiment_result: Dict) -> Dict:
        """
        Generate a complete chatbot response based on sentiment analysis.

        Args:
            sentiment_result: Output from SentimentPredictor.predict()

        Returns:
            Clean JSON response with sentiment, tone, and suggested reply
        """
        sentiment = sentiment_result.get("sentiment", "neutral")
        input_text = sentiment_result.get("input_text", "")
        confidence = sentiment_result.get("confidence", 0.0)

        # Check for sarcasm
        is_sarcastic = self.detect_sarcasm(input_text)

        # If sarcasm detected, adjust sentiment interpretation
        if is_sarcastic and sentiment == "positive":
            sentiment = "negative"
            logger.info("⚠️ Sarcasm detected - adjusting sentiment to negative")

        # Get response template
        template = self.response_templates.get(sentiment, self.response_templates["neutral"])
        tone = template["tone"]
        response_text = random.choice(template["responses"])
        follow_up = random.choice(template["follow_ups"])

        # Build JSON response
        response = {
            "status": "success",
            "input": {
                "text": input_text,
                "processed": sentiment_result.get("processed_text", ""),
            },
            "analysis": {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": sentiment_result.get("probabilities", {}),
                "sarcasm_detected": is_sarcastic,
            },
            "response": {
                "tone": tone,
                "message": response_text,
                "follow_up": follow_up,
            },
            "metadata": {
                "model_used": sentiment_result.get("model_used", "unknown"),
                "processing_time_ms": sentiment_result.get("processing_time_ms", 0),
            },
            "nlp_pipeline": sentiment_result.get("nlp_pipeline", {}),
        }

        return response

    def format_json(self, response: Dict) -> str:
        """Format response as pretty-printed JSON string."""
        return json.dumps(response, indent=2, ensure_ascii=False)


# ============================================
# Quick Test
# ============================================
if __name__ == "__main__":
    generator = ResponseGenerator()

    test_results = [
        {"input_text": "I love this!", "sentiment": "positive", "confidence": 0.95,
         "probabilities": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
         "model_used": "test", "processing_time_ms": 5.0, "processed_text": "love"},
        {"input_text": "This is terrible", "sentiment": "negative", "confidence": 0.88,
         "probabilities": {"negative": 0.88, "neutral": 0.07, "positive": 0.05},
         "model_used": "test", "processing_time_ms": 5.0, "processed_text": "terrible"},
        {"input_text": "Oh great, another bug!", "sentiment": "positive", "confidence": 0.6,
         "probabilities": {"negative": 0.3, "neutral": 0.1, "positive": 0.6},
         "model_used": "test", "processing_time_ms": 5.0, "processed_text": "great bug"},
    ]

    for result in test_results:
        response = generator.generate_response(result)
        print(generator.format_json(response))
        print("-" * 50)
