"""
🤖 AI Chatbot — Interactive Terminal Chat
==========================================
An intelligent chatbot that uses TF-IDF vectorization and cosine similarity
to match user input against a knowledge base of intents and responses.

Usage:
    python app.py
"""

import json
import os
import random
import re
import string
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChatBot:
    """An NLP-powered chatbot using TF-IDF + cosine similarity for intent matching."""

    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the chatbot with a knowledge base.

        Args:
            knowledge_base_path: Path to the JSON knowledge base file.
                Defaults to 'knowledge_base.json' in the same directory.
        """
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "knowledge_base.json"
            )

        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        self.confidence_threshold = 0.15
        self.chat_history: list[dict] = []

        # Prepare the TF-IDF matrix from all patterns
        self._prepare_model()

    def _load_knowledge_base(self, path: str) -> dict:
        """Load and validate the knowledge base from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "intents" not in data:
                raise ValueError("Knowledge base must contain an 'intents' key.")
            return data
        except FileNotFoundError:
            print(f"⚠️  Knowledge base not found at: {path}")
            print("   Using built-in fallback responses.")
            return {"intents": []}
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing knowledge base: {e}")
            return {"intents": []}

    def _prepare_model(self):
        """Build the TF-IDF model from all patterns in the knowledge base."""
        self.all_patterns = []
        self.pattern_to_intent = []

        for intent in self.knowledge_base.get("intents", []):
            for pattern in intent.get("patterns", []):
                self.all_patterns.append(self._preprocess(pattern))
                self.pattern_to_intent.append(intent)

        if self.all_patterns:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.all_patterns)
        else:
            self.tfidf_matrix = None

    @staticmethod
    def _preprocess(text: str) -> str:
        """Clean and normalize input text."""
        text = text.lower().strip()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def get_response(self, user_input: str) -> tuple[str, float]:
        """
        Find the best matching response for user input.

        Args:
            user_input: The user's message.

        Returns:
            A tuple of (response_text, confidence_score).
        """
        if not user_input.strip():
            return "Please type something so I can help you! 😊", 0.0

        if self.tfidf_matrix is None or len(self.all_patterns) == 0:
            return self._fallback_response(), 0.0

        # Transform user input using the fitted vectorizer
        processed_input = self._preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])

        # Compute cosine similarity against all patterns
        similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        confidence = float(similarities[best_idx])

        if confidence >= self.confidence_threshold:
            matched_intent = self.pattern_to_intent[best_idx]
            response = random.choice(matched_intent["responses"])
            return response, confidence
        else:
            return self._fallback_response(), confidence

    @staticmethod
    def _fallback_response() -> str:
        """Return a fallback response when confidence is too low."""
        fallbacks = [
            "I'm not sure I understand. Could you rephrase that?",
            "Hmm, I don't have a good answer for that. Try asking about AI, Python, or the projects!",
            "I'm still learning! Could you try asking in a different way?",
            "That's an interesting question, but it's outside my knowledge base. Try 'help' to see what I can do!",
            "I didn't quite catch that. You can ask me about AI concepts, Python, or say 'help' for options.",
        ]
        return random.choice(fallbacks)

    def chat(self, user_input: str) -> str:
        """
        Process a user message and return a response. Logs to history.

        Args:
            user_input: The user's message.

        Returns:
            The chatbot's response string.
        """
        response, confidence = self.get_response(user_input)

        self.chat_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "bot": response,
                "confidence": round(confidence, 4),
            }
        )

        return response

    def get_history(self) -> list[dict]:
        """Return the full chat history."""
        return self.chat_history.copy()


def print_banner():
    """Print the chatbot welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════╗
║              🤖 AI Chatbot v1.0                      ║
║                                                      ║
║   An intelligent chatbot powered by NLP              ║
║   Type 'help' for options  |  Type 'quit' to exit   ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Run the interactive chat loop."""
    print_banner()

    bot = ChatBot()
    print(f"📚 Loaded {len(bot.all_patterns)} patterns across "
          f"{len(bot.knowledge_base.get('intents', []))} intents.\n")

    exit_commands = {"quit", "exit", "bye", "goodbye", "q"}

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break

        if not user_input:
            continue

        if user_input.lower() in exit_commands:
            response = bot.chat(user_input)
            print(f"Bot: {response}\n")
            break

        response = bot.chat(user_input)
        print(f"Bot: {response}\n")

    # Print session summary
    history = bot.get_history()
    if history:
        print(f"📊 Session Summary: {len(history)} messages exchanged.")


if __name__ == "__main__":
    main()