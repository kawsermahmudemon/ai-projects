"""
📝 NLP — Sentiment Analysis
==============================
Two approaches to sentiment analysis:
  1. Rule-based: VADER (Valence Aware Dictionary and sEntiment Reasoner)
  2. ML-based: Naive Bayes trained on NLTK movie reviews corpus

Usage:
    python sentiment_analysis.py                                       # Interactive mode
    python sentiment_analysis.py --text "This movie was fantastic!"     # Single text
    python sentiment_analysis.py --compare                             # Compare both methods
"""

import argparse

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def download_nltk_data():
    """Download required NLTK data."""
    packages = ["vader_lexicon", "movie_reviews", "punkt", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


# ── VADER (Rule-Based) ────────────────────────────────────────────────────────

class VaderAnalyzer:
    """Rule-based sentiment analysis using VADER."""

    def __init__(self):
        download_nltk_data()
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text using VADER.

        Returns:
            Dict with 'compound', 'pos', 'neg', 'neu', 'label', and 'emoji'.
        """
        scores = self.analyzer.polarity_scores(text)

        # Classify based on compound score
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
            emoji = "😊"
        elif compound <= -0.05:
            label = "NEGATIVE"
            emoji = "😞"
        else:
            label = "NEUTRAL"
            emoji = "😐"

        return {
            "compound": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "label": label,
            "emoji": emoji,
        }


# ── Naive Bayes (ML-Based) ────────────────────────────────────────────────────

class NaiveBayesAnalyzer:
    """ML-based sentiment analysis using Naive Bayes on movie reviews."""

    def __init__(self):
        download_nltk_data()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.model = MultinomialNB()
        self.is_trained = False
        self._train()

    def _train(self):
        """Train on NLTK movie reviews corpus."""
        print("🏋️  Training Naive Bayes on movie reviews...")

        # Load movie reviews
        neg_reviews = [
            movie_reviews.raw(fid) for fid in movie_reviews.fileids("neg")
        ]
        pos_reviews = [
            movie_reviews.raw(fid) for fid in movie_reviews.fileids("pos")
        ]

        texts = neg_reviews + pos_reviews
        labels = [0] * len(neg_reviews) + [1] * len(pos_reviews)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels,
        )

        # Vectorize
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"   Training samples: {len(X_train)}")
        print(f"   Test accuracy:    {accuracy * 100:.1f}%\n")

        self.is_trained = True

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using trained Naive Bayes.

        Returns:
            Dict with 'label', 'confidence', 'probabilities', 'emoji'.
        """
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = float(max(probs))
        emoji = "😊" if pred == 1 else "😞"

        return {
            "label": label,
            "confidence": confidence,
            "prob_negative": float(probs[0]),
            "prob_positive": float(probs[1]),
            "emoji": emoji,
        }


def display_vader_result(text: str, result: dict):
    """Display VADER analysis results."""
    print(f"\n  📊 VADER (Rule-Based):")
    print(f"  {result['emoji']} Sentiment: {result['label']}")
    print(f"  📈 Compound Score: {result['compound']:.3f}")

    # Visual breakdown
    pos_bar = "█" * int(result["positive"] * 30)
    neg_bar = "█" * int(result["negative"] * 30)
    neu_bar = "█" * int(result["neutral"] * 30)
    print(f"  + Positive: {pos_bar} {result['positive']*100:.1f}%")
    print(f"  - Negative: {neg_bar} {result['negative']*100:.1f}%")
    print(f"  = Neutral:  {neu_bar} {result['neutral']*100:.1f}%")


def display_nb_result(text: str, result: dict):
    """Display Naive Bayes analysis results."""
    print(f"\n  🤖 Naive Bayes (ML-Based):")
    print(f"  {result['emoji']} Sentiment: {result['label']}")
    print(f"  📈 Confidence: {result['confidence']*100:.1f}%")

    pos_bar = "█" * int(result["prob_positive"] * 30)
    neg_bar = "█" * int(result["prob_negative"] * 30)
    print(f"  + Positive: {pos_bar} {result['prob_positive']*100:.1f}%")
    print(f"  - Negative: {neg_bar} {result['prob_negative']*100:.1f}%")


SAMPLE_TEXTS = [
    "This movie is absolutely wonderful! The acting was superb and the story was captivating.",
    "Terrible film. Waste of time and money. The plot made no sense at all.",
    "The product works okay, nothing special but gets the job done.",
    "I love this restaurant! The food is amazing and the service is excellent.",
    "Worst experience ever. Rude staff, cold food, and overpriced.",
    "The book was interesting but a bit too long in some parts.",
]


def main():
    parser = argparse.ArgumentParser(description="📝 Sentiment Analysis")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare both methods on sample texts")
    args = parser.parse_args()

    print("=" * 55)
    print("  📝 NLP — Sentiment Analysis")
    print("  📊 VADER (Rule-Based) + Naive Bayes (ML-Based)")
    print("=" * 55 + "\n")

    # Initialize analyzers
    vader = VaderAnalyzer()
    nb = NaiveBayesAnalyzer()

    if args.compare:
        # Compare both methods on samples
        print("📋 Comparing Methods on Sample Texts:\n")
        print(f"{'Text':<55} {'VADER':<12} {'NB':<12}")
        print("─" * 80)
        for text in SAMPLE_TEXTS:
            v_result = vader.analyze(text)
            nb_result = nb.analyze(text)
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f"{preview:<55} {v_result['emoji']} {v_result['label']:<10} "
                  f"{nb_result['emoji']} {nb_result['label']:<10}")
        print()

    elif args.text:
        # Analyze single text
        print(f"📝 Text: {args.text}\n")
        v_result = vader.analyze(args.text)
        display_vader_result(args.text, v_result)
        nb_result = nb.analyze(args.text)
        display_nb_result(args.text, nb_result)

    else:
        # Interactive mode
        print("🎮 Interactive Mode — Type text to analyze sentiment.")
        print("   Type 'quit' to exit.\n")

        while True:
            try:
                text = input("📝 Text: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Goodbye!")
                break

            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            v_result = vader.analyze(text)
            display_vader_result(text, v_result)
            nb_result = nb.analyze(text)
            display_nb_result(text, nb_result)
            print()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
