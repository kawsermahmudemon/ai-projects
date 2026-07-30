"""
📝 NLP — Text Preprocessor
==============================
Demonstrates core NLP preprocessing techniques using NLTK:
tokenization, stopword removal, stemming, lemmatization, and TF-IDF vectorization.

Usage:
    python text_preprocessor.py                         # Demo with sample text
    python text_preprocessor.py --text "Your text here"  # Custom text
"""

import argparse
import re
import string

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def download_nltk_data():
    """Download required NLTK data packages."""
    packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


class TextPreprocessor:
    """A comprehensive text preprocessing pipeline."""

    def __init__(self, language: str = "english"):
        download_nltk_data()
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Remove special characters, extra whitespace, and lowercase."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
        text = re.sub(r"\S+@\S+", "", text)  # Emails
        text = re.sub(r"[^\w\s]", " ", text)  # Punctuation
        text = re.sub(r"\d+", "", text)  # Numbers
        text = re.sub(r"\s+", " ", text).strip()  # Extra whitespace
        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def sentence_tokenize(self, text: str) -> list[str]:
        """Split text into sentences."""
        return sent_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove common stop words."""
        return [t for t in tokens if t.lower() not in self.stop_words]

    def stem(self, tokens: list[str]) -> list[str]:
        """Apply Porter stemming to tokens."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Apply WordNet lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str, steps: list[str] = None) -> dict:
        """
        Run the full preprocessing pipeline.

        Args:
            text: Input text.
            steps: List of steps to run. Default: all steps.

        Returns:
            Dict with results from each step.
        """
        if steps is None:
            steps = ["clean", "tokenize", "stopwords", "stem", "lemmatize"]

        results = {"original": text}

        current_text = text

        if "clean" in steps:
            current_text = self.clean_text(current_text)
            results["cleaned"] = current_text

        tokens = self.tokenize(current_text)
        results["tokens"] = tokens
        results["token_count"] = len(tokens)

        if "stopwords" in steps:
            filtered = self.remove_stopwords(tokens)
            results["no_stopwords"] = filtered
            results["stopwords_removed"] = len(tokens) - len(filtered)

        if "stem" in steps:
            stemmed = self.stem(results.get("no_stopwords", tokens))
            results["stemmed"] = stemmed

        if "lemmatize" in steps:
            lemmatized = self.lemmatize(results.get("no_stopwords", tokens))
            results["lemmatized"] = lemmatized

        return results


def demo_tfidf(documents: list[str]):
    """Demonstrate TF-IDF vectorization on a list of documents."""
    print("\n📊 TF-IDF Vectorization Demo")
    print("─" * 50)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n   Documents:  {len(documents)}")
    print(f"   Vocabulary: {len(feature_names)} terms")
    print(f"   Matrix:     {tfidf_matrix.shape}")

    print(f"\n   Top terms by TF-IDF score:")
    for i, doc in enumerate(documents):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-5:][::-1]
        top_terms = [(feature_names[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        preview = doc[:60] + "..." if len(doc) > 60 else doc
        print(f"\n   Doc {i+1}: \"{preview}\"")
        for term, score in top_terms:
            bar = "█" * int(score * 30)
            print(f"     {term:<15} {bar} {score:.3f}")


SAMPLE_TEXT = """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze large 
amounts of natural language data. The result is a computer capable of understanding 
the contents of documents, including the contextual nuances of the language within them.
NLP combines computational linguistics with statistical, machine learning, and deep 
learning models. These technologies enable computers to process human language in the 
form of text or voice data and to understand its full meaning, complete with the 
speaker's or writer's intent and sentiment.
"""

SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Deep learning uses neural networks with multiple layers to analyze complex patterns in large datasets.",
    "Computer vision is the field of AI that enables machines to interpret and understand visual information.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors and punishing undesired ones.",
]


def main():
    parser = argparse.ArgumentParser(description="📝 NLP Text Preprocessor")
    parser.add_argument("--text", type=str, default=None, help="Text to preprocess")
    args = parser.parse_args()

    print("=" * 55)
    print("  📝 NLP — Text Preprocessing Pipeline")
    print("=" * 55 + "\n")

    preprocessor = TextPreprocessor()
    text = args.text or SAMPLE_TEXT.strip()

    print(f"📄 Original Text ({len(text)} chars):\n")
    print(f"   {text[:200]}{'...' if len(text) > 200 else ''}\n")

    # Run preprocessing
    results = preprocessor.preprocess(text)

    # Display results
    print("─" * 50)
    print("  Step 1: Text Cleaning")
    print("─" * 50)
    print(f"   {results['cleaned'][:200]}{'...' if len(results.get('cleaned', '')) > 200 else ''}\n")

    print("─" * 50)
    print("  Step 2: Tokenization")
    print("─" * 50)
    print(f"   Tokens ({results['token_count']}): {results['tokens'][:20]}{'...' if results['token_count'] > 20 else ''}\n")

    print("─" * 50)
    print("  Step 3: Stopword Removal")
    print("─" * 50)
    filtered = results.get("no_stopwords", [])
    print(f"   Removed {results.get('stopwords_removed', 0)} stopwords")
    print(f"   Remaining ({len(filtered)}): {filtered[:20]}{'...' if len(filtered) > 20 else ''}\n")

    print("─" * 50)
    print("  Step 4: Stemming (Porter)")
    print("─" * 50)
    stemmed = results.get("stemmed", [])
    print(f"   Stemmed ({len(stemmed)}): {stemmed[:20]}{'...' if len(stemmed) > 20 else ''}\n")

    print("─" * 50)
    print("  Step 5: Lemmatization (WordNet)")
    print("─" * 50)
    lemmatized = results.get("lemmatized", [])
    print(f"   Lemmatized ({len(lemmatized)}): {lemmatized[:20]}{'...' if len(lemmatized) > 20 else ''}\n")

    # Stem vs Lemmatize comparison
    print("─" * 50)
    print("  📊 Stemming vs. Lemmatization Comparison")
    print("─" * 50)
    print(f"   {'Original':<20} {'Stemmed':<20} {'Lemmatized':<20}")
    print(f"   {'─'*20} {'─'*20} {'─'*20}")
    for orig, stem, lemma in zip(filtered[:10], stemmed[:10], lemmatized[:10]):
        print(f"   {orig:<20} {stem:<20} {lemma:<20}")

    # TF-IDF demo
    demo_tfidf(SAMPLE_DOCUMENTS)

    # Sentence tokenization
    print("\n─" * 50)
    print("  ✂️  Sentence Tokenization")
    print("─" * 50)
    sentences = preprocessor.sentence_tokenize(text)
    for i, sent in enumerate(sentences, 1):
        print(f"   {i}. {sent.strip()}")

    print(f"\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
