"""
✨ Generative AI — Markov Chain Text Generator
================================================
Generate novel text using an n-gram Markov chain model.
Learns word transition probabilities from a text corpus and generates
new text that mimics the style of the training data.

Usage:
    python markov_generator.py                                    # Use default corpus
    python markov_generator.py --file path/to/text.txt            # Custom corpus
    python markov_generator.py --order 3 --length 100             # 3-gram, 100 words
    python markov_generator.py --seed "To be or"                  # Start with seed phrase
    python markov_generator.py --interactive                      # Interactive mode
"""

import argparse
import os
import random
import re
from collections import defaultdict


class MarkovChain:
    """
    N-gram Markov chain text generator.

    Builds a probability model of word sequences from training text,
    then generates new text by sampling from learned transitions.
    """

    def __init__(self, order: int = 2):
        """
        Initialize the Markov chain.

        Args:
            order: The n-gram order (number of preceding words to consider).
                   Higher order = more coherent but less creative.
        """
        self.order = order
        self.chain: dict[tuple, list[str]] = defaultdict(list)
        self.start_tokens: list[tuple] = []
        self.word_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """Split text into words while preserving punctuation."""
        # Split on whitespace, keep punctuation attached
        words = text.split()
        return [w for w in words if w]

    def train(self, text: str):
        """
        Train the Markov chain on a text corpus.

        Args:
            text: The training text.
        """
        words = self._tokenize(text)
        self.word_count = len(words)

        if len(words) <= self.order:
            print("⚠️  Text too short for the specified order.")
            return

        print(f"📚 Training on {self.word_count:,} words with order={self.order}...")

        for i in range(len(words) - self.order):
            key = tuple(words[i : i + self.order])
            next_word = words[i + self.order]
            self.chain[key].append(next_word)

            # Track sentence starts (after period, question mark, etc.)
            if i == 0 or words[i - 1][-1] in ".!?":
                self.start_tokens.append(key)

        # Ensure we have start tokens
        if not self.start_tokens:
            self.start_tokens = list(self.chain.keys())[:10]

        print(f"   Unique {self.order}-grams: {len(self.chain):,}")
        print(f"   Start tokens:    {len(self.start_tokens):,}")
        print()

    def train_from_file(self, filepath: str):
        """Train the model from a text file."""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        self.train(text)

    def generate(self, length: int = 50, seed: str = None) -> str:
        """
        Generate text using the trained Markov chain.

        Args:
            length: Number of words to generate.
            seed: Optional seed phrase to start generation.
                  Must match the order of the chain.

        Returns:
            Generated text string.
        """
        if not self.chain:
            return "⚠️  Model not trained yet. Call train() first."

        # Choose starting point
        if seed:
            seed_words = tuple(seed.split())
            if len(seed_words) >= self.order:
                current = seed_words[-self.order:]
            else:
                # Pad with random start
                current = random.choice(self.start_tokens)
                print(f"⚠️  Seed too short (need {self.order} words). Using random start.")
        else:
            current = random.choice(self.start_tokens)

        output = list(current)

        for _ in range(length - self.order):
            key = tuple(output[-self.order:])
            if key in self.chain:
                next_word = random.choice(self.chain[key])
                output.append(next_word)
            else:
                # Dead end — pick a random start
                restart = random.choice(self.start_tokens)
                output.extend(restart)

        return " ".join(output)

    def get_stats(self) -> dict:
        """Return model statistics."""
        total_transitions = sum(len(v) for v in self.chain.values())
        avg_choices = total_transitions / len(self.chain) if self.chain else 0
        return {
            "order": self.order,
            "unique_ngrams": len(self.chain),
            "total_transitions": total_transitions,
            "avg_choices_per_ngram": round(avg_choices, 2),
            "training_words": self.word_count,
        }


def main():
    parser = argparse.ArgumentParser(
        description="✨ Markov Chain Text Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", type=str, default=None, help="Path to training text file")
    parser.add_argument("--order", type=int, default=2, help="N-gram order (default: 2)")
    parser.add_argument("--length", type=int, default=50, help="Words to generate (default: 50)")
    parser.add_argument("--seed", type=str, default=None, help="Seed phrase to start generation")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    args = parser.parse_args()

    print("=" * 55)
    print("  ✨ Generative AI — Markov Chain Text Generator")
    print("=" * 55 + "\n")

    # Default corpus
    if args.file is None:
        args.file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "sample_corpus.txt"
        )

    # Build model
    model = MarkovChain(order=args.order)
    model.train_from_file(args.file)

    # Show stats
    stats = model.get_stats()
    print("📊 Model Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()

    if args.interactive:
        # Interactive mode
        print("🎮 Interactive Mode — Type a seed phrase or press Enter for random.\n"
              "   Type 'quit' to exit.\n")
        while True:
            try:
                seed = input("Seed: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if seed.lower() in ("quit", "exit", "q"):
                break
            generated = model.generate(length=args.length, seed=seed if seed else None)
            print(f"\n📝 Generated:\n{generated}\n")
    else:
        # Generate samples
        for i in range(args.samples):
            print(f"── Sample {i + 1} {'─' * 42}")
            generated = model.generate(length=args.length, seed=args.seed)
            print(f"{generated}\n")

    print("✅ Done!")


if __name__ == "__main__":
    main()
