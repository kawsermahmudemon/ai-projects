"""
📚 LLM — Sentiment Analyzer
==============================
Sentiment analysis using a pre-trained DistilBERT model fine-tuned on SST-2.
Analyzes text and returns positive/negative sentiment with confidence scores.

Usage:
    python sentiment_analyzer.py                                    # Interactive mode
    python sentiment_analyzer.py --text "This movie is amazing!"    # Single text
    python sentiment_analyzer.py --file reviews.txt                 # Analyze file
"""

import argparse
import sys


def load_analyzer(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Load the sentiment analysis pipeline.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Transformers sentiment-analysis pipeline.
    """
    from transformers import pipeline

    print(f"🔄 Loading model: {model_name}")
    print("   (First run will download ~260MB...)\n")

    analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=-1,  # CPU
    )

    print("✅ Model loaded!\n")
    return analyzer


def analyze_text(analyzer, text: str) -> dict:
    """
    Analyze sentiment of a single text.

    Args:
        analyzer: Hugging Face pipeline.
        text: Text to analyze.

    Returns:
        Dict with 'label', 'score', and formatted 'display'.
    """
    result = analyzer(text, truncation=True, max_length=512)[0]

    label = result["label"]
    score = result["score"]

    emoji = "😊" if label == "POSITIVE" else "😞"
    bar_length = int(score * 30)
    bar = "█" * bar_length + "░" * (30 - bar_length)

    return {
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        "label": label,
        "score": score,
        "emoji": emoji,
        "bar": bar,
    }


def analyze_file(analyzer, filepath: str) -> list[dict]:
    """Analyze each line of a text file."""
    import os

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"📄 Analyzing {len(lines)} lines from: {filepath}\n")

    results = []
    for i, line in enumerate(lines, 1):
        result = analyze_text(analyzer, line)
        results.append(result)
        print(f"  {i:3d}. {result['emoji']} {result['label']:8s} "
              f"{result['bar']} {result['score']*100:.1f}%  │ {result['text']}")

    return results


def interactive_mode(analyzer):
    """Run interactive sentiment analysis loop."""
    print("🎮 Interactive Mode")
    print("   Type text and press Enter to analyze sentiment.")
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

        result = analyze_text(analyzer, text)

        print(f"\n   {result['emoji']} Sentiment: {result['label']}")
        print(f"   📊 Confidence: {result['bar']} {result['score']*100:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="📚 Sentiment Analyzer (DistilBERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", type=str, default=None, help="Text to analyze")
    parser.add_argument("--file", type=str, default=None, help="Path to text file (one text per line)")
    parser.add_argument("--model", type=str,
                        default="distilbert-base-uncased-finetuned-sst-2-english",
                        help="Model name")
    args = parser.parse_args()

    print("=" * 55)
    print("  📚 LLM — Sentiment Analyzer")
    print("  📦 Model: DistilBERT (SST-2)")
    print("=" * 55 + "\n")

    analyzer = load_analyzer(args.model)

    if args.text:
        result = analyze_text(analyzer, args.text)
        print(f"📝 Text: {args.text}\n")
        print(f"   {result['emoji']} Sentiment: {result['label']}")
        print(f"   📊 Confidence: {result['bar']} {result['score']*100:.1f}%")
    elif args.file:
        results = analyze_file(analyzer, args.file)
        # Summary
        positive = sum(1 for r in results if r["label"] == "POSITIVE")
        negative = len(results) - positive
        print(f"\n📊 Summary: {positive} positive, {negative} negative "
              f"({positive/len(results)*100:.0f}% positive)")
    else:
        interactive_mode(analyzer)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
