"""
📚 LLM — Text Summarizer
===========================
Text summarization using a pre-trained BART model via Hugging Face Transformers.
Takes long text and produces concise summaries.

Usage:
    python summarizer.py                                           # Interactive mode
    python summarizer.py --text "Long article text..."             # Single text
    python summarizer.py --file article.txt                        # Summarize file
    python summarizer.py --max-length 100 --min-length 30          # Control length
"""

import argparse
import os
import sys


def load_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    """
    Load the summarization pipeline.

    Uses DistilBART (a smaller, faster version of BART) by default.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Transformers summarization pipeline.
    """
    from transformers import pipeline

    print(f"🔄 Loading model: {model_name}")
    print("   (First run will download ~1.2GB...)\n")

    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=-1,  # CPU
    )

    print("✅ Model loaded!\n")
    return summarizer


def summarize_text(
    summarizer,
    text: str,
    max_length: int = 130,
    min_length: int = 30,
) -> dict:
    """
    Summarize a text.

    Args:
        summarizer: Hugging Face pipeline.
        text: Text to summarize.
        max_length: Maximum summary length in tokens.
        min_length: Minimum summary length in tokens.

    Returns:
        Dict with 'summary', 'original_length', 'summary_length', 'compression_ratio'.
    """
    # Truncate very long texts to model's max input
    max_input = 1024
    words = text.split()
    if len(words) > max_input:
        text = " ".join(words[:max_input])
        print(f"   ⚠️ Input truncated to {max_input} words.")

    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )[0]

    summary = result["summary_text"]
    original_words = len(text.split())
    summary_words = len(summary.split())

    return {
        "summary": summary,
        "original_length": original_words,
        "summary_length": summary_words,
        "compression_ratio": round(summary_words / original_words * 100, 1) if original_words > 0 else 0,
    }


def summarize_file(summarizer, filepath: str, max_length: int = 130, min_length: int = 30) -> dict:
    """Summarize text from a file."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"📄 Summarizing file: {filepath}")
    print(f"   Original: {len(text.split())} words\n")

    return summarize_text(summarizer, text, max_length, min_length)


def interactive_mode(summarizer, args):
    """Run interactive summarization loop."""
    print("🎮 Interactive Mode")
    print("   Paste text and press Enter twice to summarize.")
    print("   Type 'quit' to exit.\n")

    while True:
        try:
            print("📝 Paste text (press Enter twice when done):")
            lines = []
            empty_count = 0
            while True:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                    lines.append(line)
                else:
                    empty_count = 0
                    lines.append(line)

            text = "\n".join(lines).strip()

            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                break

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if len(text.split()) < 10:
            print("⚠️ Text is too short to summarize. Please provide longer text.\n")
            continue

        print("\n🔄 Summarizing...\n")
        result = summarize_text(summarizer, text, args.max_length, args.min_length)

        print(f"📋 Summary ({result['summary_length']} words, "
              f"{result['compression_ratio']}% of original):\n")
        print(f"   {result['summary']}\n")


SAMPLE_TEXT = """
Artificial intelligence (AI) has become one of the most transformative technologies 
of the 21st century, impacting virtually every industry and aspect of human life. 
From healthcare to transportation, education to entertainment, AI systems are being 
deployed at an unprecedented scale. Machine learning, a subset of AI, enables computers 
to learn from data without being explicitly programmed, while deep learning uses neural 
networks with multiple layers to analyze complex patterns in large datasets.

Recent advances in large language models (LLMs) have demonstrated remarkable capabilities 
in natural language understanding and generation. Models like GPT-4, Claude, and Gemini 
can write code, analyze documents, create content, and engage in sophisticated 
conversations. These models are trained on vast amounts of text data and use the 
transformer architecture, which was introduced in the landmark "Attention Is All You 
Need" paper in 2017.

The development of AI also raises important ethical considerations, including bias in 
training data, privacy concerns, job displacement, and the need for responsible AI 
governance. Researchers and policymakers worldwide are working to establish frameworks 
that ensure AI is developed and deployed in ways that benefit humanity while minimizing 
potential risks.
"""


def main():
    parser = argparse.ArgumentParser(
        description="📚 Text Summarizer (DistilBART)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", type=str, default=None, help="Text to summarize")
    parser.add_argument("--file", type=str, default=None, help="Path to text file to summarize")
    parser.add_argument("--model", type=str, default="sshleifer/distilbart-cnn-12-6", help="Model name")
    parser.add_argument("--max-length", type=int, default=130, help="Max summary length in tokens (default: 130)")
    parser.add_argument("--min-length", type=int, default=30, help="Min summary length in tokens (default: 30)")
    parser.add_argument("--demo", action="store_true", help="Run with built-in sample text")
    args = parser.parse_args()

    print("=" * 55)
    print("  📚 LLM — Text Summarizer")
    print("  📦 Model: DistilBART")
    print("=" * 55 + "\n")

    summarizer = load_summarizer(args.model)

    if args.demo:
        print("📄 Using built-in sample text about AI...\n")
        print("─" * 55)
        print(SAMPLE_TEXT.strip())
        print("─" * 55 + "\n")
        result = summarize_text(summarizer, SAMPLE_TEXT, args.max_length, args.min_length)
    elif args.text:
        result = summarize_text(summarizer, args.text, args.max_length, args.min_length)
    elif args.file:
        result = summarize_file(summarizer, args.file, args.max_length, args.min_length)
    else:
        interactive_mode(summarizer, args)
        print("\n✅ Done!")
        return

    print(f"📋 Summary ({result['summary_length']} words, "
          f"{result['compression_ratio']}% of original):\n")
    print(f"   {result['summary']}")
    print(f"\n📊 Compression: {result['original_length']} → {result['summary_length']} words")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
