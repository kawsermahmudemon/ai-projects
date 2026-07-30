"""
📚 LLM — Text Generator (GPT-2)
==================================
Interactive text generation using a pre-trained GPT-2 model via Hugging Face Transformers.
Runs on CPU with the small GPT-2 model (~500MB download on first run).

Usage:
    python text_generator.py                                   # Interactive mode
    python text_generator.py --prompt "Once upon a time"       # Single generation
    python text_generator.py --temperature 0.7 --max-length 200
"""

import argparse
import sys


def load_generator(model_name: str = "gpt2"):
    """
    Load the GPT-2 text generation pipeline.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Transformers text-generation pipeline.
    """
    from transformers import pipeline, set_seed

    print(f"🔄 Loading model: {model_name}")
    print("   (First run will download the model ~500MB...)\n")

    generator = pipeline(
        "text-generation",
        model=model_name,
        device=-1,  # CPU
    )

    print(f"✅ Model loaded: {model_name}\n")
    return generator


def generate_text(
    generator,
    prompt: str,
    max_length: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
) -> list[str]:
    """
    Generate text from a prompt.

    Args:
        generator: Hugging Face pipeline.
        prompt: Input text to continue.
        max_length: Maximum total length (prompt + generated).
        temperature: Randomness control (0.1=conservative, 2.0=creative).
        top_k: Only sample from top-k most likely tokens.
        top_p: Nucleus sampling threshold.
        num_return_sequences: Number of completions to generate.

    Returns:
        List of generated text strings.
    """
    results = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    return [r["generated_text"] for r in results]


def interactive_mode(generator, args):
    """Run interactive text generation loop."""
    print("🎮 Interactive Mode")
    print("   Type a prompt and press Enter to generate.")
    print("   Type 'quit' to exit.\n")
    print(f"   Settings: temp={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}, max_length={args.max_length}\n")

    while True:
        try:
            prompt = input("📝 Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        print("\n🔄 Generating...\n")
        texts = generate_text(
            generator, prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences,
        )

        for i, text in enumerate(texts, 1):
            if args.num_sequences > 1:
                print(f"── Completion {i} {'─' * 40}")
            print(text)
            print()


def main():
    parser = argparse.ArgumentParser(
        description="📚 GPT-2 Text Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text_generator.py                                    # Interactive
  python text_generator.py --prompt "The future of AI is"     # Single prompt
  python text_generator.py --temperature 0.3 --max-length 300 # Conservative, long
  python text_generator.py --num-sequences 3                  # 3 completions
        """,
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (if omitted, enters interactive mode)")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--max-length", type=int, default=150, help="Max total tokens (default: 150)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold (default: 0.95)")
    parser.add_argument("--num-sequences", type=int, default=1, help="Number of completions (default: 1)")
    args = parser.parse_args()

    print("=" * 55)
    print("  📚 LLM — GPT-2 Text Generator")
    print("  📦 Model: " + args.model)
    print("=" * 55 + "\n")

    generator = load_generator(args.model)

    if args.prompt:
        print(f"📝 Prompt: {args.prompt}\n")
        print("🔄 Generating...\n")
        texts = generate_text(
            generator, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences,
        )
        for i, text in enumerate(texts, 1):
            if args.num_sequences > 1:
                print(f"── Completion {i} {'─' * 40}")
            print(text)
            print()
    else:
        interactive_mode(generator, args)

    print("✅ Done!")


if __name__ == "__main__":
    main()
