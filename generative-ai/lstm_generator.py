"""
✨ Generative AI — Character-Level LSTM Text Generator
=======================================================
Train a character-level LSTM neural network on a text corpus and generate
creative text using temperature-controlled sampling.

Usage:
    python lstm_generator.py                            # Train + generate (default corpus)
    python lstm_generator.py --file text.txt            # Custom corpus
    python lstm_generator.py --epochs 30                # More training epochs
    python lstm_generator.py --generate-only            # Skip training, load saved model
    python lstm_generator.py --temperature 0.5          # Lower temp = more conservative
"""

import argparse
import os
import sys

import numpy as np


def load_corpus(filepath: str) -> str:
    """Load and return the text corpus."""
    if not os.path.exists(filepath):
        print(f"❌ Corpus not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📚 Loaded corpus: {len(text):,} characters")
    return text


def prepare_data(text: str, seq_length: int = 40):
    """
    Prepare training data for the LSTM.

    Converts text to sequences of character indices.

    Args:
        text: The training text.
        seq_length: Length of input sequences.

    Returns:
        Tuple of (X, y, char_to_idx, idx_to_char, chars).
    """
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"   Unique characters: {len(chars)}")
    print(f"   Sequence length:   {seq_length}")

    # Build sequences
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i : i + seq_length])
        next_chars.append(text[i + seq_length])

    print(f"   Training sequences: {len(sequences):,}")

    # Vectorize
    X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.float32)
    y = np.zeros((len(sequences), len(chars)), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1.0
        y[i, char_to_idx[next_chars[i]]] = 1.0

    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}\n")

    return X, y, char_to_idx, idx_to_char, chars


def build_lstm_model(seq_length: int, num_chars: int):
    """
    Build a character-level LSTM model.

    Architecture:
        LSTM(128) → Dropout(0.2) → Dense(num_chars, softmax)
    """
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.LSTM(128, input_shape=(seq_length, num_chars), name="lstm"),
        layers.Dropout(0.2, name="dropout"),
        layers.Dense(num_chars, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def sample_with_temperature(preds: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a character index from predictions with temperature control.

    Args:
        preds: Probability distribution over characters.
        temperature: Controls randomness.
            - Low (0.2-0.5): More conservative, repetitive but coherent.
            - Medium (0.8-1.0): Balanced creativity and coherence.
            - High (1.2-2.0): More random and creative.

    Returns:
        Sampled character index.
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.clip(preds, 1e-10, 1.0)  # Prevent log(0)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return int(np.random.choice(len(preds), p=preds))


def generate_text(
    model, seed_text: str, char_to_idx: dict, idx_to_char: dict,
    seq_length: int, num_chars: int, length: int = 300,
    temperature: float = 0.8,
) -> str:
    """
    Generate text using the trained LSTM model.

    Args:
        model: Trained Keras model.
        seed_text: Starting text (must be seq_length characters).
        char_to_idx: Character to index mapping.
        idx_to_char: Index to character mapping.
        seq_length: Sequence length the model expects.
        num_chars: Total unique characters.
        length: Number of characters to generate.
        temperature: Sampling temperature.

    Returns:
        Generated text string.
    """
    generated = seed_text
    current = seed_text

    for _ in range(length):
        # Encode current sequence
        x = np.zeros((1, seq_length, num_chars), dtype=np.float32)
        for t, char in enumerate(current):
            if char in char_to_idx:
                x[0, t, char_to_idx[char]] = 1.0

        # Predict next character
        preds = model.predict(x, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]

        generated += next_char
        current = current[1:] + next_char

    return generated


def main():
    parser = argparse.ArgumentParser(description="✨ LSTM Text Generator")
    parser.add_argument("--file", type=str, default=None, help="Path to training corpus")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--seq-length", type=int, default=40, help="Sequence length (default: 40)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--length", type=int, default=300, help="Characters to generate (default: 300)")
    parser.add_argument("--generate-only", action="store_true", help="Load saved model and generate")
    parser.add_argument("--model-path", type=str, default="models/lstm_generator.keras", help="Model save path")
    args = parser.parse_args()

    print("=" * 55)
    print("  ✨ Generative AI — LSTM Text Generator")
    print("=" * 55 + "\n")

    # Default corpus
    if args.file is None:
        args.file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "sample_corpus.txt"
        )

    # Load corpus
    text = load_corpus(args.file)
    X, y, char_to_idx, idx_to_char, chars = prepare_data(text, args.seq_length)

    if args.generate_only:
        # Load existing model
        from tensorflow.keras.models import load_model as keras_load_model
        if not os.path.exists(args.model_path):
            print(f"❌ No saved model at: {args.model_path}")
            print("   Run without --generate-only to train first.")
            sys.exit(1)
        model = keras_load_model(args.model_path)
        print(f"✅ Loaded model from: {args.model_path}\n")
    else:
        # Build and train
        print("🔨 Building LSTM model...")
        model = build_lstm_model(args.seq_length, len(chars))
        model.summary()
        print()

        print(f"🏋️ Training for {args.epochs} epochs...\n")
        history = model.fit(
            X, y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            verbose=1,
        )

        # Save model
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        print(f"\n💾 Model saved to: {args.model_path}\n")

    # Generate text at different temperatures
    # Pick a random seed from the corpus
    start_idx = np.random.randint(0, len(text) - args.seq_length)
    seed = text[start_idx : start_idx + args.seq_length]

    print(f"🌱 Seed text: \"{seed}\"\n")

    temperatures = [0.3, 0.8, 1.2] if not args.generate_only else [args.temperature]

    for temp in temperatures:
        print(f"── Temperature: {temp} {'─' * 38}")
        generated = generate_text(
            model, seed, char_to_idx, idx_to_char,
            args.seq_length, len(chars),
            length=args.length, temperature=temp,
        )
        # Print only the generated part (after seed)
        print(f"{generated[len(seed):]}\n")

    print("✅ Generation complete!")


if __name__ == "__main__":
    main()
