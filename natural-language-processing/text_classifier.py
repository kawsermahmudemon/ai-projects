"""
📝 NLP — Text Classifier
===========================
Multi-class text classification using TF-IDF + SVM on the 20 Newsgroups dataset.
Classifies text into topic categories like sports, technology, politics, etc.

Usage:
    python text_classifier.py                                       # Train and evaluate
    python text_classifier.py --predict "Text to classify..."       # Classify custom text
    python text_classifier.py --categories sci.med sci.space         # Specific categories
"""

import argparse

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# Simplified category groups for a cleaner demo
DEFAULT_CATEGORIES = [
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "comp.graphics",
    "talk.politics.guns",
    "soc.religion.christian",
    "misc.forsale",
    "rec.autos",
]

CATEGORY_LABELS = {
    "rec.sport.baseball": "⚾ Baseball",
    "sci.med": "🏥 Medicine",
    "sci.space": "🚀 Space",
    "comp.graphics": "🖥️ Graphics",
    "talk.politics.guns": "🔫 Politics/Guns",
    "soc.religion.christian": "⛪ Religion",
    "misc.forsale": "🏷️ For Sale",
    "rec.autos": "🚗 Automobiles",
}


def load_data(categories: list[str] = None):
    """Load the 20 Newsgroups dataset."""
    if categories is None:
        categories = DEFAULT_CATEGORIES

    print("📥 Loading 20 Newsgroups dataset...")
    train = fetch_20newsgroups(subset="train", categories=categories, random_state=42)
    test = fetch_20newsgroups(subset="test", categories=categories, random_state=42)

    print(f"   Training samples: {len(train.data)}")
    print(f"   Test samples:     {len(test.data)}")
    print(f"   Categories:       {len(train.target_names)}\n")

    # Category distribution
    print("   📊 Category Distribution (Train):")
    for i, name in enumerate(train.target_names):
        count = np.sum(train.target == i)
        label = CATEGORY_LABELS.get(name, name)
        print(f"     {label:<25} {count:>5} samples")
    print()

    return train, test


def build_pipeline():
    """Build a TF-IDF + SVM classification pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams + bigrams
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=42,
        )),
    ])
    return pipeline


def plot_confusion_matrix(y_true, y_pred, labels, save_path="newsgroups_confusion.png"):
    """Plot confusion matrix heatmap."""
    if not HAS_PLOT:
        return

    cm = confusion_matrix(y_true, y_pred)
    short_labels = [CATEGORY_LABELS.get(l, l).split(" ", 1)[-1] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Text Classification — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()


def predict_text(pipeline, text: str, target_names: list[str]) -> dict:
    """Classify a single text and show results."""
    prediction = pipeline.predict([text])[0]
    category = target_names[prediction]

    # Get decision function scores for ranking
    scores = pipeline.decision_function([text])[0]
    # Normalize to pseudo-probabilities
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    ranked = sorted(
        zip(target_names, probs),
        key=lambda x: x[1], reverse=True,
    )

    return {
        "predicted": category,
        "label": CATEGORY_LABELS.get(category, category),
        "ranked": ranked,
    }


SAMPLE_PREDICTIONS = [
    "NASA launched a new spacecraft to Mars carrying scientific instruments to study the red planet's atmosphere.",
    "The pitcher threw a perfect game last night, striking out 12 batters in the process.",
    "New advances in gene therapy show promising results for treating hereditary diseases.",
    "The new GPU can render 3D graphics at 120 frames per second with ray tracing enabled.",
    "Looking to sell my 2019 Honda Civic, low mileage, excellent condition, $15,000 OBO.",
    "The new Toyota Supra has a twin-turbo inline-six engine producing 382 horsepower.",
]


def main():
    parser = argparse.ArgumentParser(description="📝 Text Classifier (20 Newsgroups)")
    parser.add_argument("--predict", type=str, default=None, help="Text to classify")
    parser.add_argument("--categories", nargs="+", default=None, help="Specific categories to use")
    args = parser.parse_args()

    print("=" * 55)
    print("  📝 NLP — Text Classification")
    print("  📦 TF-IDF + SVM on 20 Newsgroups")
    print("=" * 55 + "\n")

    # Load data
    train, test = load_data(args.categories)

    # Build and train
    print("🔨 Building TF-IDF + SVM pipeline...")
    pipe = build_pipeline()

    print("🏋️  Training classifier...")
    pipe.fit(train.data, train.target)
    print("✅ Training complete!\n")

    # Evaluate
    print("📏 Evaluating on test set...\n")
    y_pred = pipe.predict(test.data)
    accuracy = accuracy_score(test.target, y_pred)

    print(f"{'=' * 50}")
    print(f"  ✅ Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 50}\n")

    # Classification report
    print("📋 Per-Category Classification Report:\n")
    short_names = [CATEGORY_LABELS.get(n, n).split(" ", 1)[-1] for n in test.target_names]
    report = classification_report(test.target, y_pred, target_names=short_names, digits=4)
    print(report)

    # Cross-validation
    print("🔄 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(pipe, train.data, train.target, cv=5, scoring="accuracy")
    print(f"   CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%\n")

    # Confusion matrix
    plot_confusion_matrix(test.target, y_pred, test.target_names)

    # Predictions
    if args.predict:
        print(f"\n📝 Classifying: \"{args.predict[:80]}{'...' if len(args.predict) > 80 else ''}\"\n")
        result = predict_text(pipe, args.predict, list(test.target_names))
        print(f"   🏷️ Predicted: {result['label']}\n")
        print("   Rankings:")
        for name, prob in result["ranked"]:
            label = CATEGORY_LABELS.get(name, name)
            bar = "█" * int(prob * 30)
            print(f"     {label:<25} {bar} {prob*100:.1f}%")
    else:
        # Demo predictions
        print("\n🔮 Sample Predictions:")
        print("─" * 70)
        for text in SAMPLE_PREDICTIONS:
            result = predict_text(pipe, text, list(test.target_names))
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"   {result['label']:<25} │ {preview}")
        print("─" * 70)

    print("\n✅ Text classification complete!")


if __name__ == "__main__":
    main()
