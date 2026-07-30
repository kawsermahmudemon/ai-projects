"""
📝 NLP — Named Entity Recognition (NER)
==========================================
Extract named entities (persons, organizations, locations, dates, etc.)
from text using spaCy's pre-trained NER model.

Usage:
    python ner_extractor.py                                        # Demo with sample text
    python ner_extractor.py --text "Apple Inc. is based in Cupertino"
    python ner_extractor.py --file article.txt                     # Extract from file
"""

import argparse
import os
import sys
from collections import Counter

try:
    import spacy
except ImportError:
    print("❌ spaCy is required. Install it with:")
    print("   pip install spacy")
    print("   python -m spacy download en_core_web_sm")
    sys.exit(1)


# Entity type descriptions
ENTITY_DESCRIPTIONS = {
    "PERSON": "👤 Person",
    "ORG": "🏢 Organization",
    "GPE": "🌍 Country/City/State",
    "LOC": "📍 Location",
    "DATE": "📅 Date",
    "TIME": "🕐 Time",
    "MONEY": "💰 Money",
    "PERCENT": "📊 Percentage",
    "PRODUCT": "📦 Product",
    "EVENT": "🎉 Event",
    "WORK_OF_ART": "🎨 Work of Art",
    "LAW": "⚖️ Law",
    "LANGUAGE": "🗣️ Language",
    "NORP": "👥 Group/Nationality",
    "FAC": "🏗️ Facility",
    "QUANTITY": "📏 Quantity",
    "ORDINAL": "🔢 Ordinal",
    "CARDINAL": "🔢 Cardinal Number",
}


def load_model(model_name: str = "en_core_web_sm"):
    """Load a spaCy NLP model."""
    try:
        nlp = spacy.load(model_name)
        print(f"✅ Loaded spaCy model: {model_name}\n")
        return nlp
    except OSError:
        print(f"⚠️  Model '{model_name}' not found. Downloading...")
        os.system(f"python -m spacy download {model_name}")
        try:
            nlp = spacy.load(model_name)
            print(f"✅ Model downloaded and loaded: {model_name}\n")
            return nlp
        except OSError:
            print(f"❌ Failed to load model '{model_name}'.")
            print(f"   Try: python -m spacy download {model_name}")
            sys.exit(1)


def extract_entities(nlp, text: str) -> list[dict]:
    """
    Extract named entities from text.

    Args:
        nlp: spaCy NLP model.
        text: Input text.

    Returns:
        List of entity dicts with 'text', 'label', 'start', 'end', 'description'.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "description": ENTITY_DESCRIPTIONS.get(ent.label_, f"❓ {ent.label_}"),
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def display_entities(entities: list[dict], text: str = None):
    """Display extracted entities in a formatted table."""
    if not entities:
        print("   No entities found.\n")
        return

    # Group by type
    by_type: dict[str, list[str]] = {}
    for ent in entities:
        label = ent["label"]
        if label not in by_type:
            by_type[label] = []
        if ent["text"] not in by_type[label]:
            by_type[label].append(ent["text"])

    # Display grouped
    print(f"   Found {len(entities)} entities across {len(by_type)} types:\n")
    for label, texts in sorted(by_type.items()):
        desc = ENTITY_DESCRIPTIONS.get(label, f"❓ {label}")
        print(f"   {desc}:")
        for t in texts:
            print(f"      • {t}")
        print()

    # Summary table
    print("   ─" * 20)
    print(f"   {'Type':<12} {'Count':<8} {'Examples'}")
    print("   ─" * 20)
    for label, texts in sorted(by_type.items(), key=lambda x: -len(x[1])):
        examples = ", ".join(texts[:3])
        if len(texts) > 3:
            examples += f" (+{len(texts) - 3} more)"
        print(f"   {label:<12} {len(texts):<8} {examples}")
    print()


def annotate_text(text: str, entities: list[dict]) -> str:
    """Create an annotated version of text with entity labels inline."""
    # Sort entities by start position (reverse) to avoid offset issues
    sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)
    annotated = text
    for ent in sorted_ents:
        annotated = (
            annotated[:ent["start"]]
            + f"[{ent['text']}]({ent['label']})"
            + annotated[ent["end"]:]
        )
    return annotated


SAMPLE_TEXT = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 
1976, in Cupertino, California. The company has grown to become one of the most 
valuable corporations in the world, with a market capitalization exceeding $3 trillion 
as of January 2024.

Google, a subsidiary of Alphabet Inc., was founded by Larry Page and Sergey Brin 
while they were Ph.D. students at Stanford University in September 1998. The company 
is headquartered in Mountain View, California.

Elon Musk serves as CEO of Tesla and SpaceX. Tesla was incorporated in July 2003 
in Delaware by Martin Eberhard and Marc Tarpenning. SpaceX, founded in 2002, has 
launched numerous missions to the International Space Station for NASA.

The European Union announced new artificial intelligence regulations in Brussels 
on March 13, 2024, allocating €1.5 billion in funding for AI research initiatives 
across member states including Germany, France, and Italy.
"""


def main():
    parser = argparse.ArgumentParser(
        description="📝 Named Entity Recognition (NER)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", type=str, default=None, help="Text to analyze")
    parser.add_argument("--file", type=str, default=None, help="Path to text file")
    parser.add_argument("--model", type=str, default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--annotate", action="store_true", help="Show annotated text")
    args = parser.parse_args()

    print("=" * 55)
    print("  📝 NLP — Named Entity Recognition")
    print("  🏷️  Extract people, orgs, places, dates & more")
    print("=" * 55 + "\n")

    nlp = load_model(args.model)

    # Get input text
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"📄 Loaded file: {args.file} ({len(text)} chars)\n")
    elif args.text:
        text = args.text
    else:
        text = SAMPLE_TEXT.strip()
        print("📄 Using sample text (tech companies):\n")
        print(f"   {text[:200]}...\n")

    # Extract entities
    print("🔍 Extracting named entities...\n")
    entities = extract_entities(nlp, text)
    display_entities(entities, text)

    # Annotated text
    if args.annotate:
        print("📝 Annotated Text:")
        print("─" * 50)
        annotated = annotate_text(text, entities)
        print(annotated)
        print("─" * 50)

    print("✅ NER extraction complete!")


if __name__ == "__main__":
    main()
