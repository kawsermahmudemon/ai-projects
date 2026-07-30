"""
📥 Download Sample Images
===========================
Downloads free sample images for testing the classifier.
Images are sourced from Wikimedia Commons (public domain / CC licensed).

Usage:
    python download_samples.py
"""

import os
import urllib.request

SAMPLE_IMAGES = {
    "sample_dog.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        "description": "Yellow Labrador Retriever",
    },
    "sample_cat.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "description": "Tabby Cat",
    },
    "sample_bird.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/1200px-Eopsaltria_australis_-_Mogo_Campground.jpg",
        "description": "Eastern Yellow Robin",
    },
}


def download_samples(output_dir: str = None):
    """Download all sample images."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    print("📥 Downloading sample images for the classifier...\n")

    for filename, info in SAMPLE_IMAGES.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✅ {filename} — already exists ({info['description']})")
            continue

        try:
            print(f"  🔄 Downloading {filename} ({info['description']})...")
            urllib.request.urlretrieve(info["url"], filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"     ✅ Saved ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"     ❌ Failed: {e}")

    print(f"\n📁 Sample images are in: {output_dir}")
    print("   Run: python classifier.py sample_images/<filename>")


if __name__ == "__main__":
    download_samples()
