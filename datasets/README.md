# 📁 Datasets

Datasets used by the AI projects in this repository.

## Available Datasets

| Dataset | File | Used By | Description |
|---------|------|---------|-------------|
| **Titanic** | `titanic.csv` | `machine-learning/titanic_survival.py` | Titanic passenger survival data (191 rows) |
| **Iris** | *(built-in)* | `machine-learning/iris_classification.py` | Loaded automatically via scikit-learn |
| **MNIST** | *(built-in)* | `deep-learning/train.py` | Downloaded automatically via Keras |
| **20 Newsgroups** | *(built-in)* | `natural-language-processing/text_classifier.py` | Loaded via scikit-learn |
| **Movie Reviews** | *(built-in)* | `natural-language-processing/sentiment_analysis.py` | Loaded via NLTK |

## Adding New Datasets

Place new datasets in this directory and update this README. Use relative paths
in your scripts (e.g., `../datasets/my_data.csv`).
