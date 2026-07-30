"""
📊 Machine Learning — Iris Classification
============================================
Multi-classifier comparison on the classic Iris dataset.
Trains Random Forest, SVM, KNN, and Logistic Regression, compares
accuracy, plots confusion matrices, and saves the best model.

Usage:
    python iris_classification.py
"""

import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from utils import (
    plot_confusion_matrix,
    plot_model_comparison,
    print_classification_results,
)


def load_and_explore_data():
    """Load the Iris dataset and print exploration summary."""
    print("📥 Loading Iris dataset...\n")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    print(f"   Samples:    {X.shape[0]}")
    print(f"   Features:   {X.shape[1]}")
    print(f"   Classes:    {len(target_names)} — {target_names}")
    print(f"   Features:   {feature_names}")

    # Class distribution
    print("\n   Class Distribution:")
    for i, name in enumerate(target_names):
        count = np.sum(y == i)
        print(f"     {name}: {count} samples ({count/len(y)*100:.0f}%)")
    print()

    return X, y, feature_names, target_names


def train_and_compare_models(X_train, X_test, y_train, y_test, target_names):
    """Train multiple classifiers and compare their performance."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"🏋️  Training: {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "y_pred": y_pred,
        }

        print(f"   Test Accuracy:    {accuracy * 100:.2f}%")
        print(f"   Cross-Val (5-fold): {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%\n")

    return results


def main():
    print("=" * 55)
    print("  📊 Machine Learning — Iris Classification")
    print("  🌸 Multi-Classifier Comparison")
    print("=" * 55 + "\n")

    # Load data
    X, y, feature_names, target_names = load_and_explore_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"📂 Train/Test Split: {len(X_train)} train, {len(X_test)} test\n")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and compare
    results = train_and_compare_models(
        X_train_scaled, X_test_scaled, y_train, y_test, target_names,
    )

    # Find best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_result = results[best_name]
    print(f"🏆 Best Model: {best_name} ({best_result['accuracy'] * 100:.2f}%)\n")

    # Detailed report for best model
    print(f"📋 Detailed Report — {best_name}:")
    print_classification_results(y_test, best_result["y_pred"], target_names)

    # Plot confusion matrix for best model
    plot_confusion_matrix(
        y_test, best_result["y_pred"],
        labels=target_names,
        title=f"Confusion Matrix — {best_name}",
        save_path="iris_confusion_matrix.png",
    )

    # Plot model comparison
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]
    plot_model_comparison(
        model_names, accuracies,
        title="Iris Classification — Model Comparison",
        save_path="iris_model_comparison.png",
    )

    # Save best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/iris_best_model.joblib"
    joblib.dump(best_result["model"], model_path)
    print(f"💾 Best model saved to: {model_path}")

    scaler_path = "models/iris_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to: {scaler_path}")

    print("\n✅ Iris classification complete!")


if __name__ == "__main__":
    main()
