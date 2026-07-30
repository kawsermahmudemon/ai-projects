"""
📊 Machine Learning — Titanic Survival Prediction
====================================================
Full ML pipeline on the Titanic dataset: EDA, feature engineering,
preprocessing, training (Gradient Boosting), and evaluation.

Usage:
    python titanic_survival.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

from utils import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    print_classification_results,
)


def load_data():
    """Load the Titanic dataset."""
    # Look for CSV in datasets directory or current directory
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "titanic.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic.csv"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"📥 Loaded Titanic dataset from: {path}")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
            return df

    print("❌ Titanic dataset not found.")
    print("   Expected at: ../datasets/titanic.csv")
    sys.exit(1)


def exploratory_data_analysis(df: pd.DataFrame):
    """Perform and display EDA on the Titanic dataset."""
    print("🔍 Exploratory Data Analysis")
    print("─" * 50)

    # Basic info
    print(f"\n📋 Dataset Shape: {df.shape}")
    print(f"\n📊 Column Types:")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        null_pct = nulls / len(df) * 100
        unique = df[col].nunique()
        null_info = f" ({nulls} nulls, {null_pct:.0f}%)" if nulls > 0 else ""
        print(f"   {col:<15} {str(dtype):<10} {unique:>5} unique{null_info}")

    # Survival stats
    print(f"\n🎯 Survival Distribution:")
    survived_counts = df["Survived"].value_counts()
    for val, count in survived_counts.items():
        label = "Survived" if val == 1 else "Died"
        pct = count / len(df) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")

    # Numerical summary
    print(f"\n📈 Numerical Summary:")
    print(df.describe().round(2).to_string())
    print()

    # Survival by class
    print("🎫 Survival Rate by Class:")
    for pclass in sorted(df["Pclass"].unique()):
        rate = df[df["Pclass"] == pclass]["Survived"].mean()
        print(f"   Class {pclass}: {rate * 100:.1f}%")

    # Survival by sex
    print("\n👤 Survival Rate by Sex:")
    for sex in df["Sex"].unique():
        rate = df[df["Sex"] == sex]["Survived"].mean()
        print(f"   {sex.title()}: {rate * 100:.1f}%")

    print()


def plot_eda(df: pd.DataFrame):
    """Create EDA visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Survival count
    sns.countplot(data=df, x="Survived", ax=axes[0, 0], hue="Survived",
                  palette=["#ff6b6b", "#51cf66"], legend=False)
    axes[0, 0].set_xticklabels(["Died", "Survived"])
    axes[0, 0].set_title("Survival Count", fontweight="bold")

    # 2. Survival by class
    sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0, 1],
                  palette=["#ff6b6b", "#51cf66"])
    axes[0, 1].set_title("Survival by Class", fontweight="bold")
    axes[0, 1].legend(["Died", "Survived"])

    # 3. Survival by sex
    sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[0, 2],
                  palette=["#ff6b6b", "#51cf66"])
    axes[0, 2].set_title("Survival by Sex", fontweight="bold")
    axes[0, 2].legend(["Died", "Survived"])

    # 4. Age distribution
    axes[1, 0].hist(df[df["Survived"] == 0]["Age"].dropna(), bins=30,
                    alpha=0.6, color="#ff6b6b", label="Died")
    axes[1, 0].hist(df[df["Survived"] == 1]["Age"].dropna(), bins=30,
                    alpha=0.6, color="#51cf66", label="Survived")
    axes[1, 0].set_title("Age Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Age")
    axes[1, 0].legend()

    # 5. Fare distribution
    axes[1, 1].hist(df[df["Survived"] == 0]["Fare"].dropna(), bins=30,
                    alpha=0.6, color="#ff6b6b", label="Died")
    axes[1, 1].hist(df[df["Survived"] == 1]["Fare"].dropna(), bins=30,
                    alpha=0.6, color="#51cf66", label="Survived")
    axes[1, 1].set_title("Fare Distribution", fontweight="bold")
    axes[1, 1].set_xlabel("Fare")
    axes[1, 1].legend()

    # 6. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                fmt=".2f", ax=axes[1, 2], square=True)
    axes[1, 2].set_title("Correlation Matrix", fontweight="bold")

    plt.suptitle("Titanic — Exploratory Data Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("titanic_eda.png", dpi=150, bbox_inches="tight")
    print("📊 EDA plots saved to: titanic_eda.png")
    plt.show()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and preprocess the dataset."""
    print("🔧 Feature Engineering...")

    df = df.copy()

    # Extract title from name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    # Simplify rare titles
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    }
    df["Title"] = df["Title"].map(lambda x: title_mapping.get(x, "Rare"))

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Is alone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age bins
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                          labels=["Child", "Teen", "Adult", "Middle", "Senior"])

    # Fare bins
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Cabin: has cabin or not
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Embarked: fill missing with mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    print(f"   Created features: Title, FamilySize, IsAlone, AgeBin, HasCabin")
    print(f"   Final shape: {df.shape}\n")

    return df


def prepare_features(df: pd.DataFrame):
    """Select and encode features for modeling."""
    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "Title", "FamilySize", "IsAlone", "HasCabin",
    ]

    X = df[feature_cols].copy()
    y = df["Survived"].values

    # Encode categorical features
    le = LabelEncoder()
    for col in ["Sex", "Embarked", "Title"]:
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = list(X.columns)

    return X.values, y, feature_names


def main():
    print("=" * 55)
    print("  📊 Machine Learning — Titanic Survival Prediction")
    print("  🚢 Full ML Pipeline: EDA → Features → Train → Evaluate")
    print("=" * 55 + "\n")

    # Load data
    df = load_data()

    # EDA
    exploratory_data_analysis(df)
    plot_eda(df)

    # Feature engineering
    df = feature_engineering(df)

    # Prepare features
    X, y, feature_names = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"📂 Train/Test Split: {len(X_train)} train, {len(X_test)} test\n")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"🏋️  Training: {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "y_pred": y_pred,
        }
        print(f"   Test Accuracy: {accuracy * 100:.2f}% | "
              f"CV: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%\n")

    # Best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_result = results[best_name]
    print(f"🏆 Best Model: {best_name} ({best_result['accuracy'] * 100:.2f}%)\n")

    # Detailed report
    print_classification_results(
        y_test, best_result["y_pred"],
        target_names=["Died", "Survived"],
    )

    # Plots
    plot_confusion_matrix(
        y_test, best_result["y_pred"],
        labels=["Died", "Survived"],
        title=f"Titanic — {best_name}",
        save_path="titanic_confusion_matrix.png",
    )

    plot_model_comparison(
        list(results.keys()),
        [results[n]["accuracy"] for n in results],
        title="Titanic — Model Comparison",
        save_path="titanic_model_comparison.png",
    )

    # Feature importance (Gradient Boosting)
    gb_model = results["Gradient Boosting"]["model"]
    plot_feature_importance(
        feature_names, gb_model.feature_importances_,
        title="Titanic — Feature Importance (Gradient Boosting)",
        save_path="titanic_feature_importance.png",
    )

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_result["model"], "models/titanic_best_model.joblib")
    joblib.dump(scaler, "models/titanic_scaler.joblib")
    print(f"\n💾 Models saved to: models/")

    print("\n✅ Titanic survival prediction complete!")


if __name__ == "__main__":
    main()
