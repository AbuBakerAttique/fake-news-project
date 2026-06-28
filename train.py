"""
Fake News Detection - Model Training Pipeline
Trains Logistic Regression, Multinomial Naive Bayes, and Passive Aggressive Classifier
using TF-IDF features on the Kaggle Fake News dataset.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def load_dataset():
    """Load dataset using kagglehub or fall back to local CSV files."""
    try:
        import kagglehub
        print("Downloading dataset via kagglehub...")
        path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        print(f"Dataset downloaded to: {path}")

        true_path = os.path.join(path, "True.csv")
        fake_path = os.path.join(path, "Fake.csv")

        if os.path.exists(true_path) and os.path.exists(fake_path):
            df_true = pd.read_csv(true_path)
            df_fake = pd.read_csv(fake_path)
            df_true["label"] = 1  # Real
            df_fake["label"] = 0  # Fake
            df = pd.concat([df_true, df_fake], ignore_index=True)
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            return df[["text", "label"]]
    except Exception as e:
        print(f"kagglehub download failed: {e}")

    # Fallback: check local data directory
    for pattern in [
        ("True.csv", "Fake.csv"),
        ("true.csv", "fake.csv"),
    ]:
        true_p = os.path.join(DATA_DIR, pattern[0])
        fake_p = os.path.join(DATA_DIR, pattern[1])
        if os.path.exists(true_p) and os.path.exists(fake_p):
            print(f"Loading local dataset from {DATA_DIR}")
            df_true = pd.read_csv(true_p)
            df_fake = pd.read_csv(fake_p)
            df_true["label"] = 1
            df_fake["label"] = 0
            df = pd.concat([df_true, df_fake], ignore_index=True)
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            return df[["text", "label"]]

    # Fallback: single-file dataset with 'text' and 'label' columns
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".csv"):
            fpath = os.path.join(DATA_DIR, fname)
            df = pd.read_csv(fpath)
            if "text" in df.columns and "label" in df.columns:
                print(f"Loading local dataset: {fpath}")
                return df[["text", "label"]]

    raise FileNotFoundError(
        "No dataset found. Place True.csv and Fake.csv in the data/ directory, "
        "or ensure kagglehub can download 'clmentbisaillon/fake-and-real-news-dataset'."
    )


def plot_confusion_matrix(cm, name, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name} - Confusion Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)
    classes = ["Fake", "Real"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("  FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading dataset...")
    df = load_dataset()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Preprocess
    print("\n[2/6] Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    print(f"  After cleaning: {len(df)} samples")

    # 3. Split
    print("\n[3/6] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. TF-IDF
    print("\n[4/6] Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"  Feature matrix: {X_train_tfidf.shape}")

    # 5. Train models
    print("\n[5/6] Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(
            max_iter=1000, C=1.0, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_tfidf)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test_tfidf)
            auc = roc_auc_score(y_test, y_scores)
        else:
            auc = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4) if auc else "N/A",
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}" if auc else "    ROC-AUC:   N/A")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Fake', 'Real'])}")

        # Save confusion matrix plot
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(cm, name, os.path.join(MODELS_DIR, f"cm_{safe_name}.png"))

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(model, model_path)
        print(f"    Model saved: {model_path}")

    # Save TF-IDF vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    # Save results summary
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 6. Summary
    print("\n[6/6] Results Summary")
    print("=" * 60)
    print(f"{'Model':<35} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 60)
    for name, metrics in results.items():
        auc_str = f"{metrics['roc_auc']}" if isinstance(metrics["roc_auc"], float) else metrics["roc_auc"]
        print(f"{name:<35} {metrics['accuracy']:>10} {metrics['f1_score']:>10} {auc_str:>10}")
    print("=" * 60)

    best = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest model: {best[0]} (Accuracy: {best[1]['accuracy']})")
    print(f"\nAll artifacts saved to: {MODELS_DIR}")
    print("Run 'python app.py' to start the web application.\n")


if __name__ == "__main__":
    main()
