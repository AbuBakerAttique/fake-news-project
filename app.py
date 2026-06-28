"""
Fake News Detection - Flask Web Application
TF-IDF classifiers (Logistic Regression, Naive Bayes, Passive Aggressive),
statistical text analysis for writing-style signals, and LIME explanations.
"""

import os
import re
import json
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from lime.lime_text import LimeTextExplainer
import joblib

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

MODELS = {}
TFIDF = None
RESULTS = {}
EXPLAINER = LimeTextExplainer(class_names=["Fake", "Real"], random_state=42)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
    "Passive Aggressive Classifier": "passive_aggressive_classifier.joblib",
}


# ──────────────────────────────────────────────────
#  Statistical AI Text Analysis
# ──────────────────────────────────────────────────

def analyze_text_statistics(text):
    """
    Compute statistical features that distinguish AI-generated text from
    human-written text. AI text tends to be more uniform, uses fewer
    contractions, has less sentence-length variation, and shows specific
    vocabulary patterns.

    Returns a dict with individual scores and a combined AI probability.
    """
    sentences = nltk.sent_tokenize(text)
    words = text.split()

    if len(words) < 20 or len(sentences) < 2:
        return None

    features = {}

    # 1. Sentence length uniformity
    # AI-generated text has very consistent sentence lengths
    sent_lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(sent_lengths)
    std_len = np.std(sent_lengths)
    cv = std_len / mean_len if mean_len > 0 else 0
    # Human text: CV typically 0.4-0.8+, AI text: 0.15-0.4
    uniformity_score = max(0, min(1, 1 - (cv - 0.15) / 0.55))
    features["sentence_uniformity"] = round(uniformity_score, 3)

    # 2. Vocabulary richness (Type-Token Ratio)
    # AI text has a specific moderate TTR, neither too high nor too low
    unique_words = set(w.lower() for w in words if w.isalpha())
    ttr = len(unique_words) / len(words) if words else 0
    # AI text typically has TTR 0.45-0.65 for longer texts
    # Very high (>0.75) or very low (<0.35) is more human-like
    ttr_ai_score = max(0, min(1, 1 - abs(ttr - 0.55) / 0.25))
    features["vocabulary_pattern"] = round(ttr_ai_score, 3)

    # 3. Contraction usage
    # AI text uses significantly fewer contractions
    contractions = [
        "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
        "weren't", "won't", "wouldn't", "couldn't", "shouldn't",
        "can't", "it's", "that's", "there's", "here's", "what's",
        "who's", "he's", "she's", "we're", "they're", "you're",
        "I'm", "I've", "I'll", "I'd", "we've", "they've", "you've",
        "let's", "hasn't", "haven't", "hadn't",
    ]
    text_lower = text.lower()
    contraction_count = sum(1 for c in contractions if c in text_lower)
    contraction_rate = contraction_count / len(sentences)
    # No contractions in multi-sentence text is suspicious
    no_contractions_score = max(0, min(1, 1 - contraction_rate / 0.5))
    features["no_contractions"] = round(no_contractions_score, 3)

    # 4. Average word length
    # AI text tends toward slightly longer words (more formal)
    alpha_words = [w for w in words if w.isalpha()]
    avg_word_len = np.mean([len(w) for w in alpha_words]) if alpha_words else 0
    # AI text: avg 5.0-6.5, Human casual: 4.0-5.0
    formality_score = max(0, min(1, (avg_word_len - 4.5) / 2.0))
    features["formality"] = round(formality_score, 3)

    # 5. Burstiness (sentence length pattern variation)
    # Human writers are "bursty" - short sentences then long, AI is smooth
    if len(sent_lengths) >= 3:
        diffs = [abs(sent_lengths[i] - sent_lengths[i-1]) for i in range(1, len(sent_lengths))]
        avg_diff = np.mean(diffs)
        burstiness = avg_diff / mean_len if mean_len > 0 else 0
        # Low burstiness = likely AI
        smoothness_score = max(0, min(1, 1 - (burstiness - 0.1) / 0.5))
    else:
        smoothness_score = 0.5
    features["low_burstiness"] = round(smoothness_score, 3)

    # 6. Transition word density
    # AI text uses more transition/connector words
    transitions = [
        "however", "furthermore", "moreover", "additionally", "consequently",
        "nevertheless", "therefore", "meanwhile", "subsequently", "specifically",
        "accordingly", "notably", "significantly", "essentially", "fundamentally",
        "particularly", "importantly", "interestingly", "remarkably", "ultimately",
        "effectively", "compelling", "comprehensive", "demonstrate", "indicating",
        "suggesting", "highlighting", "underscoring", "emphasizing", "revealing",
    ]
    transition_count = sum(1 for w in words if w.lower() in transitions)
    transition_rate = transition_count / len(sentences) if sentences else 0
    transition_score = max(0, min(1, transition_rate / 1.0))
    features["transition_density"] = round(transition_score, 3)

    # 7. Paragraph/Sentence structure regularity
    # AI text tends to have very similar sentence structures
    sent_word_counts = sorted(sent_lengths)
    if len(sent_word_counts) >= 4:
        q1 = np.percentile(sent_word_counts, 25)
        q3 = np.percentile(sent_word_counts, 75)
        iqr = q3 - q1
        iqr_ratio = iqr / mean_len if mean_len > 0 else 0
        structure_score = max(0, min(1, 1 - iqr_ratio / 0.6))
    else:
        structure_score = 0.5
    features["structural_regularity"] = round(structure_score, 3)

    # Combined AI probability (weighted average)
    weights = {
        "sentence_uniformity": 0.20,
        "vocabulary_pattern": 0.10,
        "no_contractions": 0.15,
        "formality": 0.15,
        "low_burstiness": 0.15,
        "transition_density": 0.15,
        "structural_regularity": 0.10,
    }

    ai_score = sum(features[k] * weights[k] for k in weights)
    ai_score = max(0, min(1, ai_score))

    return {
        "features": {k: float(v) for k, v in features.items()},
        "ai_probability": round(float(ai_score * 100), 1),
        "human_probability": round(float((1 - ai_score) * 100), 1),
        "is_ai_generated": bool(ai_score > 0.45),
    }


# ──────────────────────────────────────────────────
#  AI Detection (Statistical Text Analysis Only)
# ──────────────────────────────────────────────────

def detect_ai_combined(text):
    """Use statistical text analysis only (no transformer)."""
    stats_result = analyze_text_statistics(text)
    if not stats_result:
        return None
    return {
        "human_prob": float(stats_result["human_probability"]),
        "ai_prob": float(stats_result["ai_probability"]),
        "is_ai_generated": bool(stats_result["is_ai_generated"]),
        "transformer": None,
        "statistical": stats_result,
    }


# ──────────────────────────────────────────────────
#  TF-IDF Models
# ──────────────────────────────────────────────────

def load_models():
    global TFIDF, RESULTS
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if not os.path.exists(tfidf_path):
        print("WARNING: Models not trained yet. Run 'python train.py' first.")
        return False

    TFIDF = joblib.load(tfidf_path)

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
            print(f"  Loaded: {name}")

    results_path = os.path.join(MODELS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            RESULTS.update(json.load(f))

    return len(MODELS) > 0


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


def get_prediction_pipeline(model_name):
    model = MODELS[model_name]

    def predict_proba(texts):
        cleaned = [preprocess_text(t) for t in texts]
        features = TFIDF.transform(cleaned)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)
        else:
            decisions = model.decision_function(features)
            if decisions.ndim == 1:
                proba_pos = 1 / (1 + np.exp(-decisions))
                return np.column_stack([1 - proba_pos, proba_pos])
            return decisions

    return predict_proba


# ──────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=list(MODELS.keys()),
        results=RESULTS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    model_name = data.get("model", "Logistic Regression")

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if model_name not in MODELS:
        return jsonify({"error": f"Model '{model_name}' not found."}), 400

    predict_fn = get_prediction_pipeline(model_name)

    probabilities = predict_fn([text])[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    label = "Real" if prediction == 1 else "Fake"

    # LIME explanation
    try:
        explanation = EXPLAINER.explain_instance(
            text,
            predict_fn,
            num_features=15,
            num_samples=500,
        )
        lime_weights = explanation.as_list()
        lime_data = [
            {"word": w, "weight": round(float(score), 4)}
            for w, score in lime_weights
        ]
    except Exception as e:
        lime_data = []
        print(f"LIME error: {e}")

    # All TF-IDF models
    all_predictions = {}
    for mname in MODELS:
        pfn = get_prediction_pipeline(mname)
        probs = pfn([text])[0]
        pred = int(np.argmax(probs))
        all_predictions[mname] = {
            "label": "Real" if pred == 1 else "Fake",
            "confidence": round(float(probs[pred]) * 100, 1),
            "fake_prob": round(float(probs[0]) * 100, 1),
            "real_prob": round(float(probs[1]) * 100, 1),
        }

    # AI-style detection (statistical text analysis only)
    ai_detection = detect_ai_combined(text)

    # Combined verdict
    tfidf_votes = [v["label"] for v in all_predictions.values()]
    tfidf_label = max(set(tfidf_votes), key=tfidf_votes.count)
    tfidf_avg_conf = np.mean([
        v["confidence"] for v in all_predictions.values()
        if v["label"] == tfidf_label
    ])

    if ai_detection:
        ai_says_fake = ai_detection["is_ai_generated"]
        ai_conf = ai_detection["ai_prob"]

        if ai_says_fake and ai_conf >= 45:
            final_label = "Fake"
            if tfidf_label == "Fake":
                final_confidence = round(max(float(tfidf_avg_conf), float(ai_conf)), 1)
            else:
                final_confidence = round(0.35 * float(tfidf_avg_conf) + 0.65 * float(ai_conf), 1)
        elif not ai_says_fake and tfidf_label == "Real":
            final_label = "Real"
            human_conf = ai_detection["human_prob"]
            final_confidence = round(0.5 * float(tfidf_avg_conf) + 0.5 * float(human_conf), 1)
        elif tfidf_label == "Fake" and not ai_says_fake:
            final_label = "Fake"
            final_confidence = round(float(tfidf_avg_conf) * 0.8, 1)
        else:
            final_label = tfidf_label
            final_confidence = round(float(tfidf_avg_conf), 1)
    else:
        final_label = tfidf_label
        final_confidence = round(float(tfidf_avg_conf), 1)

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "fake_probability": round(float(probabilities[0]) * 100, 1),
        "real_probability": round(float(probabilities[1]) * 100, 1),
        "model": model_name,
        "lime_explanation": lime_data,
        "all_predictions": all_predictions,
        "ai_detection": ai_detection,
        "combined_verdict": {
            "label": final_label,
            "confidence": final_confidence,
        },
        "ensemble": {
            "label": tfidf_label,
            "confidence": round(float(tfidf_avg_conf), 1),
        },
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  FAKE NEWS DETECTION - WEB APPLICATION")
    print("=" * 50)
    print("\nLoading models...")

    if load_models():
        print(f"\n{len(MODELS)} TF-IDF model(s) loaded.")
        print("  AI detection: statistical text analysis only (no transformer).")
        print(f"\nStarting server at http://127.0.0.1:5002\n")
        app.run(debug=False, host="0.0.0.0", port=5002)
    else:
        print("\nERROR: No trained models found.")
        print("Please run 'python train.py' first to train the models.\n")
