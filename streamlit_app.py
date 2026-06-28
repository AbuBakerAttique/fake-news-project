import json
import os
import re

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
    "Passive Aggressive Classifier": "passive_aggressive_classifier.joblib",
}


st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_nltk_tools():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    return WordNetLemmatizer(), set(stopwords.words("english"))


@st.cache_resource(show_spinner="Loading TF-IDF models...")
def load_model_artifacts():
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if not os.path.exists(tfidf_path):
        return None, {}, load_results()

    tfidf = joblib.load(tfidf_path)
    models = {}
    for name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)

    return tfidf, models, load_results()


def load_results():
    results_path = os.path.join(MODELS_DIR, "results.json")
    if not os.path.exists(results_path):
        return {}

    with open(results_path) as file:
        return json.load(file)


def preprocess_text(text):
    lemmatizer, stop_words = get_nltk_tools()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def analyze_text_statistics(text):
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    sentences = nltk.sent_tokenize(text)
    words = text.split()

    if len(words) < 20 or len(sentences) < 2:
        return None

    sent_lengths = [len(sentence.split()) for sentence in sentences]
    mean_len = np.mean(sent_lengths)
    std_len = np.std(sent_lengths)
    cv = std_len / mean_len if mean_len > 0 else 0

    unique_words = set(word.lower() for word in words if word.isalpha())
    ttr = len(unique_words) / len(words) if words else 0

    contractions = [
        "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
        "weren't", "won't", "wouldn't", "couldn't", "shouldn't",
        "can't", "it's", "that's", "there's", "here's", "what's",
        "who's", "he's", "she's", "we're", "they're", "you're",
        "i'm", "i've", "i'll", "i'd", "we've", "they've", "you've",
        "let's", "hasn't", "haven't", "hadn't",
    ]
    text_lower = text.lower()
    contraction_count = sum(1 for contraction in contractions if contraction in text_lower)
    contraction_rate = contraction_count / len(sentences)

    alpha_words = [word for word in words if word.isalpha()]
    avg_word_len = np.mean([len(word) for word in alpha_words]) if alpha_words else 0

    if len(sent_lengths) >= 3:
        diffs = [
            abs(sent_lengths[index] - sent_lengths[index - 1])
            for index in range(1, len(sent_lengths))
        ]
        avg_diff = np.mean(diffs)
        burstiness = avg_diff / mean_len if mean_len > 0 else 0
        smoothness_score = max(0, min(1, 1 - (burstiness - 0.1) / 0.5))
    else:
        smoothness_score = 0.5

    transitions = [
        "however", "furthermore", "moreover", "additionally", "consequently",
        "nevertheless", "therefore", "meanwhile", "subsequently", "specifically",
        "accordingly", "notably", "significantly", "essentially", "fundamentally",
        "particularly", "importantly", "interestingly", "remarkably", "ultimately",
        "effectively", "compelling", "comprehensive", "demonstrate", "indicating",
        "suggesting", "highlighting", "underscoring", "emphasizing", "revealing",
    ]
    transition_count = sum(1 for word in words if word.lower() in transitions)
    transition_rate = transition_count / len(sentences) if sentences else 0

    sent_word_counts = sorted(sent_lengths)
    if len(sent_word_counts) >= 4:
        q1 = np.percentile(sent_word_counts, 25)
        q3 = np.percentile(sent_word_counts, 75)
        iqr = q3 - q1
        iqr_ratio = iqr / mean_len if mean_len > 0 else 0
        structure_score = max(0, min(1, 1 - iqr_ratio / 0.6))
    else:
        structure_score = 0.5

    features = {
        "sentence_uniformity": round(max(0, min(1, 1 - (cv - 0.15) / 0.55)), 3),
        "vocabulary_pattern": round(max(0, min(1, 1 - abs(ttr - 0.55) / 0.25)), 3),
        "no_contractions": round(max(0, min(1, 1 - contraction_rate / 0.5)), 3),
        "formality": round(max(0, min(1, (avg_word_len - 4.5) / 2.0)), 3),
        "low_burstiness": round(smoothness_score, 3),
        "transition_density": round(max(0, min(1, transition_rate / 1.0)), 3),
        "structural_regularity": round(structure_score, 3),
    }

    weights = {
        "sentence_uniformity": 0.20,
        "vocabulary_pattern": 0.10,
        "no_contractions": 0.15,
        "formality": 0.15,
        "low_burstiness": 0.15,
        "transition_density": 0.15,
        "structural_regularity": 0.10,
    }

    ai_score = sum(features[key] * weights[key] for key in weights)
    ai_score = max(0, min(1, ai_score))

    return {
        "features": features,
        "ai_probability": round(float(ai_score * 100), 1),
        "human_probability": round(float((1 - ai_score) * 100), 1),
    }


def predict_article(text, model_name, tfidf, models):
    cleaned = preprocess_text(text)
    features = tfidf.transform([cleaned])
    model = models[model_name]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
    else:
        decision = model.decision_function(features)
        proba_pos = 1 / (1 + np.exp(-decision))
        probabilities = np.array([1 - proba_pos[0], proba_pos[0]])

    prediction = int(np.argmax(probabilities))
    label = "Real" if prediction == 1 else "Fake"

    all_predictions = {}
    for name, current_model in models.items():
        if hasattr(current_model, "predict_proba"):
            model_probs = current_model.predict_proba(features)[0]
        else:
            decision = current_model.decision_function(features)
            proba_pos = 1 / (1 + np.exp(-decision))
            model_probs = np.array([1 - proba_pos[0], proba_pos[0]])

        model_prediction = int(np.argmax(model_probs))
        all_predictions[name] = {
            "label": "Real" if model_prediction == 1 else "Fake",
            "confidence": round(float(model_probs[model_prediction]) * 100, 1),
            "fake_probability": round(float(model_probs[0]) * 100, 1),
            "real_probability": round(float(model_probs[1]) * 100, 1),
        }

    votes = [result["label"] for result in all_predictions.values()]
    ensemble_label = max(set(votes), key=votes.count)
    ensemble_confidence = round(
        float(
            np.mean(
                [
                    result["confidence"]
                    for result in all_predictions.values()
                    if result["label"] == ensemble_label
                ]
            )
        ),
        1,
    )

    return {
        "label": label,
        "confidence": round(float(probabilities[prediction]) * 100, 1),
        "fake_probability": round(float(probabilities[0]) * 100, 1),
        "real_probability": round(float(probabilities[1]) * 100, 1),
        "all_predictions": all_predictions,
        "ensemble": {
            "label": ensemble_label,
            "confidence": ensemble_confidence,
        },
        "ai_detection": analyze_text_statistics(text),
    }


def show_model_results(results):
    st.subheader("Model Results")

    if results:
        metrics = pd.DataFrame(results).T
        st.dataframe(metrics, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            "models/cm_logistic_regression.png",
            caption="Logistic Regression",
            use_container_width=True,
        )
    with col2:
        st.image(
            "models/cm_multinomial_naive_bayes.png",
            caption="Multinomial Naive Bayes",
            use_container_width=True,
        )
    with col3:
        st.image(
            "models/cm_passive_aggressive_classifier.png",
            caption="Passive Aggressive Classifier",
            use_container_width=True,
        )


def main():
    st.title("Fake News Detector")
    st.caption("Explainable TF-IDF classification with statistical writing-style analysis")

    tfidf, models, results = load_model_artifacts()
    models_loaded = tfidf is not None and bool(models)

    if not models_loaded:
        st.warning("Live prediction is waiting for trained model files.")
        st.info(
            "The app is deployed correctly. Add the `.joblib` files to `models/` "
            "when you want prediction enabled."
        )
        show_model_results(results)
        return

    with st.sidebar:
        st.header("Analysis Settings")
        selected_model = st.selectbox("Primary model", list(models.keys()))
        st.markdown(
            "[Open EDA notebook](https://github.com/AbuBakerAttique/fake-news-project/blob/main/eda.ipynb)"
        )

    article_text = st.text_area(
        "Paste a news article",
        height=240,
        placeholder="Paste the full news article text here...",
    )

    if st.button("Analyze Article", type="primary"):
        if not article_text.strip():
            st.warning("Please paste an article before analyzing.")
        else:
            result = predict_article(article_text.strip(), selected_model, tfidf, models)

            verdict_col, fake_col, real_col = st.columns(3)
            verdict_col.metric(
                "Primary verdict",
                result["label"],
                f"{result['confidence']}% confidence",
            )
            fake_col.metric("Fake probability", f"{result['fake_probability']}%")
            real_col.metric("Real probability", f"{result['real_probability']}%")

            st.subheader("Ensemble Verdict")
            st.metric(
                "Majority vote",
                result["ensemble"]["label"],
                f"{result['ensemble']['confidence']}% average confidence",
            )

            comparison = pd.DataFrame(result["all_predictions"]).T
            st.subheader("Model Comparison")
            st.dataframe(comparison, use_container_width=True)

            ai_detection = result["ai_detection"]
            if ai_detection:
                st.subheader("Statistical Writing-Style Analysis")
                ai_col, human_col = st.columns(2)
                ai_col.metric("AI-style probability", f"{ai_detection['ai_probability']}%")
                human_col.metric(
                    "Human-style probability",
                    f"{ai_detection['human_probability']}%",
                )
                st.dataframe(
                    pd.DataFrame(
                        ai_detection["features"].items(),
                        columns=["Indicator", "Score"],
                    ),
                    use_container_width=True,
                )

    st.divider()
    show_model_results(results)


if __name__ == "__main__":
    main()
