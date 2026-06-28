import numpy as np
import pandas as pd
import streamlit as st

from app import (
    MODELS,
    RESULTS,
    detect_ai_combined,
    get_prediction_pipeline,
    load_models,
)


st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading TF-IDF models...")
def initialize_models():
    return load_models()


def predict_article(text, model_name):
    predict_fn = get_prediction_pipeline(model_name)
    probabilities = predict_fn([text])[0]
    prediction = int(np.argmax(probabilities))
    label = "Real" if prediction == 1 else "Fake"
    confidence = round(float(probabilities[prediction]) * 100, 1)

    all_predictions = {}
    for name in MODELS:
        model_predict_fn = get_prediction_pipeline(name)
        model_probs = model_predict_fn([text])[0]
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

    ai_detection = detect_ai_combined(text)

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": round(float(probabilities[0]) * 100, 1),
        "real_probability": round(float(probabilities[1]) * 100, 1),
        "all_predictions": all_predictions,
        "ensemble": {
            "label": ensemble_label,
            "confidence": ensemble_confidence,
        },
        "ai_detection": ai_detection,
    }


def show_model_results():
    st.subheader("Model Results")

    if RESULTS:
        metrics = pd.DataFrame(RESULTS).T
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

    models_loaded = initialize_models()

    if not models_loaded:
        st.error("Trained model files were not found.")
        st.info(
            "Run `python train.py` locally to generate the `.joblib` files in `models/`. "
            "For Streamlit Cloud deployment, those model files must be available to the app."
        )
        show_model_results()
        return

    with st.sidebar:
        st.header("Analysis Settings")
        selected_model = st.selectbox("Primary model", list(MODELS.keys()))
        st.markdown(
            "[Open EDA notebook](https://github.com/AbuBakerAttique/fake-news-project/blob/main/eda.ipynb)"
        )

    article_text = st.text_area(
        "Paste a news article",
        height=240,
        placeholder="Paste the full news article text here...",
    )

    analyze = st.button("Analyze Article", type="primary")

    if analyze:
        if not article_text.strip():
            st.warning("Please paste an article before analyzing.")
        else:
            result = predict_article(article_text.strip(), selected_model)

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
                ai_col.metric("AI-style probability", f"{ai_detection['ai_prob']}%")
                human_col.metric("Human-style probability", f"{ai_detection['human_prob']}%")
                st.dataframe(
                    pd.DataFrame(
                        ai_detection["statistical"]["features"].items(),
                        columns=["Indicator", "Score"],
                    ),
                    use_container_width=True,
                )

    st.divider()
    show_model_results()


if __name__ == "__main__":
    main()
