# Fake News Detector

An explainable web application that classifies news text as **Fake** or **Real** using TF-IDF machine learning models, LIME explanations, and statistical writing-style analysis.

## Features

- Flask web interface for submitting and analyzing news articles.
- Streamlit app entrypoint for simple cloud deployment.
- Three classical ML classifiers: Logistic Regression, Multinomial Naive Bayes, and Passive Aggressive Classifier.
- TF-IDF preprocessing pipeline with stopword removal and lemmatization.
- Combined verdict with individual model comparison.
- LIME feature explanations showing which words influenced the prediction.
- Statistical writing-style indicators for sentence uniformity, vocabulary pattern, formality, burstiness, and transition density.
- Word cloud visualizations for Fake vs Real influence terms.

## Tech Stack

- Python, Flask
- Streamlit
- scikit-learn, NLTK, LIME
- HTML, CSS, JavaScript
- wordcloud2.js
- Kaggle Fake and Real News dataset

## Project Structure

```text
.
├── app.py
├── streamlit_app.py
├── train.py
├── requirements.txt
├── run.bat
├── run.command
├── models/
│   ├── results.json
│   └── cm_*.png
├── static/
│   └── style.css
└── templates/
    └── index.html
```

Generated model binaries are intentionally not committed. Run the training script to create them locally.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bat
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the Flask app or local model training, also install:

```bash
pip install flask lime matplotlib seaborn kagglehub
```

## Train Models

```bash
python train.py
```

The training script downloads the public Kaggle dataset with `kagglehub` when available. You can also place `True.csv` and `Fake.csv` inside a local `data/` directory.

Training creates:

- `models/tfidf_vectorizer.joblib`
- `models/logistic_regression.joblib`
- `models/multinomial_naive_bayes.joblib`
- `models/passive_aggressive_classifier.joblib`
- `models/results.json`
- confusion matrix images

## EDA and Model Results

The repository includes the exploratory analysis notebook and saved model result images:

- [Exploratory Data Analysis notebook](eda.ipynb)
- [Model metrics JSON](models/results.json)

### Logistic Regression

![Logistic Regression confusion matrix](models/cm_logistic_regression.png)

### Multinomial Naive Bayes

![Multinomial Naive Bayes confusion matrix](models/cm_multinomial_naive_bayes.png)

### Passive Aggressive Classifier

![Passive Aggressive Classifier confusion matrix](models/cm_passive_aggressive_classifier.png)

## Run

Run the Flask app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5002
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud

Use these settings when creating the app on Streamlit Community Cloud:

- Repository: `AbuBakerAttique/fake-news-project`
- Branch: `main`
- Main file path: `streamlit_app.py`

The app needs the trained `.joblib` files in `models/` to make live predictions. If they are not present, the Streamlit page still opens and shows the EDA/model result images, but prediction is disabled until the model artifacts are available.

## API

`POST /predict`

```json
{
  "text": "Full article text to classify.",
  "model": "Logistic Regression"
}
```

Supported model names:

- `Logistic Regression`
- `Multinomial Naive Bayes`
- `Passive Aggressive Classifier`

The response includes the primary prediction, combined verdict, per-model predictions, statistical writing-style results, LIME explanation data, and word cloud data.

## Notes

This project is a portfolio-ready implementation of an explainable text classification workflow. The model is trained on a public English news dataset and should be treated as a decision-support demo, not a definitive fact-checking system.
