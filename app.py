"""
Fake News Detector - small Streamlit demo for Project 15.

Run:
    pip install streamlit scikit-learn pandas joblib
    streamlit run app.py

Put Fake.csv and True.csv (ISOT dataset) in the same folder as this file.
The first run trains a Logistic Regression model (about a minute) and saves it
to model.joblib; later runs load the saved model instantly.
"""
import os
import re

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = "model.joblib"
REUTERS_PREFIX = re.compile(r"^.{0,150}?\(reuters\)\s*[-–—]*\s*", re.IGNORECASE | re.DOTALL)


def clean_text(t):
    """Same cleaning as the notebook (removes the Reuters dateline shortcut)."""
    t = "" if pd.isna(t) else str(t)
    t = REUTERS_PREFIX.sub("", t)
    t = re.sub(r"http\S+|www\.\S+|pic\.twitter\.com\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.lower()
    t = re.sub(r"\breuters\b", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def train_model(data_dir="."):
    fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    true = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake["label"], true["label"] = 1, 0  # Fake = 1, Real = 0
    df = pd.concat([fake, true], ignore_index=True)
    df["clean"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).apply(clean_text)
    df = df[df["clean"].str.len() > 0].drop_duplicates(subset="clean")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000,
                                  min_df=5, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear", C=10, random_state=42)),
    ])
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    joblib.dump({"model": model, "accuracy": accuracy}, MODEL_PATH)
    return model, accuracy


@st.cache_resource(show_spinner="Loading model (first run trains it, ~1 minute)...")
def load_model():
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved["accuracy"]
    return train_model()


def top_words(model, cleaned, n=8):
    """Words in the input that pushed the prediction most towards Fake / Real."""
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vec = tfidf.transform([cleaned])
    vocab = tfidf.get_feature_names_out()
    contrib = vec.toarray()[0] * clf.coef_[0]
    idx = contrib.argsort()
    fake_words = [vocab[i] for i in idx[::-1][:n] if contrib[i] > 0]
    real_words = [vocab[i] for i in idx[:n] if contrib[i] < 0]
    return fake_words, real_words


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.title("📰 Fake News Detector")
    st.caption("Logistic Regression + TF-IDF trained on the ISOT dataset")

    try:
        model, accuracy = load_model()
    except FileNotFoundError:
        st.error("Fake.csv and True.csv were not found. Put them in the same folder as app.py and rerun.")
        st.stop()

    st.write(f"Model accuracy on held-out test data: **{accuracy:.1%}**")

    title = st.text_input("Headline (optional)")
    body = st.text_area("Paste the article text", height=220)

    if st.button("Check article", type="primary"):
        cleaned = clean_text(f"{title}. {body}")
        if len(cleaned.split()) < 20:
            st.warning("Please paste at least a few sentences (20+ words) for a meaningful result.")
        else:
            p_fake = float(model.predict_proba([cleaned])[0][1])
            if p_fake >= 0.5:
                st.error(f"Likely FAKE  ({p_fake:.1%} confidence)")
            else:
                st.success(f"Likely REAL  ({1 - p_fake:.1%} confidence)")
            st.progress(p_fake, text=f"Fake probability: {p_fake:.1%}")

            fake_w, real_w = top_words(model, cleaned)
            col1, col2 = st.columns(2)
            col1.markdown("**Words pushing towards Fake**")
            col1.write(", ".join(fake_w) or "none")
            col2.markdown("**Words pushing towards Real**")
            col2.write(", ".join(real_w) or "none")

    st.info(
        "Demo only. The model learned patterns from 2015-2018 US political news, "
        "so it can be wrong on other topics, newer events or other writing styles. "
        "Don't use it to judge real-world claims."
    )


if __name__ == "__main__":
    main()
