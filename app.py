"""
Fake News Detector - Streamlit demo for Project 15 (restyled UI).

Run:
    pip install streamlit scikit-learn pandas joblib
    python -m streamlit run app.py

Put Fake.csv and True.csv (ISOT dataset) in the same folder the first time.
The first run trains a Logistic Regression model and saves it to model.joblib;
later runs (and online deployments) load the saved model instantly.
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

SAMPLES = {
    "Sample: news-wire style": (
        "Senate committee approves revised budget bill after two days of talks",
        "A Senate committee voted on Tuesday to approve a revised budget bill after two days of "
        "negotiations between members of both parties, officials said. The bill, which now moves "
        "to the full chamber, would increase funding for infrastructure projects and delay several "
        "planned spending cuts, according to a statement from the committee chairman. A spokesman "
        "for the minority party told reporters that his members would continue to push for changes "
        "before the final vote, which is expected next week.",
    ),
    "Sample: sensational style": (
        "SHOCKING: Media EXPOSED for hiding the truth about secret deal!!!",
        "You will not believe what the mainstream media does not want you to see! Insiders are "
        "claiming that a secret deal was made behind closed doors and nobody is talking about it. "
        "Outraged citizens are sharing the story everywhere on Twitter because the establishment "
        "is terrified of the truth getting out. Wake up America, they are lying to you again, and "
        "this scandal will blow the whole thing wide open!",
    ),
}


# --------------------------------------------------------------------------- backend
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


# --------------------------------------------------------------------------- frontend
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 880px; padding-top: 1.5rem;}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    border-radius: 20px; padding: 2rem 2rem 1.7rem 2rem; color: #fff;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35); margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff; padding: 0;}
.hero p {margin: .5rem 0 0 0; font-size: 1.02rem; opacity: .92; color: #fff;}

.stat-row {display: flex; gap: .8rem; margin-bottom: 1.2rem; flex-wrap: wrap;}
.stat {
    flex: 1; min-width: 150px; border-radius: 14px; padding: .85rem 1rem;
    background: rgba(127, 127, 127, 0.10); border: 1px solid rgba(127, 127, 127, 0.22);
}
.stat .v {font-size: 1.35rem; font-weight: 700;}
.stat .k {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em;}

.result {
    border-radius: 18px; padding: 1.4rem 1.5rem; margin-top: 1rem; border: 2px solid;
}
.result.fake {background: rgba(239, 68, 68, 0.12); border-color: #ef4444;}
.result.real {background: rgba(34, 197, 94, 0.12); border-color: #22c55e;}
.result .badge {font-size: 1.7rem; font-weight: 800; margin: 0;}
.result.fake .badge {color: #ef4444;}
.result.real .badge {color: #16a34a;}
.result .sub {opacity: .8; margin: .15rem 0 1rem 0;}

.bar {height: 14px; border-radius: 999px; background: rgba(127,127,127,.25); overflow: hidden;}
.bar > div {height: 100%; border-radius: 999px;}
.fake .bar > div {background: linear-gradient(90deg, #f97316, #ef4444);}
.real .bar > div {background: linear-gradient(90deg, #22c55e, #16a34a);}
.bar-labels {display: flex; justify-content: space-between; font-size: .78rem; opacity: .7; margin-top: .3rem;}

.chips-title {font-weight: 700; margin: 1.1rem 0 .4rem 0;}
.chip {
    display: inline-block; padding: .22rem .7rem; margin: .15rem .25rem .15rem 0;
    border-radius: 999px; font-size: .88rem; font-weight: 600;
}
.chip.f {background: rgba(239, 68, 68, 0.16); color: #ef4444;}
.chip.r {background: rgba(34, 197, 94, 0.18); color: #16a34a;}

.stButton > button {border-radius: 12px; font-weight: 600;}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed); border: none; color: #fff;
}
.footer-note {text-align: center; font-size: .8rem; opacity: .6; margin-top: 2rem;}
</style>
"""


def set_sample(key):
    title, body = SAMPLES[key]
    st.session_state["title_in"] = title
    st.session_state["body_in"] = body


def chips(words, kind):
    if not words:
        return "<span style='opacity:.6'>none found</span>"
    return "".join(f"<span class='chip {kind}'>{w}</span>" for w in words)


def render_result(p_fake, fake_w, real_w):
    is_fake = p_fake >= 0.5
    css = "fake" if is_fake else "real"
    verdict = "🚨 Likely FAKE" if is_fake else "✅ Likely REAL"
    conf = p_fake if is_fake else 1 - p_fake
    st.markdown(
        f"""
        <div class="result {css}">
            <p class="badge">{verdict}</p>
            <p class="sub">The model is {conf:.1%} confident in this prediction.</p>
            <div class="bar"><div style="width:{p_fake * 100:.1f}%"></div></div>
            <div class="bar-labels"><span>Real</span><span>Fake probability: {p_fake:.1%}</span><span>Fake</span></div>
            <div class="chips-title">Words pushing towards Fake</div>{chips(fake_w, "f")}
            <div class="chips-title">Words pushing towards Real</div>{chips(real_w, "r")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>📰 Fake News Detector</h1>
            <p>Paste a news article and a machine-learning model will tell you whether
            its language looks like fake or genuine news.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, accuracy = load_model()
    except FileNotFoundError:
        st.error("Fake.csv and True.csv were not found. Put them in the same folder as app.py and rerun.")
        st.stop()

    n_features = len(model.named_steps["tfidf"].get_feature_names_out())
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat"><div class="v">{accuracy:.1%}</div><div class="k">Test accuracy</div></div>
            <div class="stat"><div class="v">Logistic Reg.</div><div class="k">Model</div></div>
            <div class="stat"><div class="v">{n_features:,}</div><div class="k">TF-IDF features</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_check, tab_about = st.tabs(["🔍 Check an article", "ℹ️ About"])

    with tab_check:
        st.caption("Try an example:")
        c1, c2 = st.columns(2)
        keys = list(SAMPLES)
        c1.button(keys[0], on_click=set_sample, args=(keys[0],), use_container_width=True)
        c2.button(keys[1], on_click=set_sample, args=(keys[1],), use_container_width=True)

        title = st.text_input("Headline (optional)", key="title_in",
                              placeholder="e.g. Senate committee approves budget bill")
        body = st.text_area("Article text", key="body_in", height=230,
                            placeholder="Paste the full article here (at least a few sentences)...")
        st.caption(f"{len(body.split())} words")

        if st.button("Check article", type="primary", use_container_width=True):
            cleaned = clean_text(f"{title}. {body}")
            if len(cleaned.split()) < 20:
                st.warning("Please paste at least a few sentences (20+ words) for a meaningful result.")
            else:
                with st.spinner("Analysing..."):
                    p_fake = float(model.predict_proba([cleaned])[0][1])
                    fake_w, real_w = top_words(model, cleaned)
                render_result(p_fake, fake_w, real_w)

    with tab_about:
        st.markdown(
            """
            ### How it works
            1. **Cleaning:** the text is lower-cased and URLs, symbols and the `(Reuters)` source tag are removed.
            2. **TF-IDF:** each article becomes a vector that weights words which are distinctive for that article.
            3. **Logistic Regression:** a linear classifier turns that vector into a probability of being fake.
            4. **Explanation:** the highlighted words are the ones in your text with the largest effect on the result.

            ### Training data
            The [ISOT Fake News dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset/data):
            about 38,000 articles (after removing duplicates), mostly US political news from 2015 to 2018.

            ### Limitations
            * The model learned the writing patterns of that period, so it can be wrong on other topics,
              newer events or other writing styles.
            * It judges **language**, not facts. It cannot verify whether a claim is true.
            * Do not use it to decide what is true in the real world. This is a college project demo.
            """
        )

    st.markdown("<div class='footer-note'>Project 15 · Fake News Detection · Built with Streamlit &amp; scikit-learn</div>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()