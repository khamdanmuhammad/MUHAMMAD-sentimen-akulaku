import streamlit as st
import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ================== KONFIGURASI ==================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    layout="wide"
)

# ================== CSS ==================
st.markdown("""
<style>
.card {background:white;padding:20px;border-radius:14px;
box-shadow:0 4px 10px rgba(0,0,0,.08);margin-bottom:20px}
.pos{color:#16a34a;font-size:22px;font-weight:bold}
.net{color:#6b7280;font-size:22px;font-weight:bold}
.neg{color:#dc2626;font-size:22px;font-weight:bold}
</style>
""", unsafe_allow_html=True)

# ================== NLTK ==================
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("indonesian"))

# ================== FUNGSI CLEAN ==================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# ================== RULE BASED (PASTI BENAR ARAH) ==================
NEGATIVE_WORDS = ["bajingan", "jelek", "buruk", "penipu", "parah", "sampah", "kecewa"]
POSITIVE_WORDS = ["bagus", "mantap", "baik", "membantu", "recommended", "puas"]

def rule_based_sentiment(text):
    text = text.lower()
    for w in NEGATIVE_WORDS:
        if w in text:
            return "Negatif"
    for w in POSITIVE_WORDS:
        if w in text:
            return "Positif"
    return "Netral"

# ================== LABEL DARI RATING (AMAN) ==================
def label_from_score(score, text):
    try:
        score = float(score)
        if score >= 4:
            return "Positif"
        elif score == 3:
            return "Netral"
        else:
            return "Negatif"
    except:
        # kalau rating rusak → pakai teks
        return rule_based_sentiment(text)

# ================== LOAD CSV AMAN ==================
def load_csv_safe(file):
    for enc in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            pass
    return None

# ================== TRAIN MODEL ==================
def train_model(df):
    X = df["clean"]
    y = df["sentiment"]

    tfidf = TfidfVectorizer(max_features=3000)
    X_vec = tfidf.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

# ================== PREDIKSI AMAN ==================
def predict_safe(text):
    # MODE ML
    if os.path.exists("model/model.pkl"):
        model = joblib.load("model/model.pkl")
        tfidf = joblib.load("model/tfidf.pkl")
        vec = tfidf.transform([clean_text(text)])
        return model.predict(vec)[0]

    # MODE FALLBACK
    return rule_based_sentiment(text)

# ================== HEADER ==================
st.markdown("""
<div class="card">
<h1>📊 Sistem Analisis Sentimen Akulaku</h1>
<p>Anti Error • Anti Kebalik • Siap Sidang • Siap Cloud</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "Menu",
    ["Upload Dataset (Opsional)", "Prediksi Kalimat"]
)

# ================== UPLOAD CSV ==================
if menu == "Upload Dataset (Opsional)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV (bebas format kolom)", type=["csv"])

    if file:
        df = load_csv_safe(file)

        if df is None:
            st.error("❌ CSV tidak dapat dibaca")
        else:
            # DETEKSI KOLOM CERDAS
            text_candidates = ["content", "review", "ulasan", "text"]
            score_candidates = ["score", "rating", "star", "rate"]

            text_col = None
            score_col = None

            for c in df.columns:
                if c.lower() in text_candidates:
                    text_col = c
                if c.lower() in score_candidates:
                    score_col = c

            if text_col is None:
                text_col = df.columns[0]

            df["clean"] = df[text_col].astype(str).apply(clean_text)

            # 🔥 INTI PERBAIKAN
            if score_col is not None:
                df["sentiment"] = df.apply(
                    lambda x: label_from_score(x[score_col], x[text_col]), axis=1
                )
            else:
                df["sentiment"] = df[text_col].apply(rule_based_sentiment)

            train_model(df)

            st.success("✅ Dataset diproses & model berhasil dibuat")
            st.dataframe(df[[text_col, "sentiment"]].head())

    st.markdown("</div>", unsafe_allow_html=True)

# ================== PREDIKSI KALIMAT ==================
elif menu == "Prediksi Kalimat":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    text = st.text_area("Masukkan kalimat ulasan")

    if st.button("Analisis Sentimen"):
        hasil = predict_safe(text)

        if hasil == "Positif":
            st.markdown("<p class='pos'>✅ Positif</p>", unsafe_allow_html=True)
        elif hasil == "Netral":
            st.markdown("<p class='net'>⚖️ Netral</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='neg'>❌ Negatif</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
