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
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))

# ================== FUNGSI DASAR ==================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

def label_sentiment(score):
    if score >= 4:
        return "Positif"
    elif score == 3:
        return "Netral"
    else:
        return "Negatif"

# ================== RULE BASED (FALLBACK) ==================
NEGATIVE_WORDS = ["bajingan", "jelek", "buruk", "penipu", "parah", "sampah"]
POSITIVE_WORDS = ["bagus", "mantap", "baik", "membantu", "recommended"]

def rule_based_sentiment(text):
    text = text.lower()
    for w in NEGATIVE_WORDS:
        if w in text:
            return "Negatif"
    for w in POSITIVE_WORDS:
        if w in text:
            return "Positif"
    return "Netral"

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

    return model, tfidf

# ================== PREDIKSI AMAN ==================
def predict_safe(text):
    # MODE 1: MODEL ML
    if os.path.exists("model/model.pkl"):
        model = joblib.load("model/model.pkl")
        tfidf = joblib.load("model/tfidf.pkl")
        vec = tfidf.transform([clean_text(text)])
        return model.predict(vec)[0]
    # MODE 2: FALLBACK
    return rule_based_sentiment(text)

# ================== HEADER ==================
st.markdown("""
<div class="card">
<h1>📊 Sistem Analisis Sentimen Akulaku</h1>
<p>Anti Error • Siap Sidang • Siap Cloud</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "Menu",
    ["Upload Dataset (Opsional)", "Prediksi Kalimat"]
)

# ================== UPLOAD (TIDAK WAJIB) ==================
if menu == "Upload Dataset (Opsional)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV (opsional)", type=["csv"])

    if file:
        df = load_csv_safe(file)

        if df is None:
            st.error("CSV tidak dapat dibaca")
        else:
            # AUTO DETECT KOLOM
            text_col = df.columns[0]
            score_col = df.columns[-1]

            df["clean"] = df[text_col].apply(clean_text)
            df["sentiment"] = df[score_col].apply(label_sentiment)

            train_model(df)
            st.success("✅ Model berhasil dibuat dari dataset")
            st.dataframe(df.head())

    st.markdown("</div>", unsafe_allow_html=True)

# ================== PREDIKSI ==================
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
