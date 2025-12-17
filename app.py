import streamlit as st
import pandas as pd
import re
import nltk
import joblib
import os
import matplotlib.pyplot as plt

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

# ================== HEADER ==================
st.markdown("""
<div class="card">
<h1>📊 Sistem Analisis Sentimen Akulaku</h1>
<p>Anti Error • Anti Kebalik • Siap Sidang • Siap Cloud</p>
</div>
""", unsafe_allow_html=True)

# ================== MENU ==================
menu = st.sidebar.selectbox(
    "Menu",
    ["Upload Dataset (Opsional)", "Prediksi Kalimat", "Dashboard"]
)

# ================== NLTK AMAN ==================
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

# ================== RULE BASED ==================
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

# ================== LOAD CSV ==================
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

# ================== PREDIKSI ==================
def predict_safe(text):
    if os.path.exists("model/model.pkl"):
        model = joblib.load("model/model.pkl")
        tfidf = joblib.load("model/tfidf.pkl")
        vec = tfidf.transform([clean_text(text)])
        return model.predict(vec)[0]
    return rule_based_sentiment(text)

# ================== UPLOAD DATASET ==================
if menu == "Upload Dataset (Opsional)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = load_csv_safe(file)

        if df is None:
            st.error("❌ CSV tidak bisa dibaca")
        else:
            text_col = df.columns[0]

            df["clean"] = df[text_col].astype(str).apply(clean_text)
            df["sentiment"] = df[text_col].astype(str).apply(rule_based_sentiment)

            train_model(df)

            # SIMPAN KE SESSION
            st.session_state["df_result"] = df.copy()
            st.session_state["text_col"] = text_col

            st.success("✅ Dataset berhasil diproses")
            st.dataframe(df[[text_col, "sentiment"]].head())

    st.markdown("</div>", unsafe_allow_html=True)

# ================== PREDIKSI KALIMAT ==================
elif menu == "Prediksi Kalimat":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    text = st.text_area("Masukkan kalimat ulasan")

    if st.button("Analisis Sentimen"):
        if not os.path.exists("model/model.pkl"):
            st.info("ℹ️ Model belum dilatih, memakai rule-based")

        hasil = predict_safe(text)

        if hasil == "Positif":
            st.markdown("<p class='pos'>✅ Positif</p>", unsafe_allow_html=True)
        elif hasil == "Netral":
            st.markdown("<p class='net'>⚖️ Netral</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='neg'>❌ Negatif</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================== DASHBOARD ==================
elif menu == "Dashboard":

    if "df_result" not in st.session_state:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state["df_result"]
    text_col = st.session_state["text_col"]

    st.markdown("<div class='card'><h2>📊 Dashboard Sentimen</h2></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    pos_count = (df["sentiment"] == "Positif").sum()
    net_count = (df["sentiment"] == "Netral").sum()
    neg_count = (df["sentiment"] == "Negatif").sum()

    col1.markdown(f"<p class='pos'>Positif<br>{pos_count}</p>", unsafe_allow_html=True)
    col2.markdown(f"<p class='net'>Netral<br>{net_count}</p>", unsafe_allow_html=True)
    col3.markdown(f"<p class='neg'>Negatif<br>{neg_count}</p>", unsafe_allow_html=True)

    st.dataframe(df[[text_col, "sentiment"]], use_container_width=True)

    csv = df[[text_col, "sentiment"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Ulasan + Sentimen",
        csv,
        "hasil_ulasan_sentimen_akulaku.csv",
        "text/csv"
    )

    fig, ax = plt.subplots()
    df["sentiment"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
