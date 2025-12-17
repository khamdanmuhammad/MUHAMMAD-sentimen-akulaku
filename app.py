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
<p>Upload Dataset • Training Otomatis • Prediksi • Dashboard • Download CSV</p>
</div>
""", unsafe_allow_html=True)

# ================== MENU ==================
menu = st.sidebar.selectbox(
    "📌 Menu",
    ["Upload Dataset (Opsional)", "Prediksi Kalimat", "Dashboard"]
)

# ================== NLTK ==================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("indonesian"))

# ================== CLEAN TEXT ==================
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

    st.markdown("### 📂 Upload Dataset CSV")
    st.write("✔ Mendukung berbagai encoding (UTF-8, Latin1, ISO)")
    st.write("✔ Training otomatis TF-IDF & Naïve Bayes")

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file:
        df = load_csv_safe(file)

        if df is None:
            st.error("❌ File CSV tidak bisa dibaca")
        else:
            text_col = df.columns[0]

            df["clean"] = df[text_col].astype(str).apply(clean_text)
            df["sentiment"] = df[text_col].astype(str).apply(rule_based_sentiment)

            train_model(df)

            st.session_state["df_result"] = df.copy()
            st.session_state["text_col"] = text_col

            st.success("✅ Dataset berhasil diproses & model dilatih")
            st.dataframe(df[[text_col, "sentiment"]].head())

    st.markdown("</div>", unsafe_allow_html=True)

# ================== PREDIKSI ==================
elif menu == "Prediksi Kalimat":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### ✍️ Prediksi Sentimen Kalimat")
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

# ================== DASHBOARD ==================
elif menu == "Dashboard":

    if "df_result" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state["df_result"]
    text_col = st.session_state["text_col"]

    st.markdown("<div class='card'><h2>📊 Dashboard Analisis Sentimen</h2></div>", unsafe_allow_html=True)

    with st.expander("📌 Fitur Dashboard"):
        st.markdown("""
        - 📂 Upload dataset CSV  
        - 🧠 Training otomatis TF-IDF & Naïve Bayes  
        - ✍️ Prediksi sentimen real-time  
        - 📊 Ringkasan sentimen  
        - 📈 Visualisasi grafik batang  
        - ⬇️ Download hasil ulasan + sentimen  
        - 🛡️ Fallback rule-based  
        """)

    col1, col2, col3 = st.columns(3)
    pos = (df["sentiment"] == "Positif").sum()
    net = (df["sentiment"] == "Netral").sum()
    neg = (df["sentiment"] == "Negatif").sum()

    col1.markdown(f"<div class='card'><p class='pos'>Positif<br>{pos}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><p class='net'>Netral<br>{net}</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><p class='neg'>Negatif<br>{neg}</p></div>", unsafe_allow_html=True)

    st.markdown("### 📄 Data Ulasan & Sentimen")
    st.dataframe(df[[text_col, "sentiment"]], use_container_width=True)

    st.markdown("### 📈 Visualisasi Grafik Batang")
    fig, ax = plt.subplots()
    df["sentiment"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    st.markdown("### ⬇️ Download Hasil Ulasan")
    csv = df[[text_col, "sentiment"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Ulasan + Sentimen (CSV)",
        csv,
        "hasil_ulasan_sentimen_akulaku.csv",
        "text/csv"
    )
