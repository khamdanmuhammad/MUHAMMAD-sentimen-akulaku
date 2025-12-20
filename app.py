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

# ================== KONFIG ==================
st.set_page_config(page_title="Analisis Sentimen Akulaku", layout="wide")

# ================== NLTK ==================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("indonesian"))

# ================== CLEAN TEXT ==================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ================== RULE-BASED SENTIMENT ==================
def rule_based_sentiment(text):
    text = str(text).lower()

    # pola keluhan (NEGATIF KUAT)
    negative_patterns = [
        "tidak bisa", "tidak dapat", "tidak berhasil",
        "gagal", "ditolak", "error", "masalah",
        "verifikasi lama", "limit tidak", "tidak masuk",
        "susah", "ribet", "kecewa", "parah"
    ]

    # pola kepuasan (POSITIF KUAT)
    positive_patterns = [
        "sangat membantu", "sangat puas", "mudah digunakan",
        "cepat", "lancar", "berfungsi dengan baik",
        "tidak ada masalah", "mantap", "bagus"
    ]

    for p in negative_patterns:
        if p in text:
            return "Negatif"

    for p in positive_patterns:
        if p in text:
            return "Positif"

    # fallback logis (BUKAN RANDOM)
    if any(k in text for k in ["tidak", "nggak", "belum"]):
        return "Negatif"

    return "Netral"


# ================== NORMALISASI LABEL ==================
def normalize_label(x):
    if isinstance(x, str):
        x = x.lower()
        if "pos" in x:
            return "Positif"
        if "neg" in x:
            return "Negatif"
        if "net" in x:
            return "Netral"
    return "Netral"

# ================== DETEKSI KOLOM ==================
def detect_column(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

# ================== LOAD CSV ==================
def load_csv_safe(file):
    for enc in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

# ================== TRAIN MODEL ==================
def train_model(df, text_col, label_col):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000
    )
    X = vectorizer.fit_transform(df[text_col])
    y = df[label_col]

    model = MultinomialNB(alpha=0.5)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(vectorizer, "model/tfidf.pkl")

# ================== UI ==================
st.title("📊 Sistem Analisis Sentimen Akulaku")

menu = st.sidebar.selectbox(
    "📌 Menu",
    [
        "📂 Upload Dataset",
        "✍️ Prediksi Kalimat",
        "📊 Dashboard",
        "⬇️ Download",
        "⚙️ Pengaturan Grafik"
    ]
)

# ================== UPLOAD DATASET ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)

        if df is None:
            st.error("Gagal membaca file CSV")
            st.stop()

        # Deteksi kolom teks
        text_col = detect_column(df, ["review", "ulasan", "komentar", "content", "text"])
        if text_col is None:
            text_col = df.columns[0]  # fallback aman

        # Deteksi / buat label
        label_col = detect_column(df, ["sentiment", "label", "polarity"])
        if label_col is None:
            st.warning("Kolom label tidak ditemukan → label dibuat otomatis (rule-based)")
            df["sentiment"] = df[text_col].astype(str).apply(rule_based_sentiment)
            st.write("Contoh 20 label pertama:", df[["sentiment"]].head(20))

            label_col = "sentiment"

        # Preprocessing
        df[text_col] = df[text_col].apply(clean_text)
        df[label_col] = df[label_col].apply(normalize_label)
        df = df[df[text_col].str.len() > 3]

        st.subheader("Distribusi Sentimen")
        st.bar_chart(df[label_col].value_counts())

        train_model(df, text_col, label_col)

        st.session_state.df = df
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col

        st.success("✅ Dataset berhasil diproses & model dilatih")

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")
    if st.button("Analisis"):
        if not os.path.exists("model/model.pkl"):
            st.error("Model belum dilatih")
        else:
            model = joblib.load("model/model.pkl")
            tfidf = joblib.load("model/tfidf.pkl")
            pred = model.predict(tfidf.transform([clean_text(text)]))[0]
            st.success(pred)

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("Upload dataset dulu")
    else:
        df = st.session_state.df
        label_col = st.session_state.label_col

        st.dataframe(df[[st.session_state.text_col, label_col]])

        st.subheader("Distribusi Sentimen")
        counts = df[label_col].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download":
    if "df" in st.session_state:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "hasil_sentimen.csv", "text/csv")

# ================== SETTING GRAFIK ==================
elif menu == "⚙️ Pengaturan Grafik":
    if "df" not in st.session_state:
        st.warning("Belum ada data")
    else:
        chart_type = st.selectbox("Pilih Jenis Chart", ["Bar", "Pie"])
        data = st.session_state.df[st.session_state.label_col].value_counts()

        fig, ax = plt.subplots()
        if chart_type == "Bar":
            data.plot(kind="bar", ax=ax)
        else:
            data.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")

        st.pyplot(fig)
