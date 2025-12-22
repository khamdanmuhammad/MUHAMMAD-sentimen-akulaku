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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ================== KONFIG ==================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    layout="wide"
)

# ================== STYLE ==================
st.markdown("""
<style>
.main {background-color: #f8fafc;}
h1, h2, h3 {color: #0f172a;}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.6em 1.2em;
}
.stTextArea textarea {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

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
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ================== RULE BASED ==================
NEGATIVE_STRONG = [
    "anjing","bangsat","kontol","tai","bajingan","penipu",
    "tidak bisa","gagal","error","ditolak","parah",
    "kecewa","ribet","susah","bohong","limit tidak",
    "verifikasi lama","tidak masuk","pending"
]

POSITIVE_STRONG = [
    "bagus","mantap","puas","membantu","cepat",
    "mudah","lancar","recommended","rekomendasi",
    "berfungsi dengan baik","sangat puas"
]

def rule_based_sentiment(text):
    text = str(text).lower()
    for w in NEGATIVE_STRONG:
        if w in text:
            return "Negatif"
    for w in POSITIVE_STRONG:
        if w in text:
            return "Positif"
    return None

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

# ================== UTIL ==================
def detect_column(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

def load_csv_safe(file):
    for enc in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

# ================== TRAIN MODEL ==================
def train_model(df, text_col, label_col):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
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
        "🧠 Modeling & Evaluasi",
        "⬇️ Download"
    ]
)

# ================== UPLOAD DATASET ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload file CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)
        if df is None:
            st.error("❌ Gagal membaca file CSV")
            st.stop()

        text_col = detect_column(df, ["review","ulasan","content","text","komentar"])
        if text_col is None:
            text_col = df.columns[0]

        label_col = detect_column(df, ["sentiment","label","polarity"])
        if label_col is None:
            df["sentiment"] = df[text_col].apply(rule_based_sentiment).fillna("Netral")
            label_col = "sentiment"

        df[text_col] = df[text_col].apply(clean_text)
        df[label_col] = df[label_col].apply(normalize_label)
        df = df[df[text_col].str.len() > 3]

        train_model(df, text_col, label_col)

        st.session_state.df = df
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col

        st.success("✅ Dataset berhasil diproses & model dilatih")
        st.bar_chart(df[label_col].value_counts())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    st.subheader("✍️ Prediksi Sentimen")

    text = st.text_area("Masukkan ulasan", height=120)

    if st.button("🔍 Analisis"):
        if not text.strip():
            st.warning("⚠️ Teks kosong")
        elif not os.path.exists("model/model.pkl"):
            st.error("⚠️ Model belum tersedia")
        else:
            rule = rule_based_sentiment(text)
            if rule:
                st.success(f"Hasil: {rule} (Rule-Based)")
            else:
                model = joblib.load("model/model.pkl")
                tfidf = joblib.load("model/tfidf.pkl")
                pred = model.predict(tfidf.transform([clean_text(text)]))[0]
                st.success(f"Hasil: {pred} (Machine Learning)")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
    else:
        st.subheader("=== LABELING SENTIMEN ===")
        df = st.session_state.df
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Distribusi Sentimen")
        fig, ax = plt.subplots()
        df[st.session_state.label_col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ================== MODELING & EVALUASI ==================
elif menu == "🧠 Modeling & Evaluasi":
    if "df" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        text_col = st.session_state.text_col
        label_col = st.session_state.label_col

        X_train, X_test, y_train, y_test = train_test_split(
            df[text_col], df[label_col],
            test_size=0.2, random_state=42, stratify=df[label_col]
        )

        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = MultinomialNB(alpha=0.5)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"🎯 Akurasi Model: {acc:.4f}")

        st.text(classification_report(y_test, y_pred))

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download":
    if "df" not in st.session_state:
        st.warning("⚠️ Tidak ada data untuk di-download")
    else:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV Hasil Sentimen",
            csv,
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )
