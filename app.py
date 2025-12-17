import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================== KONFIGURASI ==================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    layout="wide"
)

# ================== CSS UI ==================
st.markdown("""
<style>
.main { background-color: #f5f7fa; }

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.positif { color: #16a34a; font-weight: bold; font-size: 22px; }
.netral  { color: #6b7280; font-weight: bold; font-size: 22px; }
.negatif { color: #dc2626; font-weight: bold; font-size: 22px; }
</style>
""", unsafe_allow_html=True)

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# ================== FUNGSI ==================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

def label_sentiment(score):
    if score >= 4:
        return "Positif"
    elif score == 3:
        return "Netral"
    else:
        return "Negatif"

def train_model(df):
    X = df["clean"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

    return model, X_test_tfidf, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average="weighted"),
        recall_score(y_test, y_pred, average="weighted"),
        f1_score(y_test, y_pred, average="weighted")
    )

def predict_text(text):
    model = joblib.load("model/model.pkl")
    tfidf = joblib.load("model/tfidf.pkl")
    text = clean_text(text)
    vec = tfidf.transform([text])
    return model.predict(vec)[0]

# ================== HEADER ==================
st.markdown("""
<div class="card">
<h1>📊 Sistem Analisis Sentimen Aplikasi Akulaku</h1>
<p>Metode <b>Naïve Bayes</b> dengan <b>TF-IDF</b> berbasis Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.markdown("## 📌 Menu Utama")
st.sidebar.markdown("---")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Upload Dataset", "Training Model", "Evaluasi Model", "Prediksi Kalimat"]
)

# ================== UPLOAD ==================
if menu == "Upload Dataset":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📁 Upload Dataset CSV")
    file = st.file_uploader("File harus memiliki kolom: content & score", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if file:
        df = pd.read_csv(file)

        if "content" not in df.columns or "score" not in df.columns:
            st.error("CSV harus punya kolom: content & score")
        else:
            df["clean"] = df["content"].apply(clean_text)
            df["sentiment"] = df["score"].apply(label_sentiment)

            st.session_state["df"] = df

            st.success("Dataset berhasil diproses")
            st.dataframe(df.head())

            st.bar_chart(df["sentiment"].value_counts())

# ================== TRAIN ==================
elif menu == "Training Model":
    if "df" not in st.session_state:
        st.warning("Upload dataset terlebih dahulu")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Training Model")
        if st.button("Latih Model"):
            model, X_test, y_test = train_model(st.session_state["df"])
            st.session_state["model_data"] = (model, X_test, y_test)
            st.success("Model berhasil dilatih")
        st.markdown("</div>", unsafe_allow_html=True)

# ================== EVALUASI ==================
elif menu == "Evaluasi Model":
    if "model_data" not in st.session_state:
        st.warning("Latih model terlebih dahulu")
    else:
        model, X_test, y_test = st.session_state["model_data"]
        acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Hasil Evaluasi Model")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("Precision", f"{prec:.2%}")
        col3.metric("Recall", f"{rec:.2%}")
        col4.metric("F1-Score", f"{f1:.2%}")

        st.markdown("</div>", unsafe_allow_html=True)

# ================== PREDIKSI ==================
elif menu == "Prediksi Kalimat":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Prediksi Sentimen Kalimat")
    text = st.text_area("Masukkan kalimat ulasan")
    if st.button("Analisis Sentimen"):
        if text.strip() == "":
            st.warning("Kalimat tidak boleh kosong")
        elif not os.path.exists("model/model.pkl"):
            st.error("Model belum dilatih")
        else:
            hasil = predict_text(text)
            if hasil == "Positif":
                st.markdown("<p class='positif'>✅ Positif</p>", unsafe_allow_html=True)
            elif hasil == "Netral":
                st.markdown("<p class='netral'>⚖️ Netral</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='negatif'>❌ Negatif</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<center><small>© 2025 | Sistem Analisis Sentimen Akulaku "
    "– Naïve Bayes & TF-IDF</small></center>",
    unsafe_allow_html=True
)
