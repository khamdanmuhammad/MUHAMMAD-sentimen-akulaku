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
<p>Upload • Training • Prediksi • Dashboard • Download</p>
</div>
""", unsafe_allow_html=True)

# ================== MENU ==================
menu = st.sidebar.selectbox(
    "📌 Menu",
    [
        "📂 Upload Dataset",
        "✍️ Prediksi Kalimat",
        "📊 Dashboard",
        "⬇️ Download Hasil",
        "⚙️ Pengaturan Grafik"
    ]
)

# ================== NLTK ==================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("indonesian"))

# ================== CLEAN ==================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# ================== RULE BASED ==================
NEGATIVE_WORDS = ["bajingan","jelek","buruk","penipu","parah","sampah","kecewa"]
POSITIVE_WORDS = ["bagus","mantap","baik","membantu","recommended","puas"]

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
    for enc in ["utf-8","latin1","ISO-8859-1"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            pass
    return None

# ================== TRAIN ==================
def train_model(df):
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["clean"])
    y = df["sentiment"]

    model = MultinomialNB()
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

# ================== PREDIKSI ==================
def predict_safe(text):
    if os.path.exists("model/model.pkl"):
        model = joblib.load("model/model.pkl")
        tfidf = joblib.load("model/tfidf.pkl")
        return model.predict(tfidf.transform([clean_text(text)]))[0]
    return rule_based_sentiment(text)

# ================== MENU UPLOAD ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)
        if df is not None:
            text_col = df.columns[0]
            df["clean"] = df[text_col].astype(str).apply(clean_text)
            df["sentiment"] = df[text_col].astype(str).apply(rule_based_sentiment)

            train_model(df)

            st.session_state.df = df
            st.session_state.text_col = text_col
            st.success("✅ Dataset & model siap")

# ================== MENU PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")
    if st.button("Analisis"):
        st.success(predict_safe(text))

# ================== MENU DASHBOARD ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("Upload dataset dulu")
    else:
        df = st.session_state.df
        st.dataframe(df[[st.session_state.text_col,"sentiment"]])

# ================== MENU DOWNLOAD ==================
elif menu == "⬇️ Download Hasil":
    if "df" in st.session_state:
        csv = st.session_state.df[
            [st.session_state.text_col,"sentiment"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            csv,
            "hasil_sentimen.csv",
            "text/csv"
        )
    else:
        st.warning("Belum ada data")

# ================== MENU SETTING GRAFIK ==================
elif menu == "⚙️ Pengaturan Grafik":
    if "df" not in st.session_state:
        st.warning("Belum ada data")
    else:
        chart_type = st.selectbox("Pilih Jenis Chart", ["Bar","Pie"])
        data = st.session_state.df["sentiment"].value_counts()

        fig, ax = plt.subplots()
        if chart_type == "Bar":
            data.plot(kind="bar", ax=ax)
        else:
            data.plot(kind="pie", autopct="%1.1f%%", ax=ax)

        st.pyplot(fig)
