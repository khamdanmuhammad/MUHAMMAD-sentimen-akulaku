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

        st.session_state.df = df
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col

        st.success("✅ Dataset berhasil diproses")
        st.bar_chart(df[label_col].value_counts())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan", height=120)
    if st.button("🔍 Analisis"):
        if not text.strip():
            st.warning("⚠️ Teks kosong")
        else:
            rule = rule_based_sentiment(text)
            if rule:
                st.success(f"Hasil: {rule} (Rule-Based)")
            elif "df" in st.session_state:
                df = st.session_state.df
                tfidf = TfidfVectorizer(max_features=15000)
                X = tfidf.fit_transform(df[st.session_state.text_col])
                y = df[st.session_state.label_col]
                model = MultinomialNB()
                model.fit(X, y)
                pred = model.predict(tfidf.transform([clean_text(text)]))[0]
                st.success(f"Hasil: {pred} (Machine Learning)")
            else:
                st.warning("Upload dataset dulu")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df
    label_col = st.session_state.label_col

    st.subheader("🎯 DISTRIBUSI SENTIMEN")

    counts = df[label_col].value_counts()
    total = len(df)

    for k in ["Positif","Netral","Negatif"]:
        if k not in counts:
            counts[k] = 0

    st.text(
        f"""POSITIVE : {counts['Positif']:,} ({counts['Positif']/total*100:.1f}%)
NEUTRAL  : {counts['Netral']:,} ({counts['Netral']/total*100:.1f}%)
NEGATIVE : {counts['Negatif']:,} ({counts['Negatif']/total*100:.1f}%)
"""
    )

    col1, col2, col3 = st.columns(3)

    # Grafik 1
    with col1:
        fig, ax = plt.subplots()
        counts.loc[["Positif","Negatif","Netral"]].plot(
            kind="bar", ax=ax,
            color=["green","red","gold"]
        )
        ax.set_title("Jumlah Review per Sentimen")
        st.pyplot(fig)

    # Grafik 2
    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            counts.loc[["Positif","Negatif","Netral"]],
            labels=["positive","negative","neutral"],
            autopct="%1.1f%%",
            colors=["green","red","gold"],
            startangle=90
        )
        ax.set_title("Persentase Sentimen")
        st.pyplot(fig)

    # Grafik 3
    with col3:
        rating_col = detect_column(df, ["rating","score","bintang"])
        if rating_col is None:
            st.info("ℹ️ Kolom rating tidak tersedia")
        else:
            fig, ax = plt.subplots()
            grp = df.groupby([rating_col, label_col]).size().unstack(fill_value=0)
            grp.plot(kind="bar", ax=ax)
            ax.set_title("Distribusi Rating per Sentimen")
            st.pyplot(fig)

    st.subheader("=== LABELING SENTIMEN ===")
    st.dataframe(df, use_container_width=True)

# ================== MODELING & EVALUASI ==================
elif menu == "🧠 Modeling & Evaluasi":
    if "df" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(
            df[st.session_state.text_col],
            df[st.session_state.label_col],
            test_size=0.2, random_state=42, stratify=df[st.session_state.label_col]
        )

        tfidf = TfidfVectorizer(max_features=15000)
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        st.success(f"Akurasi: {accuracy_score(y_test,y_pred):.4f}")
        st.text(classification_report(y_test,y_pred))

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download":
    if "df" not in st.session_state:
        st.warning("⚠️ Tidak ada data")
    else:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )
