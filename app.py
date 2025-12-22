import streamlit as st
import pandas as pd
import re
import nltk
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
.stApp {
    background: linear-gradient(135deg, #e0f2fe, #fefce8);
    color: #1e293b;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #1e40af);
}
section[data-testid="stSidebar"] * {
    color: black !important;
}
.stButton>button {
    background: linear-gradient(135deg, #facc15, #eab308);
    color: black;
    border-radius: 10px;
    font-weight: 700;
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
NEGATIVE_STRONG = ["gagal","error","parah","kecewa","bohong","pending"]
POSITIVE_STRONG = ["bagus","mantap","puas","mudah","cepat","rekomendasi"]

def rule_based_sentiment(text):
    text = str(text).lower()
    for w in NEGATIVE_STRONG:
        if w in text:
            return "Negatif"
    for w in POSITIVE_STRONG:
        if w in text:
            return "Positif"
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

# ================== INIT SESSION STATE (ANTI ERROR) ==================
if "df" not in st.session_state:
    st.session_state.df = None
if "text_col" not in st.session_state:
    st.session_state.text_col = None
if "label_col" not in st.session_state:
    st.session_state.label_col = None

# ================== UI ==================
st.title("📊 Sistem Analisis Sentimen Akulaku")

menu = st.sidebar.selectbox(
    "📌 Menu",
    ["📂 Upload Dataset", "✍️ Prediksi Kalimat", "📊 Dashboard", "🧠 Modeling & Evaluasi", "⬇️ Download"]
)

# ================== UPLOAD ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)
        if df is None:
            st.error("CSV tidak bisa dibaca")
        else:
            text_col = detect_column(df, ["review","ulasan","content","text"])
            if text_col is None:
                text_col = df.columns[0]

            df["sentiment"] = df[text_col].apply(rule_based_sentiment)
            df[text_col] = df[text_col].apply(clean_text)

            st.session_state.df = df
            st.session_state.text_col = text_col
            st.session_state.label_col = "sentiment"

            st.success("Dataset berhasil dimuat")
            st.dataframe(df.head())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")
    if st.button("Analisis"):
        if not text.strip():
            st.warning("Teks kosong")
        else:
            st.success(f"Hasil: {rule_based_sentiment(text)}")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        label_col = st.session_state.label_col

        st.subheader("Distribusi Sentimen")
        counts = df[label_col].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax, color=["#2563eb","#dc2626","#facc15"])
        st.pyplot(fig)

        st.dataframe(df)

# ================== MODELING ==================
elif menu == "🧠 Modeling & Evaluasi":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        X = df[st.session_state.text_col]
        y = df[st.session_state.label_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        tfidf = TfidfVectorizer()
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        st.success(f"Akurasi: {accuracy_score(y_test,y_pred):.4f}")
        st.text(classification_report(y_test,y_pred))

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download":
    if st.session_state.df is None:
        st.warning("Tidak ada data")
    else:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "hasil_sentimen.csv", "text/csv")
