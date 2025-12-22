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

# ================== INIT SESSION (ANTI ERROR) ==================
if "df" not in st.session_state:
    st.session_state.df = None
if "text_col" not in st.session_state:
    st.session_state.text_col = None
if "label_col" not in st.session_state:
    st.session_state.label_col = "sentiment"

# ================== CLEAN TEXT ==================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# ================== KATA KASAR (PASTI NEGATIF) ==================
BAD_WORDS = [
    "anjing","bangsat","kontol","tai","bajingan","penipu",
    "tolol","bodoh","goblok","parah","kecewa","bohong",
    "gagal","error","ribet","susah","pending"
]

def contains_bad_words(text):
    text = str(text).lower()
    return any(w in text for w in BAD_WORDS)

# ================== RULE SENTIMENT (FINAL FIX) ==================
def sentiment_from_score_and_text(score, text):
    # kata kasar SELALU negatif
    if contains_bad_words(text):
        return "Negatif"

    try:
        score = int(score)
    except:
        return "Netral"

    if score <= 2:
        return "Negatif"
    elif score == 3:
        return "Netral"
    else:  # 4–5
        return "Positif"

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
    ["📂 Upload Dataset", "✍️ Prediksi Kalimat", "📊 Dashboard", "🧠 Modeling & Evaluasi", "⬇️ Download"]
)

# ================== UPLOAD DATASET ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload file CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)
        if df is None:
            st.error("CSV tidak bisa dibaca")
            st.stop()

        text_col = detect_column(df, ["review","ulasan","content","text"])
        rating_col = detect_column(df, ["rating","score","bintang"])

        if text_col is None or rating_col is None:
            st.error("CSV wajib punya kolom TEKS dan RATING")
            st.stop()

        df["clean_text"] = df[text_col].apply(clean_text)
        df["sentiment"] = df.apply(
            lambda x: sentiment_from_score_and_text(x[rating_col], x[text_col]),
            axis=1
        )

        st.session_state.df = df
        st.session_state.text_col = text_col
        st.session_state.rating_col = rating_col

        st.success("✅ Dataset berhasil diproses (aturan skor sudah BENAR)")
        st.dataframe(df.head())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")
    score = st.selectbox("Pilih skor (0–5)", [0,1,2,3,4,5])

    if st.button("Analisis"):
        hasil = sentiment_from_score_and_text(score, text)
        if hasil == "Negatif":
            st.error(f"Hasil Sentimen: {hasil}")
        elif hasil == "Netral":
            st.info(f"Hasil Sentimen: {hasil}")
        else:
            st.success(f"Hasil Sentimen: {hasil}")

# ================== DASHBOARD (GRAFIK LENGKAP) ==================
elif menu == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df

    st.subheader("🎯 DISTRIBUSI SENTIMEN")

    counts = df["sentiment"].value_counts()
    total = len(df)

    st.text(
        f"""POSITIVE : {counts.get('Positif',0):,}
NEUTRAL  : {counts.get('Netral',0):,}
NEGATIVE : {counts.get('Negatif',0):,}
"""
    )

    col1, col2, col3 = st.columns(3)

    # Bar Sentimen
    with col1:
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax, color=["green","gold","red"])
        ax.set_title("Jumlah Review per Sentimen")
        st.pyplot(fig)

    # Pie
    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=["green","gold","red"],
            startangle=90
        )
        ax.set_title("Persentase Sentimen")
        st.pyplot(fig)

    # Rating vs Sentimen
    with col3:
        grp = df.groupby([st.session_state.rating_col, "sentiment"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        grp.plot(kind="bar", ax=ax)
        ax.set_title("Distribusi Rating per Sentimen")
        st.pyplot(fig)

    st.subheader("📋 DATASET")
    st.dataframe(df, use_container_width=True)

# ================== MODELING ==================
elif menu == "🧠 Modeling & Evaluasi":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df
    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        st.download_button(
            "Download CSV",
            csv,
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )
