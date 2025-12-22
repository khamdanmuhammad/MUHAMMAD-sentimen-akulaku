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

# ================== KATA KASAR (PAKSA NEGATIF) ==================
BAD_WORDS = [
    "anjing","bangsat","kontol","tai","bajingan","penipu",
    "tolol","bodoh","goblok","brengsek","keparat"
]

def contains_bad_words(text):
    text = str(text).lower()
    return any(w in text for w in BAD_WORDS)

# ================== SENTIMEN BERDASARKAN RATING ==================
def sentiment_from_rating(rating):
    try:
        rating = int(rating)
    except:
        return "Netral"

    if rating <= 2:
        return "Negatif"
    elif rating == 3:
        return "Netral"
    else:
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

# ================== SESSION STATE (ANTI ERROR) ==================
for key in ["df", "text_col", "label_col", "rating_col"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
            st.stop()

        text_col = detect_column(df, ["review","ulasan","content","text"])
        rating_col = detect_column(df, ["rating","score","bintang"])

        if text_col is None:
            text_col = df.columns[0]

        df["clean_text"] = df[text_col].apply(clean_text)

        # SENTIMEN DARI RATING
        if rating_col:
            df["sentiment"] = df[rating_col].apply(sentiment_from_rating)
        else:
            df["sentiment"] = "Netral"

        # PAKSA NEGATIF JIKA ADA KATA KASAR
        df.loc[df["clean_text"].apply(contains_bad_words), "sentiment"] = "Negatif"

        st.session_state.df = df
        st.session_state.text_col = text_col
        st.session_state.label_col = "sentiment"
        st.session_state.rating_col = rating_col

        st.success("✅ Dataset berhasil dimuat & diproses")
        st.dataframe(df.head())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")
    rating = st.selectbox("Rating (opsional)", [None,1,2,3,4,5])

    if st.button("Analisis"):
        if not text.strip():
            st.warning("Teks kosong")
        elif contains_bad_words(text):
            st.error("Hasil: Negatif (kata kasar terdeteksi)")
        elif rating is not None:
            st.success(f"Hasil: {sentiment_from_rating(rating)}")
        else:
            st.info("Hasil: Netral")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df
    label_col = st.session_state.label_col
    rating_col = st.session_state.rating_col

    st.subheader("📊 DISTRIBUSI SENTIMEN")

    counts = df[label_col].value_counts()

    col1, col2, col3 = st.columns(3)

    # BAR
    with col1:
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax, color=["green","red","gold"])
        ax.set_title("Jumlah Review per Sentimen")
        st.pyplot(fig)

    # PIE
    with col2:
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Persentase Sentimen")
        st.pyplot(fig)

    # RATING
    with col3:
        if rating_col:
            fig, ax = plt.subplots()
            grp = df.groupby([rating_col, label_col]).size().unstack(fill_value=0)
            grp.plot(kind="bar", ax=ax)
            ax.set_title("Distribusi Rating per Sentimen")
            st.pyplot(fig)
        else:
            st.info("Kolom rating tidak tersedia")

    st.subheader("📄 DATA LABELING")
    st.dataframe(df)

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
        st.download_button("Download CSV", csv, "hasil_sentimen.csv", "text/csv")
