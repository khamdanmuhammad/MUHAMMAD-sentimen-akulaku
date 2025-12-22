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

# ================== KAMUS SENTIMEN (UNTUK PREDIKSI TEKS) ==================
NEGATIVE_WORDS = [
    "kecewa","buruk","jelek","lambat","ribet","error","parah","mengecewakan",
    "anjing","bangsat","kontol","tai","bajingan","penipu","tolol","bodoh","goblok"
]

NEUTRAL_WORDS = ["lumayan","biasa","cukup","standar","oke"]

POSITIVE_WORDS = [
    "bagus","baik","mantap","membantu","recommended","good","suka","cepat","aman"
]

# ================== RATING OTOMATIS ==================
def auto_rating_from_text(text):
    text = text.lower()
    if any(w in text for w in NEGATIVE_WORDS):
        return 1
    elif any(w in text for w in NEUTRAL_WORDS):
        return 3
    elif any(w in text for w in POSITIVE_WORDS):
        return 5
    else:
        return 3

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

# ================== SESSION ==================
if "df" not in st.session_state:
    st.session_state.df = None

# ================== UI ==================
st.title("📊 Sistem Analisis Sentimen Akulaku")

menu = st.sidebar.selectbox(
    "📌 Menu",
    [
        "📂 Upload Dataset",
        "✍️ Prediksi Kalimat",
        "📊 Dashboard",
        "🧠 Modeling & Evaluasi",
        "⬇️ Download CSV"
    ]
)

# ================== UPLOAD DATASET ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # Deteksi kolom teks & rating
        text_col = None
        rating_col = None

        for col in df.columns:
            if any(k in col.lower() for k in ["review", "ulasan", "content", "text"]):
                text_col = col
            if any(k in col.lower() for k in ["rating", "score", "bintang"]):
                rating_col = col

        if text_col is None:
            text_col = df.columns[0]

        df["clean_text"] = df[text_col].apply(clean_text)

        # ================== SENTIMEN SESUAI RATING CSV ==================
        if rating_col is not None:
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
            df["rating_auto"] = df[rating_col]
            df["sentiment"] = df["rating_auto"].apply(sentiment_from_rating)

        # ================== JIKA TIDAK ADA RATING ==================
        else:
            df["rating_auto"] = df["clean_text"].apply(auto_rating_from_text)
            df["sentiment"] = df["rating_auto"].apply(sentiment_from_rating)

        st.session_state.df = df

        st.success("✅ Dataset berhasil diproses sesuai skor")
        st.dataframe(df.head())

# ================== PREDIKSI KALIMAT ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")

    if st.button("Analisis"):
        if not text.strip():
            st.warning("Teks kosong")
        else:
            clean = clean_text(text)
            rating = auto_rating_from_text(clean)
            sentiment = sentiment_from_rating(rating)

            st.success(f"""
### ✅ Hasil Analisis Otomatis
- **Rating** : {rating}
- **Sentimen** : {sentiment}
""")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        counts = df["sentiment"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Distribusi Sentimen")
        st.pyplot(fig)

        st.dataframe(df)

# ================== MODELING ==================
elif menu == "🧠 Modeling & Evaluasi":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df

        X = df["clean_text"]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        tfidf = TfidfVectorizer()
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        st.success(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
        st.text(classification_report(y_test, y_pred))

# ================== DOWNLOAD CSV ==================
elif menu == "⬇️ Download CSV":
    if st.session_state.df is None:
        st.warning("Belum ada dataset")
    else:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV Hasil Analisis",
            data=csv,
            file_name="hasil_analisis_sentimen_akulaku.csv",
            mime="text/csv"
        )
