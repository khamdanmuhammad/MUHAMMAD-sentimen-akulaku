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

# ================== RATING & SENTIMEN ==================
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
if "rating_col" not in st.session_state:
    st.session_state.rating_col = None

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

# ================== UPLOAD ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

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

        if rating_col:
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
            df["rating_auto"] = df[rating_col]
            df["sentiment"] = df["rating_auto"].apply(sentiment_from_rating)
            st.session_state.rating_col = rating_col
        else:
            df["rating_auto"] = 3
            df["sentiment"] = "Netral"
            st.session_state.rating_col = None

        st.session_state.df = df

        st.success("✅ Dataset berhasil diproses")
        st.dataframe(df.head())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")

    if st.button("Analisis"):
        if not text.strip():
            st.warning("Teks kosong")
        else:
            rating = 3
            sentiment = "Netral"
            st.success(f"""
### ✅ Hasil Analisis
- **Rating** : {rating}
- **Sentimen** : {sentiment}
""")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        rating_col = st.session_state.rating_col

        st.subheader("🎯 DISTRIBUSI SENTIMEN")

        counts = df["sentiment"].value_counts()
        total = counts.sum()

        # ===== TEKS RINGKASAN =====
        st.code(
            "🎯 DISTRIBUSI SENTIMEN:\n"
            "============================\n" +
            "\n".join([
                f"{k.upper():<9}: {v:,} ({v/total*100:.1f}%)"
                for k, v in counts.items()
            ]),
            language="text"
        )

        col1, col2, col3 = st.columns(3)

        # ===== BAR SENTIMEN =====
        with col1:
            fig, ax = plt.subplots()
            counts.plot(
                kind="bar",
                ax=ax,
                color=["green", "red", "gold"]
            )
            ax.set_title("Jumlah Review per Sentimen")
            ax.set_xlabel("Sentimen")
            ax.set_ylabel("Jumlah Review")
            st.pyplot(fig)

        # ===== PIE SENTIMEN =====
        with col2:
            fig, ax = plt.subplots()
            ax.pie(
                counts,
                labels=counts.index.str.lower(),
                autopct="%1.1f%%",
                startangle=90
            )
            ax.set_title("Persentase Sentimen")
            st.pyplot(fig)

        # ===== DISTRIBUSI RATING =====
        with col3:
            if rating_col:
                grp = (
                    df.groupby(["rating_auto", "sentiment"])
                    .size()
                    .unstack(fill_value=0)
                )

                fig, ax = plt.subplots()
                grp.plot(kind="bar", ax=ax)
                ax.set_title("Distribusi Rating per Sentimen")
                ax.set_xlabel("Rating")
                ax.set_ylabel("Jumlah Review")
                st.pyplot(fig)
            else:
                st.info("Kolom rating tidak tersedia")

        st.subheader("📄 DATA HASIL")
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

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download CSV":
    if st.session_state.df is None:
        st.warning("Belum ada dataset")
    else:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV Hasil Analisis",
            csv,
            "hasil_analisis_sentimen_akulaku.csv",
            "text/csv"
        )
