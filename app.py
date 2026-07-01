import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

# ================== KONFIG ==================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    layout="wide"
)

# ================== STYLE ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    color: #000000;
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #ecfdf5);
    border-right: 1px solid #bbf7d0;
}
section[data-testid="stSidebar"] * {
    color: #000000 !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stSelectbox > div {
    background-color: #ffffff;
    border-radius: 14px;
    box-shadow: 0 6px 14px rgba(16, 185, 129, 0.15);
    border: 1px solid #bbf7d0;
}
section[data-testid="stSidebar"] li[aria-selected="true"] {
    background: linear-gradient(135deg, #86efac, #4ade80);
    border-radius: 10px;
    font-weight: 700;
}
.stButton > button {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    color: #000000;
    border-radius: 14px;
    font-weight: 700;
    padding: 0.6em 1.4em;
    border: none;
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

# ================== KAMUS ==================
NEGATIVE_WORDS = ["kecewa","buruk","jelek","lambat","ribet","error","parah",
                  "anjing","bangsat","kontol","tai","penipu","tolol","bodoh"]
NEUTRAL_WORDS = ["lumayan","biasa","cukup","standar","oke"]
POSITIVE_WORDS = ["bagus","baik","mantap","membantu","recommended",
                  "good","suka","cepat","aman"]

def auto_rating_from_text(text):
    if any(w in text for w in NEGATIVE_WORDS):
        return 1
    elif any(w in text for w in POSITIVE_WORDS):
        return 5
    elif any(w in text for w in NEUTRAL_WORDS):
        return 3
    else:
        return 3

def sentiment_from_rating(rating):
    rating = int(rating)
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

# -------- CSS khusus menu ala Webframe --------
st.markdown("""
<style>
.sidebar-title{
    font-size:18px;
    font-weight:700;
    margin-bottom:6px;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: #ffffff !important;
    border-radius: 16px;
    padding: 12px 14px !important;
    margin-bottom: 10px !important;
    border:1px solid #bbf7d0 !important;
    box-shadow: 0 6px 14px rgba(16, 185, 129, .18);
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background:#ecfdf5 !important;
    border:1px solid #4ade80 !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label div:first-child{
    font-size:15px !important;
    font-weight:600 !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"]{
    background:linear-gradient(135deg,#86efac,#4ade80) !important;
    color:#000 !important;
    font-weight:700 !important;
}
</style>
""", unsafe_allow_html=True)

# -------- Judul Sidebar --------
st.sidebar.markdown("<p class='sidebar-title'>📌 Menu</p>", unsafe_allow_html=True)

# -------- Menu Ala Webframe (Radio Button) --------
menu = st.sidebar.radio(
    "",
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

        text_col = next((c for c in df.columns if any(k in c.lower() for k in ["review","ulasan","content","text"])), df.columns[0])
        rating_col = next((c for c in df.columns if any(k in c.lower() for k in ["rating","score","bintang"])), None)

        df["clean_text"] = df[text_col].apply(clean_text)

        if rating_col:
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
            df["rating_auto"] = df[rating_col]
        else:
            df["rating_auto"] = df["clean_text"].apply(auto_rating_from_text)

        df["sentiment"] = df["rating_auto"].apply(sentiment_from_rating)

        st.session_state.df = df
        st.success("✅ Dataset berhasil diproses")
        st.dataframe(df.head())

# ================== PREDIKSI ==================
elif menu == "✍️ Prediksi Kalimat":
    text = st.text_area("Masukkan ulasan")
    if st.button("Analisis"):
        clean = clean_text(text)
        rating = auto_rating_from_text(clean)
        sentiment = sentiment_from_rating(rating)
        st.success(f"Rating: {rating} | Sentimen: {sentiment}")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        counts = df["sentiment"].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
            st.pyplot(fig)

        # ===== WORDCLOUD =====
        st.subheader("☁️ WordCloud")
        text_all = " ".join(df["clean_text"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # ===== CONFUSION MATRIX =====
        st.subheader("📌 Confusion Matrix")
        X = df["clean_text"]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i,j], ha="center", va="center")

        st.pyplot(fig)
        st.dataframe(df)

# ================== MODEL ==================
elif menu == "🧠 Modeling & Evaluasi":
    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        df = st.session_state.df
        X = df["clean_text"]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
        # st.text(classification_report(y_test, y_pred))
        laporan = classification_report(y_test, y_pred, output_dict=True)
        df_laporan = pd.DataFrame(laporan).transpose()
        st.table(df_laporan)

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download CSV":
    if st.session_state.df is not None:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "hasil_sentimen.csv", "text/csv")
