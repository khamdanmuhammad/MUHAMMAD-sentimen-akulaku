import streamlit as st
import pandas as pd
import re
import nltk
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

# ================== STYLE (BIRU & KUNING – SIDEBAR TEKS HITAM) ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f2fe 0%, #fefce8 55%, #fff7cc 100%);
    color: #1e293b;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #1e40af);
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 12px;
}
section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Card */
div[data-testid="stVerticalBlock"],
div[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 10px 28px rgba(30, 64, 175, 0.15);
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #facc15, #eab308);
    color: #1e293b;
    border-radius: 12px;
    padding: 0.6em 1.4em;
    font-weight: 700;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #fde047, #facc15);
}

/* Input */
textarea, input {
    border-radius: 10px !important;
    border: 1px solid #fde68a !important;
}

footer {visibility: hidden;}
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
    return " ".join([w for w in text.split() if w not in stop_words and len(w) > 2])

# ================== RULE BASED ==================
NEGATIVE_STRONG = ["gagal","error","parah","kecewa","ribet","susah","bohong","pending"]
POSITIVE_STRONG = ["bagus","mantap","puas","mudah","cepat","lancar","rekomendasi"]

def rule_based_sentiment(text):
    t = str(text).lower()
    for w in NEGATIVE_STRONG:
        if w in t:
            return "Negatif"
    for w in POSITIVE_STRONG:
        if w in t:
            return "Positif"
    return None

# ================== UTIL ==================
def detect_column(df, keys):
    for c in df.columns:
        for k in keys:
            if k in c.lower():
                return c
    return None

def load_csv_safe(file):
    for enc in ["utf-8","latin1","ISO-8859-1"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            pass
    return None

# ================== UI ==================
st.title("📊 Sistem Analisis Sentimen Akulaku")

menu = st.sidebar.selectbox(
    "📌 Menu",
    ["📂 Upload Dataset","✍️ Prediksi Kalimat","📊 Dashboard","🧠 Modeling & Evaluasi","⬇️ Download"]
)

# ================== UPLOAD ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)
        if df is None:
            st.error("CSV tidak bisa dibaca")
            st.stop()

        text_col = detect_column(df, ["review","ulasan","content","text"]) or df.columns[0]
        label_col = detect_column(df, ["sentiment","label"]) 

        if label_col is None:
            df["sentiment"] = df[text_col].apply(rule_based_sentiment).fillna("Netral")
            label_col = "sentiment"

        df[text_col] = df[text_col].apply(clean_text)
        df = df[df[text_col].str.len() > 3]

        st.session_state.df = df
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col

        st.success("Dataset berhasil diproses")

# ================== DASHBOARD (SESUI GAMBAR) ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("Upload dataset terlebih dahulu")
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
NEGATIVE : {counts['Negatif']:,} ({counts['Negatif']/total*100:.1f}%)"""
    )

    c1, c2, c3 = st.columns(3)

    # Bar
    with c1:
        fig, ax = plt.subplots()
        counts.loc[["Positif","Negatif","Netral"]].plot(
            kind="bar", ax=ax,
            color=["#22c55e","#ef4444","#facc15"]
        )
        ax.set_title("Jumlah Review per Sentimen")
        st.pyplot(fig)

    # Pie
    with c2:
        fig, ax = plt.subplots()
        ax.pie(
            counts.loc[["Positif","Negatif","Netral"]],
            labels=["positive","negative","neutral"],
            autopct="%1.1f%%",
            colors=["#22c55e","#ef4444","#facc15"],
            startangle=90
        )
        ax.set_title("Persentase Sentimen")
        st.pyplot(fig)

    # Rating
    with c3:
        rating_col = detect_column(df, ["rating","score","bintang"])
        if rating_col is None:
            st.info("Kolom rating tidak tersedia")
        else:
            fig, ax = plt.subplots()
            grp = df.groupby([rating_col,label_col]).size().unstack(fill_value=0)
            grp.plot(kind="bar", ax=ax)
            ax.set_title("Distribusi Rating per Sentimen")
            st.pyplot(fig)

# ================== MODELING ==================
elif menu == "🧠 Modeling & Evaluasi":
    if "df" not in st.session_state:
        st.warning("Upload dataset dulu")
    else:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(
            df[st.session_state.text_col],
            df[st.session_state.label_col],
            test_size=0.2, random_state=42, stratify=df[st.session_state.label_col]
        )
        tfidf = TfidfVectorizer(max_features=15000)
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success(f"Akurasi: {accuracy_score(y_test,y_pred):.4f}")
        st.text(classification_report(y_test,y_pred))

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download":
    if "df" not in st.session_state:
        st.warning("Tidak ada data")
    else:
        st.download_button(
            "Download CSV",
            st.session_state.df.to_csv(index=False).encode("utf-8"),
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )
