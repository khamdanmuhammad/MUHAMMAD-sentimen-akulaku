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

# ================== STYLE (BIRU & KUNING, TEXT JELAS) ==================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 40%, #facc15 100%);
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-radius: 18px;
    margin: 10px;
    padding: 10px;
}

/* ===== SIDEBAR TEXT (INI YANG FIX MASALAH ANDA) ===== */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #000000 !important;
    font-weight: 600;
}

/* ===== TITLE ===== */
h1 {
    color: #ffffff;
    font-weight: 800;
}

/* ===== CARD ===== */
div[data-testid="stVerticalBlock"],
div[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,.2);
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #facc15);
    color: #000000;
    font-weight: 700;
    border-radius: 12px;
    border: none;
}

/* ===== INPUT ===== */
textarea, input {
    border-radius: 10px !important;
    border: 1px solid #facc15 !important;
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
    return " ".join([w for w in text.split() if w not in stop_words])

# ================== RULE BASED ==================
def rule_based_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["error","gagal","parah","kecewa","ribet"]):
        return "Negatif"
    if any(w in text for w in ["bagus","mantap","puas","mudah","cepat"]):
        return "Positif"
    return "Netral"

# ================== UTIL ==================
def detect_column(df, keys):
    for col in df.columns:
        for k in keys:
            if k in col.lower():
                return col
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
    [
        "📂 Upload Dataset",
        "📊 Dashboard",
        "🧠 Modeling & Evaluasi"
    ]
)

# ================== UPLOAD ==================
if menu == "📂 Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_csv_safe(file)
        if df is None:
            st.error("Gagal membaca CSV")
            st.stop()

        text_col = detect_column(df, ["review","ulasan","content","text"]) or df.columns[0]

        df["sentiment"] = df[text_col].astype(str).apply(rule_based_sentiment)
        df[text_col] = df[text_col].apply(clean_text)

        st.session_state.df = df
        st.session_state.text_col = text_col

        st.success("Dataset berhasil dimuat")

# ================== DASHBOARD (GRAFIK TIDAK HILANG) ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df

    counts = df["sentiment"].value_counts()
    total = len(df)

    st.subheader("🎯 DISTRIBUSI SENTIMEN")
    st.text(
        f"""POSITIVE : {counts.get("Positif",0):,} ({counts.get("Positif",0)/total*100:.1f}%)
NEUTRAL  : {counts.get("Netral",0):,} ({counts.get("Netral",0)/total*100:.1f}%)
NEGATIVE : {counts.get("Negatif",0):,} ({counts.get("Negatif",0)/total*100:.1f}%)
"""
    )

    col1, col2, col3 = st.columns(3)

    # BAR
    with col1:
        fig, ax = plt.subplots()
        counts.plot(kind="bar", color=["green","gold","red"], ax=ax)
        ax.set_title("Jumlah Review per Sentimen")
        st.pyplot(fig)

    # PIE
    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            counts,
            labels=counts.index.str.lower(),
            autopct="%1.1f%%",
            colors=["green","gold","red"],
            startangle=90
        )
        ax.set_title("Persentase Sentimen")
        st.pyplot(fig)

    # RATING (TETAP ADA, TIDAK DIHAPUS)
    with col3:
        if "rating" in df.columns:
            grp = df.groupby(["rating","sentiment"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots()
            grp.plot(kind="bar", ax=ax)
            ax.set_title("Distribusi Rating per Sentimen")
            st.pyplot(fig)
        else:
            st.info("Kolom rating tidak tersedia")

# ================== MODELING ==================
elif menu == "🧠 Modeling & Evaluasi":
    if "df" not in st.session_state:
        st.warning("Upload dataset dulu")
        st.stop()

    df = st.session_state.df

    X_train, X_test, y_train, y_test = train_test_split(
        df[st.session_state.text_col],
        df["sentiment"],
        test_size=0.2,
        random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    st.success(f"Akurasi: {accuracy_score(y_test,y_pred):.4f}")
    st.text(classification_report(y_test,y_pred))
