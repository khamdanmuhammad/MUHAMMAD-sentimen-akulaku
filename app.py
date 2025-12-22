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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
.stTextArea textarea {border-radius: 10px;}
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
    st.subheader("✍️ Prediksi Sentimen")
    text = st.text_area("Masukkan ulasan", height=120)

    if st.button("🔍 Analisis"):
        if not text.strip():
            st.warning("⚠️ Teks kosong")
        else:
            rule = rule_based_sentiment(text)
            if rule:
                st.success(f"Hasil: {rule} (Rule-Based)")
            else:
                if "df" not in st.session_state:
                    st.warning("⚠️ Upload dataset dulu")
                else:
                    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
                    X = tfidf.fit_transform(st.session_state.df[st.session_state.text_col])
                    y = st.session_state.df[st.session_state.label_col]
                    model = MultinomialNB(alpha=0.5)
                    model.fit(X, y)
                    pred = model.predict(tfidf.transform([clean_text(text)]))[0]
                    st.success(f"Hasil: {pred} (Machine Learning)")

# ================== DASHBOARD ==================
elif menu == "📊 Dashboard":
    if "df" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
    else:
        st.subheader("=== LABELING SENTIMEN ===")
        df = st.session_state.df
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Distribusi Sentimen")
        fig, ax = plt.subplots()
        df[st.session_state.label_col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ================== MODELING & EVALUASI ==================
elif menu == "🧠 Modeling & Evaluasi":
    st.subheader("=== MODELING DAN EVALUASI ===")

    if "df" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df
    text_col = st.session_state.text_col
    label_col = st.session_state.label_col

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col],
        test_size=0.3,
        random_state=42,
        stratify=df[label_col]
    )

    st.text(f"""📊 DISTRIBUSI DATA:
   Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)
   Test set    : {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)
""")

    def dist(y):
        return y.value_counts()

    tr, te = dist(y_train), dist(y_test)

    st.text(f"""📈 DISTRIBUSI SENTIMEN DI SETIAP SPLIT:

   Training:
      negative: {tr.get("Negatif",0):,}
      neutral : {tr.get("Netral",0):,}
      positive: {tr.get("Positif",0):,}

   Test:
      negative: {te.get("Negatif",0):,}
      neutral : {te.get("Netral",0):,}
      positive: {te.get("Positif",0):,}
""")

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    best_model = None
    best_f1 = 0
    best_pred = None

    for name, model in models.items():
        model.fit(X_train_vec, y_train)

        train_pred = model.predict(X_train_vec)
        test_pred = model.predict(X_test_vec)

        acc_tr = accuracy_score(y_train, train_pred)
        acc_te = accuracy_score(y_test, test_pred)
        f1_tr = f1_score(y_train, train_pred, average="weighted")
        f1_te = f1_score(y_test, test_pred, average="weighted")

        cv = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="f1_weighted")

        st.text(f"""
============================================================
🎯 MODEL: {name}
============================================================

   📊 PERFORMANCE:
      Akurasi Training : {acc_tr:.4f}
      Akurasi Test     : {acc_te:.4f}
      F1-Score Training: {f1_tr:.4f}
      F1-Score Test    : {f1_te:.4f}

   🔄 CROSS-VALIDATION (5-fold):
      CV Scores: {cv}
      CV Mean  : {cv.mean():.4f}
      CV Std   : {cv.std():.4f}

   📋 CLASSIFICATION REPORT (Test Set):
{classification_report(y_test, test_pred)}
""")

        results.append([name, acc_te, f1_te, cv.mean(), cv.std()])

        if f1_te > best_f1:
            best_f1 = f1_te
            best_model = name
            best_pred = test_pred

    result_df = pd.DataFrame(
        results,
        columns=["Model", "Test Accuracy", "Test F1-Score", "CV Mean", "CV Std"]
    )

    st.subheader("📈 PERBANDINGAN SEMUA MODEL")
    st.dataframe(result_df, use_container_width=True)

    st.success(f"🏆 MODEL TERBAIK: {best_model}")

    mis = (y_test != best_pred).sum()
    st.text(f"""🔍 ANALISIS ERROR - {best_model}
   Total misclassified: {mis} ({mis/len(y_test)*100:.1f}%)
""")

    if mis == 0:
        st.success("✅ Tidak ada data salah klasifikasi. Tanpa error.")

# ================== DOWNLOAD ==================
elif menu == "⬇️ Download":
    if "df" not in st.session_state:
        st.warning("⚠️ Tidak ada data untuk di-download")
    else:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV Hasil Sentimen",
            csv,
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )
