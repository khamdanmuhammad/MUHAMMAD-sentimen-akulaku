import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np

# =====================
# LOAD MODEL
# =====================
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model_nb.pkl")

# =====================
# PREPROCESS
# =====================
def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except:
        return ""

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.write("Klasifikasi sentimen: **positif – netral – negatif**")

tab1, tab2 = st.tabs(["🔍 Input Manual", "📂 Upload CSV"])

# ======================================================
# TAB 1 : MANUAL
# ======================================================
with tab1:
    text = st.text_area("Masukkan ulasan pengguna:")

    if st.button("Analisis Sentimen (Manual)"):
        try:
            text_clean = clean_text(text)
            vec = tfidf.transform([text_clean])
            pred = model.predict(vec)[0]

            if pred == "positif":
                st.success("😊 Sentimen: positif")
            elif pred == "netral":
                st.info("😐 Sentimen: netral")
            else:
                st.error("😡 Sentimen: negatif")
        except Exception as e:
            st.error("Terjadi kesalahan pada input teks")
            st.code(str(e))

# ======================================================
# TAB 2 : CSV (SUPER AMAN)
# ======================================================
with tab2:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8",
                errors="ignore"
            )

            st.success(f"📄 Data dimuat: {len(df)} baris")
            st.dataframe(df.head(50))

            col_text = st.selectbox(
                "Pilih kolom teks ulasan",
                df.columns
