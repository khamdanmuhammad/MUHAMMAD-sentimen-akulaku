import streamlit as st
import joblib
import pandas as pd
import re

# =====================
# LOAD MODEL
# =====================
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model_nb.pkl")

# =====================
# PREPROCESS
# =====================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

# =====================
# TAB MENU
# =====================
tab1, tab2 = st.tabs(["🔍 Input Manual", "📂 Upload CSV"])

# ======================================================
# TAB 1 : INPUT MANUAL
# ======================================================
with tab1:
    text = st.text_area(
        "Masukkan ulasan pengguna:",
        height=150,
        placeholder="Contoh: aplikasinya cukup membantu tapi kadang error"
    )

    if st.button("Analisis Sentimen (Manual)"):
        if text.strip() == "":
            st.warning("Masukkan teks terlebih dahulu!")
        else:
            vec = tfidf.transform([clean_text(text)])
            pred = model.predict(vec)[0]

            if pred == "positif":
                st.success("😊 Sentimen: positif")
            elif pred == "netral":
                st.info("😐 Sentimen: netral")
            else:
                st.error("😡 Sentimen: negatif")

# ======================================================
# TAB 2 : UPLOAD CSV (FIX)
# ======================================================
with tab2:
    st.write("Upload file CSV berisi ulasan pengguna")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success(f"📄 Data berhasil dimuat: {len(df)} baris")

        st.write("🔍 Preview data (50 baris pertama):")
        st.dataframe(df.head(50), use_container_width=True)

        # ✅ KURUNG SUDAH BENAR
        col_text = st.selectbox(
            "Pilih kolom yang berisi teks ulasan:",
            options=df.columns
        )

        if st.button("Analisis Sentimen CSV"):
            teks = df[col_text].fillna("").apply(clean_text)

            vec = tfidf.transform(teks)
            preds = model.predict(vec)

            df["sentimen"] = preds

            st.success("✅ Analisis sentimen selesai")

            st.subheader("📊 Ringkasan Sentimen")
            sent_count = df["sentimen"].value_counts()
            st.table(sent_count)
            st.bar_chart(sent_count)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil Lengkap (CSV)",
                data=csv_out,
                file_name="hasil_analisis_sentimen.csv",
                mime="text/csv"
            )
