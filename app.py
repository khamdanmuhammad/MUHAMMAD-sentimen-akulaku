import streamlit as st
import joblib
import pandas as pd

# =====================
# LOAD MODEL
# =====================
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model_nb.pkl")

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.write("Klasifikasi sentimen: **Positif – Netral – Negatif**")

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
            vec = tfidf.transform([text])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            if pred == "positif":
                st.success("😊 Sentimen: POSITIF")
            elif pred == "netral":
                st.info("😐 Sentimen: NETRAL")
            else:
                st.error("😡 Sentimen: NEGATIF")

            df_prob = pd.DataFrame({
                "Sentimen": model.classes_,
                "Probabilitas": prob
            })

            st.bar_chart(df_prob.set_index("Sentimen"))

# ======================================================
# TAB 2 : UPLOAD CSV
# ======================================================
with tab2:
    st.write("Upload file CSV berisi ulasan pengguna")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview Data:")
        st.dataframe(df.head())

        # Pilih kolom teks
        col_text = st.selectbox(
            "Pilih kolom yang berisi teks ulasan:",
            df.columns
        )

        if st.button("Analisis Sentimen CSV"):
            teks = df[col_text].astype(str)

            vec = tfidf.transform(teks)
            preds = model.predict(vec)

            # Tambahkan hasil sentimen
            df["sentimen"] = preds

            # Mapping label (BIAR PASTI SESUAI)
            df["sentimen"] = df["sentimen"].map({
                "positif": "Positif",
                "netral": "Netral",
                "negatif": "Negatif"
            })

            st.success("✅ Analisis selesai!")
            st.dataframe(df)

            # Rekap jumlah sentimen
            st.subheader("📊 Distribusi Sentimen")
            sent_count = df["sentimen"].value_counts()
            st.bar_chart(sent_count)

            # Download hasil
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil CSV",
                csv_out,
                "hasil_analisis_sentimen.csv",
                "text/csv"
            )
