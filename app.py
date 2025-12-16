import streamlit as st
import joblib
import pandas as pd

tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model_nb.pkl")

st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.write("Klasifikasi sentimen: **Positif – Netral – Negatif**")

text = st.text_area(
    "Masukkan ulasan pengguna:",
    height=150,
    placeholder="Contoh: aplikasinya cukup membantu tapi kadang error"
)

if st.button("Analisis Sentimen"):
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
