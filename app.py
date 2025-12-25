import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis Gojek",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD DATA FOR EDA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("gojek_reviews_clean.csv")

df_eda = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Pengaturan")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ("LSTM", "IndoBERT", "DistilBERT")
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_indobert():
    tokenizer = AutoTokenizer.from_pretrained("indobert_model")
    model = AutoModelForSequenceClassification.from_pretrained("indobert_model")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert_model")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert_model")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_lstm():
    model = load_model("lstm_sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# =========================
# LABEL MAP
# =========================
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# =========================
# TITLE
# =========================
st.title("📊 Sentiment Analysis Review Gojek")
st.markdown("Perbandingan **LSTM**, **IndoBERT**, dan **DistilBERT**")

# =========================
# EDA SECTION
# =========================
# =========================
# EDA SECTION
# =========================
st.header("📈 Exploratory Data Analysis (EDA)")

st.write("Kolom pada dataset:")
st.write(df_eda.columns.tolist())

# Cari kolom teks secara otomatis
text_col = None
for col in df_eda.columns:
    if df_eda[col].dtype == object:
        text_col = col
        break

if text_col is None:
    st.error("❌ Tidak ditemukan kolom teks pada dataset")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Data", df_eda.shape[0])
        st.metric("Jumlah Kolom", df_eda.shape[1])

    with col2:
        st.subheader("Distribusi Panjang Teks")
        df_eda["text_length"] = df_eda[text_col].astype(str).apply(len)
        st.bar_chart(df_eda["text_length"])

    st.subheader("Contoh Data")
    st.dataframe(df_eda[[text_col]].head())

# =========================
# MANUAL TEXT INPUT
# =========================
st.header("✍️ Uji Sentimen Manual")

user_text = st.text_area(
    "Masukkan kalimat review:",
    height=120,
    placeholder="Contoh: Driver ramah dan cepat sampai tujuan"
)

if st.button("🔍 Prediksi Sentimen"):

    if user_text.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        texts = [user_text]

        with st.spinner("Model sedang memprediksi..."):

            # ===== LSTM =====
            if model_choice == "LSTM":
                model, tokenizer = load_lstm()
                seq = tokenizer.texts_to_sequences(texts)
                padded = pad_sequences(seq, maxlen=100)
                pred = np.argmax(model.predict(padded), axis=1)[0]

            # ===== IndoBERT =====
            elif model_choice == "IndoBERT":
                tokenizer, model = load_indobert()
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()

            # ===== DistilBERT =====
            else:
                tokenizer, model = load_distilbert()
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()

        st.success(f"🧠 **Prediksi Sentimen: {label_map[pred]}**")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("UAP Pembelajaran Mesin | Demo Klasifikasi Sentimen Bahasa Indonesia")
