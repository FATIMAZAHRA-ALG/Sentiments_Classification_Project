import streamlit as st
import os
import zipfile
import gdown
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn import functional as F
import re

def clean_text(t):
    t = re.sub(r'<br\s*/?>', ' ', t)
    t = re.sub(r'[^a-zA-Z0-9,.!?\'" ]+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.lower().strip()

@st.cache_resource
def load_model():
    model_dir = "FATIMA-ZAHRA-Z/my_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# 🎨 Style harmonieux
st.markdown("""
<style>
h1 {
    color: #4B0082;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-align: center;
    margin-bottom: 30px;
}
.stButton>button {
    background-color: #6A5ACD;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #C397D8;
}
.stButton>button:active {
    background-color: #BA55D3 !important;
}
.stTextArea>div>textarea {
    font-size: 18px;
    padding: 12px;
    border-radius: 8px;
    border: 2px solid #6A5ACD;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.result-box {
    background-color: #F5F0FF;
    border-left: 6px solid #6A5ACD;
    padding: 15px;
    margin-top: 20px;
    border-radius: 8px;
    font-size: 18px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# 🔮 Titre et entrée
st.title("💬 Analyse de sentiments ")

st.markdown("<h3 style='font-size: 24px; color: #4B0082;'>Entrer un texte :</h3>", unsafe_allow_html=True)
text = st.text_area("", height=150)

# 🔎 Analyse du texte
if st.button("Prédire le sentiment") and text:
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    labels_map = {0: "Négatif 😞", 1: "Positif 😄"}

    # 🎨 Affichage personnalisé
    if predicted_class == 1:
        st.markdown(f"<div class='result-box'><strong>Sentiment prédit :</strong> {labels_map[predicted_class]}<br><strong>Confiance :</strong> {confidence:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box' style='border-left-color:#D8BFD8;'><strong>Sentiment prédit :</strong> {labels_map[predicted_class]}<br><strong>Confiance :</strong> {confidence:.2f}</div>", unsafe_allow_html=True)
