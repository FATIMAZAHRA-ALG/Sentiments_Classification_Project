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
    model_dir = "my_imdb_model"
    zip_path = "model.zip"
    google_drive_id = "1_yTVgWec_BrzOOFPMunKgqL8Q_Ap4CHc"  # <-- ton nouveau ID

    if not os.path.exists(model_dir):
        url = f"https://drive.google.com/uc?id={google_drive_id}"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(zip_path)
        print("Contenu du dossier après extraction :", os.listdir(model_dir))
        for root, dirs, files in os.walk(model_dir):
          print(f"Dans {root} -> Dossiers: {dirs}, Fichiers: {files}")

    

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

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
}
.stTextArea>div>textarea {
    font-size: 18px;
    padding: 12px;
    border-radius: 8px;
    border: 2px solid #6A5ACD;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Analyse de sentiments ")

text = st.text_area("Entrer un texte :", height=300)

if st.button("Prédire le sentiment") and text:
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    labels_map = {0: "Négatif 😞", 1: "Positif 😄"}

    if predicted_class == 1:
        st.success(f"**Sentiment prédit :** {labels_map[predicted_class]}")
        st.info(f"**Confiance :** {confidence:.2f}")
    else:
        st.error(f"**Sentiment prédit :** {labels_map[predicted_class]}")
        st.warning(f"**Confiance :** {confidence:.2f}")
