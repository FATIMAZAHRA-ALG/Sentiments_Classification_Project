
# Cette application web permet à un utilisateur de saisir un texte libre et d’obtenir en temps réel la prédiction du sentiment (positif ou négatif) associée à ce texte. Elle utilise un modèle de classification fine-tuné sur des données multilingues (français et anglais).

# Fonctionnalités principales

# - **Nettoyage du texte** : avant la prédiction, le texte est nettoyé pour retirer balises HTML, caractères spéciaux et espaces superflus, puis converti en minuscules pour standardiser l’entrée.
# - **Chargement optimisé du modèle** : le modèle et le tokenizer sont chargés une seule fois grâce à la mise en cache (`@st.cache_resource`), ce qui accélère les appels suivants.
# - **Interface intuitive** : un champ de saisie large permet d’entrer un texte, et un bouton déclenche la prédiction.
# - **Affichage coloré du résultat** :
#   - Le sentiment positif est affiché dans une boîte verte avec une icône souriante.
#   - Le sentiment négatif s’affiche dans une boîte rouge avec une icône triste.
# - **Confiance associée** : la probabilité (score de confiance) de la prédiction est également affichée pour informer l’utilisateur de la certitude du modèle.

# Processus utilisateur

# 1. L’utilisateur saisit un texte dans la zone prévue.
# 2. En cliquant sur “Prédire le sentiment”, le texte est nettoyé puis tokenisé.
# 3. Le modèle calcule la prédiction et renvoie la classe (positif/négatif) et la confiance.
# 4. Le résultat est affiché dans une zone colorée correspondant au sentiment détecté.

import streamlit as st
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

# 🎨 Style CSS personnalisé
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

.stTextArea>div>textarea {
    font-size: 18px;
    padding: 12px;
    border-radius: 8px;
    border: 2px solid #6A5ACD;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.result-box {
    padding: 15px;
    margin-top: 20px;
    border-radius: 8px;
    font-size: 18px;
    color: white;
    font-weight: bold;
}
.positive {
    background-color: #2E8B57;  /* Vert */
}
.negative {
    background-color: #B22222;  /* Rouge */
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Analyse de sentiments")

st.markdown("<h3 style='font-size: 24px; color: #4B0082;'>Entrer un texte :</h3>", unsafe_allow_html=True)
text = st.text_area("", height=150)

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
        st.markdown(
            f"<div class='result-box positive'> Sentiment prédit : {labels_map[predicted_class]}<br>Confiance : {confidence:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box negative'> Sentiment prédit : {labels_map[predicted_class]}<br>Confiance : {confidence:.2f}</div>",
            unsafe_allow_html=True
        )
