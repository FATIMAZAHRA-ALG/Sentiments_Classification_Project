
# Prédiction de Sentiments Multilingue (FR/EN)

Cette application web permet à un utilisateur de saisir un texte libre et d’obtenir en temps réel la **prédiction du sentiment (positif ou négatif)** associé à ce texte. Elle repose sur un modèle de classification fine-tuné sur des données multilingues (`IMDb` en anglais et `Allociné` en français).

## 🎥 Présentation vidéo

Vous pouvez regarder une démonstration de ce projet en vidéo ici :

👉 [Voir la vidéo de présentation](Video/Explication.mp4)

## 🌐 Démo en ligne

Vous pouvez tester l'application ici :

👉 [Tester l'application en ligne](https://sentiments-classification-project.streamlit.app/)


## Fonctionnalités

-  **Nettoyage du texte** : suppression des balises HTML, des caractères spéciaux et des espaces superflus.
-  **Chargement optimisé du modèle** : utilisation de `@st.cache_resource` pour accélérer les appels du modèle.
-  **Modèle multilingue** : basé sur `distilbert-base-multilingual-cased` fine-tuné sur des avis en français et en anglais.
-  **Interface intuitive** : conçue avec `Streamlit` pour une expérience utilisateur simple et agréable.
-  **Affichage dynamique** :
  - Sentiment **positif** → boîte verte avec une icône souriante 😄
  - Sentiment **négatif** → boîte rouge avec une icône triste 😞
- **Score de confiance** affiché avec chaque prédiction.

##  Données utilisées

- **IMDb** : avis de films en anglais
- **Allociné** : critiques de films en français

Les deux jeux de données ont été nettoyés, fusionnés et rééchantillonnés pour équilibrer l'entraînement et la validation.

##  Entraînement

- Tokenizer : `distilbert-base-multilingual-cased`
- Entraînement avec `Trainer` de Transformers
- Accuracy évaluée via la métrique `accuracy`
- Entraînement sur 3 epochs avec `AdamW` et `weight decay`

## Modèle pré-entraîné disponible

Le modèle fine-tuné est également **disponible sur Hugging Face Hub** et peut être chargé directement depuis ce dépôt, ce qui permet d’éviter de relancer l’entraînement localement.

Pour l’utiliser, il suffit de charger le modèle avec :  
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "FATIMA-ZAHRA-Z/my_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

```
## Dépendances
 ### Pour l'entraînement du modèle
- transformers : pour le modèle BERT multilingue et le tokenizer
  
- datasets : pour le chargement et la gestion des jeux de données
  
- evaluate : pour le calcul des métriques d’évaluation
- torch : pour l'entraînement avec PyTorch
  
- numpy : pour les opérations numériques (comme l’argmax sur les prédictions).
  
- re : pour le nettoyage des textes (suppression de balises HTML, ponctuation, etc.).

### Pour l’application Streamlit
- streamlit : pour créer l’interface web

- transformers : pour charger le modèle fine-tuné

- torch : pour faire les prédictions
  
- re : pour le nettoyage des textes (suppression de balises HTML, ponctuation, etc.).
  
##  Lancement

### Entraînement du modèle
```bash
python train_model.py
```
###  Installation des dépendances
```bash
pip install -r requirements.txt
```
### Lancer l'application
```bash
streamlit run app.py
```


  
  ## Exemples de prédiction

- Texte : *"Le film était incroyable et très émouvant."* → **Positif 😄** 
- Texte : *"Je me suis ennuyé pendant toute la réunion."* → **Négatif 😞**





