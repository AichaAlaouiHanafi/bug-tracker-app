import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sentence_transformers import SentenceTransformer, util

# === Chemins ===
# === Chemins relatifs au dossier du projet ===
BASE_DIR = os.path.dirname(__file__)  # dossier où se trouve streamlit_app.py

DATA_PATH = "src/data_cleaned_fusion_concat.xlsx"
EMB_PATH = "src/embedding.npy"
MODEL_PATH = "src/all-MiniLM-L6-v2"

GEMINI_API_KEY = "AIzaSyAAv2LbIhkkk2gGVLJwTupjl5GMxHWVFNw"

# === Chargement des données ===
df = pd.read_excel(DATA_PATH)
model = SentenceTransformer(MODEL_PATH)
embeddings = np.load(EMB_PATH, allow_pickle=True)

# === Fonction : Détection du type de problème ===
def detect_type(text):
    incident_keywords = ["incident", "no data", "erreur", "crash", "failed", "ko", "indisponible", "interruption", "injoignable"]
    bug_keywords = ["bug", "anomalie", "défaut", "incorrect", "problème", "affichage", "port", "valeur erronée"]

    text = text.lower()
    if any(kw in text for kw in incident_keywords):
        return "incident"
    elif any(kw in text for kw in bug_keywords):
        return "bug"
    else:
        return "bug"

# === Fonction IA : Génération de solution avec Gemini ===
def generate_ai_solution(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    if not prompt.strip():
        return "Aucun prompt fourni pour la génération de solution."
    if not GEMINI_API_KEY:
        return "Clé API Gemini manquante."
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Propose une solution au problème suivant : {prompt}"
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Erreur API Gemini : {response.status_code} - {response.text}"
    except Exception as e:
        return f"Erreur lors de l'appel API : {e}"

# === Interface ===
st.title(" Assistant de Suivi des Bugs Internes")
desc = st.text_area(" Décris le problème rencontré ", key="bug_description")

# === Détection et affichage du type de problème ===
if desc.strip():
    detected_type = detect_type(desc)
    if detected_type == "incident":
        st.error(" Ce problème est un **incident**.")
    else:
        st.warning(" Ce problème est un **bug**.")

# === Recherche de similarité ===
if st.button(" Rechercher"):
    if desc.strip():
        query_emb = model.encode([desc])
        scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:5]
        results = df.iloc[top_idx].copy()
        results["similarité"] = scores[top_idx]

        st.session_state['last_results'] = results
        st.session_state['new_bug'] = desc
        st.session_state['detected_type'] = detected_type  # stocke le type

# === Affichage des résultats ===
if 'last_results' in st.session_state:
    st.subheader(" Résultats similaires :")
    st.dataframe(
        st.session_state['last_results'][["Summary", "similarité", "Resolution"]]
        if "Resolution" in st.session_state['last_results'].columns
        else st.session_state['last_results']
    )

    # 🔍 Résolution trouvée ?
    top_resolution = st.session_state['last_results'].iloc[0].get("Resolution", "")
    if isinstance(top_resolution, str) and top_resolution.strip():
        st.success(f" Solution proposée depuis la base : {top_resolution}")
    else:
        st.warning(" Aucune solution trouvée. Génération via l'IA...")
        ai_solution = generate_ai_solution(st.session_state['new_bug'])
        st.info(f" Suggestion IA : {ai_solution}")
        st.session_state['ai_solution'] = ai_solution

    # ➕ Ajout du bug
    with st.expander(" Ajouter ce bug comme nouveau"):
        default_solution = st.session_state.get('ai_solution', "")
        resolution = st.text_area("Ajouter/modifier une solution", value=default_solution, key="solution_input")
        if st.button("Confirmer l’ajout"):
            new_row = {
                "Summary": st.session_state['new_bug'],
                "Issue key": "NEW-" + str(len(df) + 1),
                "Issue Type": st.session_state['detected_type'],  # type ici
                "Resolution": resolution,
                "Type": st.session_state['detected_type']  # colonne 'Type' dans Excel
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            try:
                df.to_excel("data_cleaned.xlsx", index=False, engine="openpyxl")
                st.success(" Nouveau bug ajouté avec succès !")
                st.dataframe(pd.DataFrame([new_row]))
            except Exception as e:
                st.error(f" Erreur lors de la sauvegarde : {e}")

# 🔁 Réentraînement des embeddings
if st.button(" Réentraîner les embeddings"):
    with st.spinner(" Génération des nouveaux embeddings..."):
        os.system("python generate_embeddings.py")
        embeddings = np.load(EMB_PATH, allow_pickle=True)
    st.success(" Embeddings mis à jour avec succès !")

# 📊 Infos sur le modèle
st.sidebar.header("ℹInfos modèle")
st.sidebar.write("Modèle :", MODEL_PATH)
st.sidebar.write("Taille dataset :", len(df))
st.sidebar.write("Embeddings shape :", embeddings.shape)

# === Fin ===
