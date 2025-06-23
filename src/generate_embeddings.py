import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Charger le fichier Excel
df = pd.read_excel("C:/Users/ZSMJ1267/Downloads/structured_bug_data.xlsx")

# Charger le modèle
model = SentenceTransformer("C:/Users/ZSMJ1267/Desktop/all-MiniLM-L6-v2")

# Appliquer le modèle à la colonne contenant les descriptions
descriptions = df["Summary"].astype(str).tolist()

embeddings = model.encode(descriptions, show_progress_bar=True)

# Sauvegarder les embeddings
np.save("C:/Users/ZSMJ1267/Desktop/my-project/src/embedding.npy", embeddings)

print("✅ Embeddings sauvegardés avec succès.")
