
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
df = pd.read_excel("C:/Users/ZSMJ1267/Downloads/structured_bug_data.xlsx")

df["text"] = df["Summary"].astype(str) + " " + df["All_Comments"].astype(str)




model = SentenceTransformer("C:/Users/ZSMJ1267/Desktop/all-MiniLM-L6-v2")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def search_similar_bugs(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = df.iloc[indices[0]][["Issue key", "Summary", "Project name", "Status", "Priority"]]
    results["Similarity"] = distances[0]
    return results


query = "Erreur de connexion à l'authentification interne"
print(search_similar_bugs(query))
