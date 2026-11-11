"""
Génération des embeddings pour les chunks de texte, via l'API
compatible OpenAI (GitHub Models par défaut).

Qu'est-ce qu'un embedding ?
Un embedding est une liste de nombres (un vecteur, ex. 1536 dimensions)
qui représente le "sens" d'un texte. Deux textes au sens proche auront
des vecteurs proches mathématiquement, ce qui permet de faire une
recherche par similarité (voir vector_store.py) au lieu d'une simple
recherche par mots-clés exacts.
"""
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import API_BASE_URL, API_KEY, EMBEDDING_MODEL

_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def embed_text(text: str) -> list[float]:
    """
    Génère l'embedding d'un seul texte (utilisé pour vectoriser une
    question de l'utilisateur au moment de la recherche).

    Args:
        text: texte à transformer en vecteur.

    Returns:
        Liste de nombres flottants représentant le texte.
    """
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_batch(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """
    Génère les embeddings pour une liste de textes, en les envoyant par
    petits lots (batches) plutôt qu'un par un ou tous en une seule requête.

    Pourquoi par lots :
    - Un par un serait trop lent (autant de requêtes réseau que de chunks).
    - Tous en une fois pourrait dépasser les limites de débit (rate limit)
      de l'API gratuite GitHub Models.
    batch_size=16 est un compromis raisonnable entre rapidité et sécurité.

    Args:
        texts: liste de textes à vectoriser (ex. le texte de chaque chunk).
        batch_size: nombre de textes envoyés par requête à l'API.

    Returns:
        Un tableau numpy de forme (nombre_de_textes, dimension_embedding),
        prêt à être indexé par FAISS.
    """
    all_embeddings = []
    # range(0, len(texts), batch_size) avance par pas de `batch_size`,
    # ex. pour 50 textes et batch_size=16 : 0, 16, 32, 48.
    for i in tqdm(range(0, len(texts), batch_size), desc="Génération des embeddings"):
        batch = texts[i:i + batch_size]
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
    # dtype="float32" : FAISS attend des vecteurs en float32 (pas float64),
    # c'est le format standard pour les bibliothèques de calcul vectoriel.
    return np.array(all_embeddings, dtype="float32")
