"""
Construction et interrogation d'un index vectoriel FAISS
à partir des chunks de documents médicaux marocains.

Usage :
    python -m src.indexing.vector_store   # construit l'index
"""
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CHUNKS_DIR, INDEX_DIR
from src.indexing.embedder import embed_batch, embed_text

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"


def load_chunks(chunks_path: Path = CHUNKS_DIR / "chunks.jsonl") -> list[dict]:
    """
    Charge les chunks depuis le fichier JSONL produit par chunker.py.

    Args:
        chunks_path: chemin vers le fichier chunks.jsonl.

    Returns:
        Liste de dictionnaires, un par chunk (avec text, source_document, etc.).
    """
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        # Un objet JSON par ligne : on parse chaque ligne séparément
        # plutôt que de charger tout le fichier comme un seul gros JSON.
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index(chunks: list[dict] = None) -> None:
    """
    Construit l'index vectoriel FAISS à partir des chunks : génère
    l'embedding de chaque chunk, puis les indexe pour permettre une
    recherche par similarité rapide. Sauvegarde l'index et les métadonnées
    (le texte original de chaque chunk) sur disque.

    Args:
        chunks: liste de chunks à indexer. Si None, charge automatiquement
            data/chunks/chunks.jsonl via load_chunks().
    """
    chunks = chunks or load_chunks()
    if not chunks:
        print("⚠️  Aucun chunk à indexer. Lancez d'abord src/preprocessing/chunker.py.")
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_batch(texts)

    # dimension = taille de chaque vecteur d'embedding (ex. 1536 pour
    # text-embedding-3-small). On la déduit automatiquement du résultat
    # plutôt que de la coder en dur.
    dimension = embeddings.shape[1]

    # IndexFlatIP = index FAISS basé sur le produit scalaire (Inner Product).
    # Combiné à des vecteurs normalisés (voir normalize_L2 ci-dessous), le
    # produit scalaire devient mathématiquement équivalent à la similarité
    # cosinus, la métrique standard pour comparer des embeddings de texte.
    index = faiss.IndexFlatIP(dimension)

    # normalize_L2 met chaque vecteur à une longueur de 1 (norme unitaire).
    # Sans cette étape, le produit scalaire serait influencé par la
    # longueur des vecteurs et pas seulement par leur "direction" (le sens).
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Sauvegarde de l'index (les vecteurs) et des métadonnées (le texte et
    # les infos de chaque chunk) séparément : FAISS ne sait sauvegarder que
    # les vecteurs, pas les données associées, d'où le fichier .pkl à part.
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✓ Index FAISS construit avec {index.ntotal} vecteurs → {INDEX_PATH}")


class VectorStore:
    """
    Wrapper autour de l'index FAISS et de ses métadonnées, pour offrir une
    interface simple de recherche par similarité sémantique : on lui donne
    une question en texte, il retourne les chunks les plus proches en sens.
    """

    def __init__(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH):
        """
        Charge l'index FAISS et les métadonnées associées depuis le disque.

        Raises:
            FileNotFoundError: si l'index n'a pas encore été construit
                (l'utilisateur doit d'abord lancer build_index()).
        """
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Index introuvable. Lancez d'abord build_index() "
                "(python -m src.indexing.vector_store)."
            )
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retourne les top_k chunks les plus proches sémantiquement de la
        requête, triés du plus pertinent au moins pertinent.

        Args:
            query: texte de la requête (ex. la question de l'utilisateur).
            top_k: nombre maximal de résultats à retourner.

        Returns:
            Liste de chunks (dictionnaires), chacun enrichi d'une clé
            "similarity_score" (float entre -1 et 1, proche de 1 = très
            similaire) grâce à la similarité cosinus.
        """
        # On vectorise la requête avec le MÊME modèle d'embedding que celui
        # utilisé pour indexer les chunks — indispensable pour que les
        # vecteurs soient comparables entre eux.
        query_embedding = np.array([embed_text(query)], dtype="float32")
        faiss.normalize_L2(query_embedding)

        # index.search() retourne deux tableaux parallèles : les scores de
        # similarité et les indices (positions) des chunks correspondants
        # dans self.metadata.
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        # scores[0] et indices[0] car search() est conçu pour traiter
        # plusieurs requêtes à la fois ; ici on n'en a qu'une, donc on
        # prend la première (et unique) ligne de résultats.
        for score, idx in zip(scores[0], indices[0]):
            # FAISS retourne -1 quand il y a moins de résultats disponibles
            # que top_k demandé (ex. si l'index contient moins de chunks).
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["similarity_score"] = float(score)
            results.append(chunk)
        return results


if __name__ == "__main__":
    build_index()
