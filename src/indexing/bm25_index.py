"""
Index BM25 (recherche lexicale par mots-clés) — complète la recherche
sémantique FAISS pour former une recherche hybride.

BM25 excelle sur les termes précis (sigles médicaux, noms de maladies)
que la recherche par embedding peut parfois manquer, ex: "HTA", "rougeole".

Usage :
    python -m src.indexing.bm25_index
"""
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import INDEX_DIR
from src.indexing.vector_store import load_chunks

BM25_PATH = INDEX_DIR / "bm25.pkl"

# Tokenisation simple mais adaptée aux caractères accentués du français
TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüÿçœæ0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """
    Découpe un texte en tokens (mots) minuscules, en ignorant la
    ponctuation. BM25 travaille sur des mots exacts (contrairement aux
    embeddings qui travaillent sur le "sens"), donc la qualité de cette
    tokenisation influence directement la qualité de la recherche.

    Args:
        text: texte à tokeniser.

    Returns:
        Liste de mots en minuscules (ex. "Le Diabète" -> ["le", "diabète"]).
    """
    # findall() extrait toutes les suites de caractères qui correspondent
    # au motif TOKEN_RE (lettres, accents français, chiffres), en ignorant
    # tout le reste (ponctuation, espaces...).
    return TOKEN_RE.findall(text.lower())


def build_bm25_index(chunks: list[dict] = None) -> None:
    """
    Construit l'index BM25 à partir des chunks : tokenise chaque chunk,
    puis construit une structure BM25Okapi qui permettra de calculer
    rapidement un score de pertinence lexicale pour n'importe quelle
    requête future.

    Args:
        chunks: liste de chunks à indexer. Si None, charge automatiquement
            data/chunks/chunks.jsonl (même source que l'index FAISS).
    """
    chunks = chunks or load_chunks()
    if not chunks:
        print("⚠️  Aucun chunk à indexer pour BM25. Lancez d'abord le chunker.")
        return

    # BM25Okapi a besoin du corpus déjà tokenisé (liste de listes de mots),
    # pas du texte brut.
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # On sauvegarde à la fois l'objet bm25 (la structure de recherche) ET
    # les chunks originaux dans le même fichier .pkl, car BM25Okapi ne
    # connaît que les tokens, pas le texte original ni les métadonnées
    # (source_document, etc.) — il faut donc garder le lien entre les deux.
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"✓ Index BM25 construit avec {len(chunks)} documents → {BM25_PATH}")


class BM25Store:
    """
    Wrapper autour de l'index BM25, avec la même interface que VectorStore
    (une méthode search()), pour que le retriever puisse utiliser les deux
    types de recherche de façon interchangeable.
    """

    def __init__(self, path: Path = BM25_PATH):
        """
        Charge l'index BM25 (et les chunks associés) depuis le disque.

        Raises:
            FileNotFoundError: si l'index n'a pas encore été construit.
        """
        if not path.exists():
            raise FileNotFoundError(
                "Index BM25 introuvable. Lancez d'abord : "
                "python -m src.indexing.bm25_index"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retourne les top_k chunks avec le meilleur score de pertinence
        lexicale (BM25) pour la requête donnée.

        Args:
            query: texte de la requête.
            top_k: nombre maximal de résultats à retourner.

        Returns:
            Liste de chunks enrichis d'une clé "bm25_score" (plus le score
            est élevé, plus le chunk partage de mots-clés importants avec
            la requête), triés du plus au moins pertinent.
        """
        tokenized_query = tokenize(query)
        # get_scores() calcule un score BM25 pour CHAQUE chunk de l'index
        # d'un coup (pas de recherche "top-k" native contrairement à FAISS).
        scores = self.bm25.get_scores(tokenized_query)

        # On trie nous-mêmes les indices des chunks par score décroissant,
        # et on ne garde que les top_k premiers.
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
        return results


if __name__ == "__main__":
    build_bm25_index()
