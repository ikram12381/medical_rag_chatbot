"""
Script principal : exécute tout le pipeline de bout en bout.

Étapes :
1. Scraping des pages web
2. Téléchargement des PDF officiels
3. Extraction du texte des PDF
4. Nettoyage
5. Chunking
6. Construction de l'index vectoriel (FAISS)
7. Construction de l'index lexical (BM25)
8. Test rapide du pipeline RAG

Usage :
    python main.py
"""
from src.scraping.web_scraper import scrape_all, download_all_pdfs
from src.scraping.pdf_extractor import process_all_pdfs
from src.preprocessing.cleaner import process_all as clean_all
from src.preprocessing.chunker import process_all as chunk_all
from src.indexing.vector_store import build_index
from src.indexing.bm25_index import build_bm25_index
from src.rag.pipeline import ask


def run_full_pipeline():
    """
    Exécute séquentiellement les 8 étapes du pipeline RAG, de la collecte
    des données brutes jusqu'à un test de génération de réponse.

    Chaque étape dépend du résultat de la précédente (ex: le chunking a
    besoin des textes déjà nettoyés), c'est pourquoi l'ordre est important.
    """
    # --- Étape 1 : récupère le HTML des pages du Ministère de la Santé ---
    print("=" * 60)
    print("ÉTAPE 1 — Scraping des pages web")
    print("=" * 60)
    scrape_all()

    # --- Étape 2 : télécharge les PDF officiels (Ministère + OMS) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Téléchargement des PDF officiels")
    print("=" * 60)
    download_all_pdfs()

    # --- Étape 3 : convertit chaque PDF téléchargé en texte brut ---
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Extraction du texte des PDF")
    print("=" * 60)
    process_all_pdfs()

    # --- Étape 4 : supprime le bruit (menus, en-têtes, mentions légales...) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Nettoyage des textes")
    print("=" * 60)
    clean_all()

    # --- Étape 5 : découpe les textes nettoyés en segments indexables ---
    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Chunking")
    print("=" * 60)
    chunk_all()

    # --- Étape 6 : transforme les chunks en vecteurs et construit FAISS ---
    print("\n" + "=" * 60)
    print("ÉTAPE 6 — Construction de l'index vectoriel (FAISS)")
    print("=" * 60)
    build_index()

    # --- Étape 7 : construit l'index de recherche par mots-clés (BM25) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 7 — Construction de l'index lexical (BM25)")
    print("=" * 60)
    build_bm25_index()

    # --- Étape 8 : vérifie que le pipeline complet fonctionne de bout en bout ---
    print("\n" + "=" * 60)
    print("ÉTAPE 8 — Test rapide du pipeline RAG")
    print("=" * 60)
    result = ask("Quels sont les symptômes du diabète type 2 ?")
    print("\nQuestion :", result.original_question)
    print("Réponse :", result.answer)


if __name__ == "__main__":
    # Ce bloc ne s'exécute que si le fichier est lancé directement
    # (python main.py), pas s'il est importé depuis un autre module.
    run_full_pipeline()
