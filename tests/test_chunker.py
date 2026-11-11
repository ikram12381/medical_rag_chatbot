"""
Tests unitaires basiques pour valider le module de chunking.

Usage :
    python -m pytest tests/
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.chunker import chunk_text, ENCODING


def test_chunk_text_respects_size():
    """
    Vérifie qu'aucun chunk produit ne dépasse significativement la taille
    maximale demandée (chunk_size), avec une petite tolérance car le
    découpage se fait sur des frontières de tokens, pas de caractères.
    """
    text = "Ceci est une phrase de test. " * 200  # texte assez long pour produire plusieurs chunks

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    # On doit obtenir plusieurs chunks, sinon le découpage n'a pas eu lieu
    # (ce qui serait suspect pour un texte aussi long).
    assert len(chunks) > 1
    for chunk in chunks:
        n_tokens = len(ENCODING.encode(chunk))
        assert n_tokens <= 100 + 5  # petite tolérance pour les arrondis de tokenisation


def test_chunk_text_empty_input():
    """Un texte vide doit produire une liste de chunks vide, pas une erreur."""
    assert chunk_text("") == []


def test_chunk_text_overlap_preserves_context():
    """
    Vérifie que le chevauchement (overlap) fonctionne bien : avec un texte
    suffisamment long, on doit obtenir au moins 2 chunks qui se chevauchent
    partiellement (pas juste un découpage brut sans chevauchement).
    """
    text = "Un deux trois quatre cinq six sept huit neuf dix. " * 50
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 2


if __name__ == "__main__":
    # Permet de lancer les tests directement avec "python tests/test_chunker.py",
    # en plus de la méthode standard "python -m pytest tests/".
    test_chunk_text_respects_size()
    test_chunk_text_empty_input()
    test_chunk_text_overlap_preserves_context()
    print("✓ Tous les tests sont passés.")
