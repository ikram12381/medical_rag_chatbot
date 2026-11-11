"""
Pipeline RAG complet : orchestre traduction (bonus), recherche hybride
(sémantique + lexicale), garde-fou de pertinence, et génération de la réponse.

Architecture :
Utilisateur → (Optionnel) Traduction Darija→Français → Recherche hybride
(FAISS + BM25) → Garde-fou de pertinence → LLM + Contexte (citations)
→ Réponse générée + disclaimer → (Optionnel) Génération en Darija
"""
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TOP_K, LOW_RELEVANCE_THRESHOLD, LOW_RELEVANCE_MESSAGE, DISCLAIMER
from src.rag.retriever import retrieve, format_context, max_semantic_score
from src.rag.generator import generate_answer
from src.translation.darija_translator import detect_darija, translate_darija_to_french


@dataclass
class RAGResult:
    """
    Résultat complet d'une requête RAG, regroupant tout ce dont on a besoin
    pour l'afficher dans l'interface ou l'analyser lors de l'évaluation.

    Utiliser une dataclass plutôt qu'un simple dictionnaire apporte deux
    avantages : l'auto-complétion dans l'éditeur (result.answer plutôt que
    result["answer"]), et une structure explicite et documentée des champs
    attendus.

    Attributs:
        original_question: la question telle que tapée par l'utilisateur
            (peut être en Darija).
        question_fr: la question traduite en français (identique à
            original_question si elle était déjà en français).
        answer: la réponse finale générée (avec citations + disclaimer).
        sources: liste des chunks utilisés pour construire la réponse.
        input_was_darija: True si la question a été détectée comme étant
            en Darija.
        was_rejected_low_relevance: True si le garde-fou de pertinence a
            bloqué la génération (question hors périmètre des documents).
    """
    original_question: str
    question_fr: str
    answer: str
    # field(default_factory=list) plutôt que sources=[] : évite le piège
    # classique de Python où une valeur par défaut mutable (une liste) est
    # partagée entre toutes les instances de la classe si mal déclarée.
    sources: list[dict] = field(default_factory=list)
    input_was_darija: bool = False
    was_rejected_low_relevance: bool = False


def ask(question: str, respond_in_darija: bool = False, top_k: int = TOP_K,
        use_hybrid: bool = True) -> RAGResult:
    """
    Exécute le pipeline RAG complet pour une question utilisateur :
    traduction (si besoin) -> recherche hybride -> garde-fou de pertinence
    -> génération de la réponse avec citations.

    Args:
        question: question posée par l'utilisateur (français ou darija).
        respond_in_darija: si True, force la réponse en darija même si la
            question était posée en français.
        top_k: nombre de chunks à récupérer pour construire le contexte.
        use_hybrid: si True, combine recherche sémantique et lexicale (BM25).

    Returns:
        Un RAGResult contenant la réponse et toutes les informations
        associées (sources utilisées, langue détectée, etc.).
    """
    # --- Étape 1 : traduction Darija -> Français (si nécessaire) ---
    # La recherche vectorielle et les documents indexés étant en français,
    # on traduit d'abord la question pour que la recherche fonctionne
    # correctement (voir src/translation/darija_translator.py).
    is_darija = detect_darija(question)
    question_fr = translate_darija_to_french(question) if is_darija else question

    # --- Étape 2 : recherche hybride des chunks pertinents ---
    chunks = retrieve(question_fr, top_k=top_k, use_hybrid=use_hybrid)

    # --- Étape 3 : garde-fou de pertinence ---
    # Si même le meilleur chunk récupéré a un score sémantique trop faible,
    # la question sort probablement du périmètre couvert par nos documents :
    # on répond directement, sans appeler le LLM. Cela évite deux problèmes :
    # 1) le risque d'hallucination si le LLM tente de répondre quand même,
    # 2) un appel API inutile (donc plus rapide et moins coûteux en quota).
    if max_semantic_score(chunks) < LOW_RELEVANCE_THRESHOLD:
        return RAGResult(
            original_question=question,
            question_fr=question_fr,
            answer=LOW_RELEVANCE_MESSAGE + DISCLAIMER,
            sources=chunks,
            input_was_darija=is_darija,
            was_rejected_low_relevance=True,
        )

    # --- Étape 4 : génération de la réponse avec le LLM ---
    context = format_context(chunks)
    # La réponse est en Darija si la question l'était (cohérence naturelle),
    # ou si l'utilisateur a explicitement coché "Forcer la réponse en Darija".
    language = "darija" if (is_darija or respond_in_darija) else "fr"
    answer = generate_answer(question_fr, context, language=language)

    return RAGResult(
        original_question=question,
        question_fr=question_fr,
        answer=answer,
        sources=chunks,
        input_was_darija=is_darija,
    )


def ask_without_rag(question: str) -> str:
    """
    Génère une réponse SANS RAG (LLM seul, sans contexte documentaire).

    Utilisé uniquement à des fins de comparaison en évaluation (voir
    src/evaluation/evaluator.py, fonction compare_rag_vs_no_rag), pour
    démontrer concrètement l'apport du RAG : sans contexte, le LLM devrait
    logiquement refuser de répondre ou halluciner, alors qu'avec RAG il
    répond correctement et cite ses sources.

    Args:
        question: question de l'utilisateur (en français).

    Returns:
        La réponse générée sans aucun contexte documentaire.
    """
    return generate_answer(question, context="(aucun contexte fourni)", language="fr")


if __name__ == "__main__":
    # Test manuel rapide du pipeline complet, utile pour vérifier que tout
    # fonctionne après une modification, sans passer par l'interface Streamlit.
    result = ask("Quels sont les symptômes du diabète type 2 ?")
    print("Réponse :", result.answer)
    print("\nSources utilisées :")
    for s in result.sources:
        # On affiche le score sémantique si disponible, sinon le score
        # de fusion RRF (cas où le chunk vient uniquement de BM25).
        score = s.get("similarity_score", s.get("fused_score", 0.0))
        print(f"  - {s['source_document']} (score={score:.3f})")
