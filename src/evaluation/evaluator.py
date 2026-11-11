"""
Évaluation de la qualité du pipeline RAG, pour alimenter le
rapport d'analyse des performances demandé dans les livrables.

Métriques couvertes :
- Pertinence du retrieval (score sémantique moyen des chunks récupérés)
- Precision@k (fraction des chunks récupérés jugés réellement pertinents,
  via un LLM-as-judge)
- Fidélité de la réponse au contexte ("hallucination_probable" / "fidèle")
- Comparaison qualitative RAG vs LLM seul (sans contexte), pour démontrer
  concrètement l'apport du RAG
- Temps de réponse

Usage :
    python -m src.evaluation.evaluator
"""
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import API_BASE_URL, API_KEY, CHAT_MODEL
from src.rag.pipeline import ask, ask_without_rag
from openai import OpenAI

_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Jeu de questions de test, organisé par catégorie pour une couverture
# représentative de toutes les sources indexées et de tous les
# comportements attendus du chatbot (réponse normale, garde-fou, Darija).
TEST_QUESTIONS_BY_CATEGORY = {
    "Cardiovasculaire / Hypertension": [
        "Quels sont les facteurs de risque cardiovasculaire ?",
        "Comment prévenir les complications de l'hypertension artérielle ?",
        "Quels sont les modes de vie sains recommandés pour un hypertendu ?",
    ],
    "Diabète": [
        "Quels sont les symptômes du diabète type 2 ?",
        "Comment prendre soin de ses pieds quand on est diabétique ?",
    ],
    "Vaccination": [
        "Quel est le calendrier national de vaccination au Maroc ?",
        "À quel âge un enfant doit-il recevoir le vaccin contre la rougeole ?",
    ],
    "Épidémiologie / Pédiatrie": [
        "Comment isoler un cas de rougeole ?",
        "Quelles sont les urgences pédiatriques les plus fréquentes ?",
    ],
    "Santé publique (OMS)": [
        "Quelle est l'organisation du système de santé au Maroc ?",
        "Quels programmes existent pour la santé de l'enfant au Maroc ?",
    ],
    "Hors périmètre (test du garde-fou)": [
        "Quel est le traitement du cancer du poumon ?",
        "Quelle est la capitale du Maroc ?",
        "Quels sont les effets secondaires du Doliprane ?",
    ],
    "Darija (test du bonus linguistique)": [
        "Wach kayn vaccin dyal les enfants f Maroc ?",
        "Chno homa les symptômes dyal le diabète ?",
    ],
}

# Liste à plat, utilisée par défaut par run_evaluation()
TEST_QUESTIONS = [q for questions in TEST_QUESTIONS_BY_CATEGORY.values() for q in questions]


@dataclass
class EvalResult:
    """
    Résultat de l'évaluation d'une seule question, regroupant toutes les
    métriques mesurées pour permettre une analyse question par question
    ainsi qu'un calcul de moyennes globales (voir run_evaluation()).

    Attributs:
        question: la question testée.
        answer: la réponse générée par le chatbot.
        response_time_sec: durée totale du traitement, en secondes.
        avg_similarity_score: score sémantique moyen des chunks récupérés
            (indique si le retrieval a trouvé des documents pertinents).
        precision_at_k: fraction des chunks jugés réellement pertinents
            par le LLM-juge (entre 0.0 et 1.0).
        n_sources_used: nombre de chunks effectivement utilisés pour
            construire la réponse (peut être < top_k à cause du filtre
            MIN_CHUNK_SCORE).
        faithfulness_verdict: "fidèle", "hallucination_probable",
            "incertain", "refus_honnête", ou "rejeté_par_garde-fou".
        was_rejected_low_relevance: True si le garde-fou de pertinence a
            empêché tout appel au LLM pour cette question.
    """
    question: str
    answer: str
    response_time_sec: float
    avg_similarity_score: float
    precision_at_k: float
    n_sources_used: int
    faithfulness_verdict: str
    was_rejected_low_relevance: bool


RELEVANCE_JUDGE_PROMPT = """Question posée par l'utilisateur : {question}

Extrait de document :
{chunk_text}

Cet extrait apporte-t-il une information utile pour construire une réponse à
cette question — même partiellement, même comme contexte de fond
(ex : facteurs de risque liés, définitions, populations concernées) ?
Ne réponds "non" que si l'extrait est clairement sans rapport avec le sujet
de la question. En cas de doute raisonnable, réponds "oui".
Réponds uniquement par "oui" ou "non"."""

FAITHFULNESS_PROMPT = """Voici un contexte de référence et une réponse générée par un chatbot.
Détermine si la réponse est fidèle au contexte (n'invente pas d'information absente du contexte).

Contexte :
{context}

Réponse à évaluer :
{answer}

Réponds uniquement par un mot : "fidèle", "hallucination_probable" ou "incertain"."""

# Marqueurs indiquant que le chatbot a honnêtement décliné de répondre
# (plutôt que d'avoir tenté une réponse potentiellement fausse). Sert à éviter
# de faire juger la "fidélité au contexte" d'un refus par le LLM-juge : un
# refus n'a par définition rien à voir avec le contexte fourni, ce qui peut
# être mal interprété comme une hallucination par le juge automatique.
REFUSAL_MARKERS = [
    "je ne dispose pas",
    "je n'ai pas cette information",
    "je ne sais pas",
    "hors de mon périmètre",
    "assistant spécialisé en information médicale",
    "n'a aucun rapport avec la santé",
]


def is_honest_refusal(answer: str) -> bool:
    """
    Détecte si la réponse est un refus honnête ("je ne sais pas") plutôt
    qu'une tentative de réponse factuelle.

    Args:
        answer: réponse générée par le chatbot.

    Returns:
        True si un des marqueurs de refus est trouvé dans la réponse
        (recherche insensible à la casse).
    """
    answer_lower = answer.lower()
    # any() retourne True dès qu'AU MOINS UN marqueur est trouvé dans la
    # réponse, pas besoin qu'ils y soient tous.
    return any(marker in answer_lower for marker in REFUSAL_MARKERS)


def judge_chunk_relevance(question: str, chunk_text: str) -> bool:
    """
    Technique "LLM-as-judge" : on demande à un LLM d'évaluer lui-même si un
    chunk est pertinent pour une question, plutôt que d'utiliser une règle
    automatique rigide (ex. présence de mots-clés). Le LLM comprend mieux
    les nuances de pertinence contextuelle qu'une simple recherche de mots.

    Args:
        question: question évaluée.
        chunk_text: texte du chunk à juger.

    Returns:
        True si le juge répond "oui" (chunk jugé pertinent), False sinon.
    """
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": RELEVANCE_JUDGE_PROMPT.format(
            # On tronque le chunk à 800 caractères : suffisant pour juger
            # la pertinence, et ça limite le coût/temps de l'appel API.
            question=question, chunk_text=chunk_text[:800])}],
        temperature=0.0,  # déterministe : on veut un jugement stable, pas créatif
        max_tokens=5,
    )
    return "oui" in response.choices[0].message.content.strip().lower()


def compute_precision_at_k(question: str, chunks: list[dict]) -> float:
    """
    Calcule la precision@k : parmi les k chunks récupérés, quelle fraction
    est réellement jugée pertinente pour la question ? C'est une métrique
    standard en recherche d'information (Information Retrieval).

    Args:
        question: question évaluée.
        chunks: liste des chunks récupérés par le retriever pour cette question.

    Returns:
        Un score entre 0.0 (aucun chunk pertinent) et 1.0 (tous pertinents).
    """
    if not chunks:
        return 0.0
    # On juge chaque chunk individuellement, puis on calcule la proportion
    # de "oui" parmi tous les jugements (True vaut 1 et False vaut 0 dans
    # une somme Python, d'où sum(verdicts) qui compte les True).
    verdicts = [judge_chunk_relevance(question, c["text"]) for c in chunks]
    return sum(verdicts) / len(verdicts)


def judge_faithfulness(context: str, answer: str) -> str:
    """
    Utilise le LLM comme juge pour estimer si la réponse générée est
    fidèle au contexte fourni, ou si elle contient des informations
    inventées (hallucination) absentes des documents sources.

    Args:
        context: texte des chunks utilisés pour générer la réponse.
        answer: réponse générée par le chatbot à évaluer.

    Returns:
        Un verdict textuel : "fidèle", "hallucination_probable", ou
        "incertain" (voir FAITHFULNESS_PROMPT pour les instructions
        exactes données au juge).
    """
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer)}],
        temperature=0.0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip().lower()


def evaluate_question(question: str) -> EvalResult:
    """
    Évalue une question unique sur l'ensemble des métriques : exécute le
    pipeline RAG complet, mesure le temps de réponse, calcule la precision@k,
    et détermine le verdict de fidélité approprié selon le cas de figure.

    Args:
        question: question à évaluer.

    Returns:
        Un EvalResult regroupant toutes les métriques mesurées.
    """
    # time.time() avant/après permet de mesurer la durée réelle d'exécution,
    # incluant tous les appels API (retrieval + génération).
    start = time.time()
    result = ask(question)
    elapsed = time.time() - start

    # Score de pertinence moyen des chunks récupérés (indicateur de la
    # qualité du retrieval, indépendant de la qualité de la génération).
    scores = [c.get("similarity_score", 0.0) for c in result.sources]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Trois cas de figure distincts pour déterminer le verdict de fidélité :
    if result.was_rejected_low_relevance:
        # Cas 1 : le garde-fou a bloqué la question avant même d'appeler le
        # LLM — pas de génération, donc pas de fidélité à évaluer.
        precision = 0.0
        verdict = "rejeté_par_garde-fou"
    elif is_honest_refusal(result.answer):
        # Cas 2 : le LLM a répondu, mais pour dire honnêtement qu'il ne
        # savait pas. On calcule quand même la precision@k (utile pour
        # savoir si de meilleurs chunks auraient permis de répondre), mais
        # on n'appelle PAS judge_faithfulness : comparer un refus au
        # contexte n'a pas de sens (le refus, par nature, ne "correspond"
        # à aucun passage précis du contexte) et produirait un faux verdict
        # d'hallucination.
        precision = compute_precision_at_k(question, result.sources)
        verdict = "refus_honnête"
    else:
        # Cas 3 : cas normal — le LLM a tenté une réponse factuelle, on
        # vérifie sa fidélité au contexte fourni.
        precision = compute_precision_at_k(question, result.sources)
        context = "\n".join(c["text"] for c in result.sources)
        verdict = judge_faithfulness(context, result.answer)

    return EvalResult(
        question=question,
        answer=result.answer,
        response_time_sec=round(elapsed, 2),
        avg_similarity_score=round(avg_score, 3),
        precision_at_k=round(precision, 2),
        n_sources_used=len(result.sources),
        faithfulness_verdict=verdict,
        was_rejected_low_relevance=result.was_rejected_low_relevance,
    )


def compare_rag_vs_no_rag(question: str) -> dict:
    """
    Compare la réponse générée AVEC RAG et SANS RAG (LLM seul, sans
    contexte documentaire), pour démontrer concrètement l'intérêt du RAG :
    typiquement, la version sans RAG devrait soit halluciner une réponse,
    soit refuser de répondre faute d'information fiable, alors que la
    version avec RAG répond correctement en citant ses sources.

    Args:
        question: question utilisée pour la comparaison.

    Returns:
        Un dictionnaire avec la question, les deux réponses (avec/sans
        RAG), et la liste des documents sources utilisés côté RAG.
    """
    rag_result = ask(question)
    no_rag_answer = ask_without_rag(question)
    return {
        "question": question,
        "with_rag": rag_result.answer,
        "without_rag": no_rag_answer,
        "sources_used": [c["source_document"] for c in rag_result.sources],
    }


def run_evaluation(questions: list[str] = None,
                    out_path: Path = Path("evaluation_report.json"),
                    comparison_question: str = None) -> list[EvalResult]:
    """
    Lance l'évaluation complète sur un jeu de questions, calcule les
    statistiques globales (moyennes), et sauvegarde tout dans un fichier
    JSON exploitable pour le rapport d'analyse des performances.

    Args:
        questions: liste des questions à évaluer. Si None, utilise
            TEST_QUESTIONS (les 16 questions par défaut).
        out_path: chemin du fichier JSON de sortie.
        comparison_question: question utilisée pour la comparaison
            RAG vs sans-RAG. Si None, utilise la première question testée.

    Returns:
        Liste des EvalResult, un par question évaluée.
    """
    questions = questions or TEST_QUESTIONS
    # On évalue chaque question une par une (pas de parallélisation, pour
    # rester dans les limites de débit de l'API gratuite).
    results = [evaluate_question(q) for q in questions]

    report = {
        # asdict() convertit chaque dataclass EvalResult en dictionnaire
        # simple, sérialisable en JSON (une dataclass ne l'est pas directement).
        "results": [asdict(r) for r in results],
        "summary": {
            "avg_precision_at_k": round(
                sum(r.precision_at_k for r in results) / len(results), 2
            ) if results else 0.0,
            "avg_response_time_sec": round(
                sum(r.response_time_sec for r in results) / len(results), 2
            ) if results else 0.0,
            # sum() sur une liste de booléens compte le nombre de True
            # (donc le nombre de questions rejetées par le garde-fou).
            "n_low_relevance_rejections": sum(r.was_rejected_low_relevance for r in results),
        },
    }

    # Comparaison RAG vs sans-RAG sur une question représentative, pour
    # illustrer concrètement l'apport du RAG dans le rapport final.
    comparison_question = comparison_question or questions[0]
    report["rag_vs_no_rag_example"] = compare_rag_vs_no_rag(comparison_question)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Affichage lisible dans le terminal, en plus du fichier JSON,
    # pour un retour immédiat sans avoir à ouvrir le fichier.
    print(f"✓ Évaluation terminée sur {len(results)} questions → {out_path}")
    for r in results:
        flag = " [GARDE-FOU]" if r.was_rejected_low_relevance else ""
        print(f"  - {r.question[:55]}...{flag}")
        print(f"    similarité={r.avg_similarity_score} | precision@k={r.precision_at_k} | "
              f"temps={r.response_time_sec}s | verdict={r.faithfulness_verdict}")

    print(f"\nRésumé : precision@k moyenne = {report['summary']['avg_precision_at_k']} | "
          f"temps moyen = {report['summary']['avg_response_time_sec']}s | "
          f"questions rejetées par le garde-fou = {report['summary']['n_low_relevance_rejections']}")

    return results


if __name__ == "__main__":
    run_evaluation()
