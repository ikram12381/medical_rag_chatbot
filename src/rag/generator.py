"""
Générateur de réponses : envoie le contexte récupéré + la question
au LLM (via l'API compatible OpenAI) pour produire une réponse
basée uniquement sur les sources fournies.
"""
from pathlib import Path
from openai import OpenAI

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import API_BASE_URL, API_KEY, CHAT_MODEL, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT, DISCLAIMER

# Client OpenAI créé une seule fois au chargement du module (pas à chaque
# appel de fonction), pour éviter de recréer une connexion à chaque requête.
_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def build_prompt(question: str, context: str) -> str:
    """
    Construit le prompt utilisateur en combinant le contexte documentaire
    récupéré (les chunks pertinents) et la question posée.

    Args:
        question: question de l'utilisateur, déjà traduite en français
            si elle avait été posée en Darija.
        context: texte des chunks récupérés par le retriever, déjà formaté
            avec les étiquettes [Source N] (voir retriever.format_context).

    Returns:
        Le prompt complet à envoyer au LLM en tant que message "user".
    """
    return f"""Contexte (extrait de sources médicales marocaines officielles) :
---
{context}
---

Question de l'utilisateur : {question}

Réponds en te basant uniquement sur le contexte ci-dessus. Si l'information
n'y figure pas, dis-le clairement plutôt que d'inventer une réponse."""


def generate_answer(question: str, context: str, language: str = "fr") -> str:
    """
    Génère une réponse à partir du contexte récupéré, en appelant le LLM
    via l'API compatible OpenAI (GitHub Models par défaut).

    Args:
        question: question de l'utilisateur (déjà en français).
        context: chunks de contexte formatés (voir build_prompt).
        language: "fr" pour une réponse en français, "darija" pour forcer
            une réponse en Darija marocaine (bonus linguistique).

    Returns:
        La réponse générée par le LLM, avec le disclaimer médical ajouté
        automatiquement à la fin (voir config.DISCLAIMER).
    """
    system_prompt = SYSTEM_PROMPT

    if language == "darija":
        # Instruction renforcée et explicite : le contexte documentaire et le
        # prompt système ci-dessus sont en français (sources officielles +
        # exemple de citation), ce qui pousse naturellement le LLM à répondre
        # en français par défaut. Une simple ligne ajoutée en fin de prompt
        # est trop faible face à ce contenu majoritairement français : il faut
        # une consigne explicite, répétée, et positionnée en priorité.
        system_prompt += (
            "\n\nCONSIGNE PRIORITAIRE — LANGUE DE LA RÉPONSE :\n"
            "Même si le contexte fourni ci-dessus est rédigé en français, tu dois "
            "IMPÉRATIVEMENT rédiger ta réponse ENTIÈREMENT en Darija marocaine "
            "simple, écrite en caractères latins (ex. 'chno', 'kayn', 'l-marad'). "
            "N'écris aucune phrase en français dans ta réponse, y compris les "
            "titres et les listes. Garde uniquement les citations [Source X] "
            "et les termes médicaux techniques (ex. HbA1c, mg/dL) tels quels."
        )

    user_prompt = build_prompt(question, context)

    # Appel à l'API de complétion de chat : on envoie deux messages,
    # un "system" (les règles de comportement) et un "user" (la vraie
    # question + son contexte), comme pour toute API compatible OpenAI.
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,  # basse (0.3) : réponses factuelles, peu créatives
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content

    # Le disclaimer est ajouté ici, en code, et non laissé à la seule
    # discrétion du prompt : ainsi il apparaît TOUJOURS, même si le LLM
    # "oublie" de le mentionner lui-même.
    return answer + DISCLAIMER
