"""
Module bonus : détection et traduction de la Darija marocaine.

Étape 1 : détecter si la question est en Darija
Étape 2 : traduire Darija → Français avant la recherche vectorielle
Étape 3 : (fait dans generator.py) générer la réponse finale en Darija si besoin

Note : la Darija n'étant pas standardisée et peu couverte par les outils
classiques de détection de langue, on combine ici une heuristique simple
(mots-clés fréquents en darija latinisée) avec un appel au LLM en cas de doute.
"""
from pathlib import Path
from openai import OpenAI

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import API_BASE_URL, API_KEY, CHAT_MODEL

_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Mots/particules très fréquents en Darija (écriture latine) — heuristique rapide
DARIJA_MARKERS = {
    "wach", "wash", "bghit", "3afak", "afak", "chno", "kifach", "kif",
    "labas", "mzyan", "makay", "khoya", "3andi", "andi", "hna", "dyal",
    "b7al", "wakha", "safi", "3lach", "3la", "fin", "imta", "shwiya",
}


def detect_darija(text: str) -> bool:
    """
    Heuristique légère pour détecter si un texte est probablement en Darija
    (écrite en caractères latins, comme "wach", "chno"...). Retourne False
    pour un texte en français standard.

    Cette approche est volontairement simple (comparaison de mots-clés,
    pas de modèle de détection de langue) car la Darija n'étant pas une
    langue standardisée à l'écrit, les outils classiques de détection de
    langue (souvent entraînés sur des langues officielles) la reconnaissent
    mal ou pas du tout.

    Args:
        text: texte à analyser (ex. la question posée par l'utilisateur).

    Returns:
        True si au moins un mot du texte correspond à un marqueur Darija
        connu, False sinon.
    """
    # strip(".,!?") retire la ponctuation en début/fin de chaque mot, pour
    # que "wach?" soit quand même reconnu comme "wach".
    words = {w.strip(".,!?").lower() for w in text.split()}
    # L'opérateur & entre deux sets Python calcule leur intersection :
    # si au moins un mot du texte est aussi dans DARIJA_MARKERS, l'ensemble
    # résultat n'est pas vide.
    return len(words & DARIJA_MARKERS) > 0


def translate_darija_to_french(text: str) -> str:
    """
    Traduit un texte en Darija marocaine vers le français, via un appel LLM
    dédié à la traduction (température à 0 pour un résultat déterministe et
    fidèle, sans "créativité" indésirable dans une tâche de traduction).

    Args:
        text: texte en Darija à traduire.

    Returns:
        La traduction en français, sans commentaire ajouté par le modèle.
    """
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un traducteur Darija marocaine → français. "
                    "Traduis fidèlement le message suivant en français standard, "
                    "sans ajouter de commentaire, retourne uniquement la traduction."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,  # déterministe : on veut une traduction fidèle, pas créative
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def translate_french_to_darija(text: str) -> str:
    """
    Traduit un texte français vers une Darija simple et compréhensible,
    écrite en caractères latins. Non utilisée directement dans le pipeline
    actuel (generator.py demande au LLM de générer directement la réponse
    en Darija plutôt que de traduire après coup), mais disponible comme
    fonction utilitaire réutilisable.

    Args:
        text: texte en français à traduire.

    Returns:
        La traduction en Darija (caractères latins), sans commentaire.
    """
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un traducteur français → Darija marocaine simple, "
                    "écrite en caractères latins, compréhensible par un large public. "
                    "Retourne uniquement la traduction, sans commentaire."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,  # légèrement plus élevée : autorise une formulation plus naturelle
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()
