"""
Configuration centrale du projet.
Toutes les constantes et paramètres du pipeline RAG sont définis ici, afin
d'éviter de dupliquer des valeurs "en dur" dans plusieurs fichiers : si un
réglage doit changer, il ne se change qu'à un seul endroit.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables définies dans le fichier .env (GITHUB_TOKEN, etc.)
# et les rend accessibles via os.getenv() dans tout le fichier.
load_dotenv()

# --- Chemins ---
# BASE_DIR pointe vers le dossier racine du projet (celui qui contient ce
# fichier config.py), quel que soit l'endroit d'où le script est lancé.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"              # PDF/pages téléchargés (bruts)
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # textes nettoyés
CHUNKS_DIR = DATA_DIR / "chunks"             # segments de texte (chunks.jsonl)
INDEX_DIR = DATA_DIR / "index"               # index FAISS + BM25

# Crée automatiquement l'arborescence data/ si elle n'existe pas encore
# (parents=True crée les dossiers intermédiaires, exist_ok=True évite une
# erreur si le dossier existe déjà).
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNKS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API (GitHub Models - compatible OpenAI, gratuit avec limites) ---
# Token classique : aucune permission particulière requise.
# Token fine-grained : activer la permission "Models" (read-only) dans
# Account permissions. https://github.com/settings/tokens
API_BASE_URL = os.getenv("API_BASE_URL", "https://models.github.ai/inference")
API_KEY = os.getenv("GITHUB_TOKEN", "")

# Modèles utilisés (le préfixe "openai/" est requis par GitHub Models)
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# --- Chunking ---
CHUNK_SIZE_TOKENS = 500       # entre 300 et 800 selon le sujet
CHUNK_OVERLAP_TOKENS = 80

# --- Retrieval ---
TOP_K = 4                    # nombre de documents pertinents à récupérer (réduit de 5 à 4)

# Score minimal de similarité sémantique pour qu'un chunk soit conservé.
# Filtre les chunks "faibles" ramenés uniquement pour compléter le top_k,
# ce qui améliore la precision@k sans nécessiter plus de chunks au total.
MIN_CHUNK_SCORE = 0.28

# --- Génération ---
TEMPERATURE = 0.3
MAX_TOKENS = 700

# --- Garde-fous (guardrails) ---
# Si le meilleur score de similarité sémantique est sous ce seuil, la question
# est considérée comme hors du périmètre couvert par les documents indexés :
# on répond directement sans appeler le LLM (évite les hallucinations et
# économise des appels API).
LOW_RELEVANCE_THRESHOLD = 0.22

LOW_RELEVANCE_MESSAGE = (
    "Je ne dispose pas d'information suffisamment pertinente dans mes sources "
    "médicales marocaines officielles pour répondre précisément à cette question. "
    "N'hésitez pas à reformuler, ou à consulter un professionnel de santé."
)

# Rappel affiché systématiquement après chaque réponse, quel que soit le
# contenu généré par le LLM (garantie côté code, pas seulement via le prompt).
DISCLAIMER = (
    "\n\n---\n⚠️ *Cette réponse est fournie à titre informatif à partir de sources "
    "officielles marocaines. Elle ne remplace pas une consultation médicale "
    "professionnelle.*"
)

SYSTEM_PROMPT = """Tu es un assistant médical d'information générale, destiné à un public marocain.
Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous.
- Si le contexte ne contient pas la réponse, dis clairement que tu ne disposes pas
  de cette information dans les sources disponibles, ne l'invente jamais.
- CITATIONS OBLIGATOIRES : après chaque affirmation factuelle, indique entre
  crochets la source utilisée, ex. "Le diabète de type 2 se caractérise par...[Source 1]".
  Si une phrase s'appuie sur plusieurs sources, cite-les toutes, ex. [Source 1][Source 3].
- Si la question posée n'a aucun rapport avec la santé ou la médecine, décline
  poliment en expliquant que tu es un assistant spécialisé en information médicale.
- Ne remplace jamais un avis médical professionnel : rappelle à l'utilisateur
  de consulter un médecin pour tout diagnostic ou traitement.
- Réponds de façon claire, simple et bienveillante.
"""
