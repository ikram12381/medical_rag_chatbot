"""
Retriever hybride : combine recherche sémantique (FAISS) et recherche
lexicale (BM25) via une fusion de rangs (Reciprocal Rank Fusion, RRF),
pour améliorer la pertinence — notamment sur les termes médicaux précis
que la seule recherche par embedding peut manquer.

Si l'index BM25 n'a pas été construit, le retriever retombe automatiquement
sur la recherche sémantique seule (pas d'erreur bloquante).
"""
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TOP_K, MIN_CHUNK_SCORE
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Store

_vector_store = None
_bm25_store = None
_bm25_unavailable = False  # évite de retenter le chargement à chaque appel

RRF_K = 60  # constante standard de la Reciprocal Rank Fusion


def get_vector_store() -> VectorStore:
    """
    Charge le VectorStore (FAISS) une seule fois et le réutilise ensuite
    (pattern "singleton"), pour éviter de recharger l'index à chaque appel
    de retrieve() — le chargement d'un index FAISS a un coût non négligeable.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_bm25_store():
    """
    Charge le BM25Store une seule fois, avec gestion "gracieuse" de l'absence
    d'index BM25 : si le fichier n'existe pas (pas encore construit), on
    retourne None au lieu de lever une erreur, pour que le retriever puisse
    retomber automatiquement sur la recherche sémantique seule.

    Returns:
        L'instance BM25Store, ou None si l'index n'est pas disponible.
    """
    global _bm25_store, _bm25_unavailable
    # Si on a déjà constaté l'absence de l'index lors d'un appel précédent,
    # inutile de retenter (évite de répéter l'erreur à chaque question posée).
    if _bm25_unavailable:
        return None
    if _bm25_store is None:
        try:
            _bm25_store = BM25Store()
        except FileNotFoundError:
            _bm25_unavailable = True
            return None
    return _bm25_store


def _chunk_key(chunk: dict) -> tuple:
    """
    Construit une clé unique identifiant un chunk, utilisée pour repérer
    si un même chunk apparaît à la fois dans les résultats FAISS et BM25
    (nécessaire pour la fusion RRF ci-dessous).

    Args:
        chunk: dictionnaire représentant un chunk.

    Returns:
        Tuple (nom_du_document, position_dans_le_document), qui identifie
        un chunk de façon unique et stable, contrairement à son texte qui
        pourrait théoriquement se répéter.
    """
    return (chunk["source_document"], chunk["chunk_index_in_doc"])


def retrieve(query: str, top_k: int = TOP_K, use_hybrid: bool = True) -> list[dict]:
    """
    Récupère les top_k chunks les plus pertinents pour la requête.

    Si use_hybrid=True et que l'index BM25 est disponible : combine la
    recherche sémantique (FAISS) et lexicale (BM25) via Reciprocal Rank
    Fusion (RRF). Sinon : recherche sémantique seule.

    Args:
        query: texte de la requête (question déjà traduite en français).
        top_k: nombre de chunks à retourner au maximum.
        use_hybrid: si False, force une recherche sémantique pure même si
            l'index BM25 est disponible (utile pour comparer les deux modes).

    Returns:
        Liste de chunks, chacun enrichi selon sa provenance :
        - similarity_score : score cosinus FAISS (0 à 1), si trouvé par FAISS
        - bm25_score       : score lexical BM25, si trouvé par BM25
        - fused_score      : score de fusion RRF (utilisé pour le tri final)
        La liste est aussi filtrée par _filter_weak_chunks() pour écarter
        les chunks trop peu pertinents (voir MIN_CHUNK_SCORE dans config.py).
    """
    vector_store = get_vector_store()
    # On récupère plus de candidats que top_k avant la fusion (x3, minimum
    # 15), car après fusion certains candidats seront écartés — mieux vaut
    # avoir un pool large pour ne pas manquer de bons résultats.
    candidate_k = max(top_k * 3, 15)
    semantic_results = vector_store.search(query, top_k=candidate_k)

    bm25_store = get_bm25_store() if use_hybrid else None

    # --- Cas 1 : pas de recherche hybride (BM25 indisponible ou désactivé) ---
    if bm25_store is None:
        for chunk in semantic_results:
            chunk["fused_score"] = chunk.get("similarity_score", 0.0)
        return _filter_weak_chunks(semantic_results[:top_k])

    # --- Cas 2 : recherche hybride (fusion FAISS + BM25) ---
    lexical_results = bm25_store.search(query, top_k=candidate_k)

    # Reciprocal Rank Fusion (RRF) : au lieu de comparer directement les
    # scores FAISS et BM25 (qui ne sont pas sur la même échelle numérique,
    # donc pas comparables tels quels), on combine leurs RANGS. Un chunk
    # bien classé dans les deux recherches obtient un score RRF élevé.
    # Formule : score += 1 / (RRF_K + rang + 1) pour chaque apparition.
    rrf_scores = {}
    chunk_lookup = {}  # retrouve le chunk complet à partir de sa clé

    for rank, chunk in enumerate(semantic_results):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[key] = chunk

    for rank, chunk in enumerate(lexical_results):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)
        if key in chunk_lookup:
            # Le chunk a été trouvé par FAISS ET BM25 : on complète ses
            # infos avec le score BM25, sans perdre le score FAISS déjà présent.
            chunk_lookup[key]["bm25_score"] = chunk["bm25_score"]
        else:
            # Le chunk n'a été trouvé QUE par BM25 (pas par FAISS).
            chunk_lookup[key] = chunk

    # On trie toutes les clés par score RRF décroissant, et on ne garde
    # que les top_k meilleures.
    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

    results = []
    for key in ranked_keys:
        chunk = dict(chunk_lookup[key])
        chunk["fused_score"] = rrf_scores[key]
        results.append(chunk)
    return _filter_weak_chunks(results)


def _filter_weak_chunks(chunks: list[dict], min_score: float = MIN_CHUNK_SCORE) -> list[dict]:
    """
    Écarte les chunks dont le score sémantique est trop faible pour être
    réellement pertinents, même s'ils faisaient partie du top_k initial
    (ils n'y étaient parfois que faute de meilleurs candidats disponibles).
    Ce filtre améliore la precision@k en évitant d'envoyer au LLM des
    passages peu utiles qui "diluent" le contexte.

    Args:
        chunks: liste de chunks déjà triés par pertinence.
        min_score: score minimal de similarité sémantique pour qu'un chunk
            soit conservé (voir config.MIN_CHUNK_SCORE).

    Returns:
        La liste filtrée. Garde-fou : si le filtre éliminerait TOUS les
        chunks, on conserve quand même le meilleur d'entre eux, pour éviter
        un contexte totalement vide qui ferait perdre toute information.
    """
    if not chunks:
        return chunks
    # Un chunk trouvé uniquement via BM25 (sans similarity_score, car BM25
    # ne calcule pas de score sémantique) est conservé par défaut : la
    # valeur par défaut 1.0 passe toujours le seuil min_score.
    filtered = [c for c in chunks if c.get("similarity_score", 1.0) >= min_score]
    return filtered if filtered else chunks[:1]


def format_context(chunks: list[dict]) -> str:
    """
    Formate les chunks récupérés en un bloc de texte numéroté, prêt à être
    inséré dans le prompt envoyé au LLM. La numérotation [Source N] permet
    au LLM de citer précisément quelle source il utilise pour chaque
    affirmation (voir SYSTEM_PROMPT dans config.py).

    Args:
        chunks: liste de chunks à formater, dans l'ordre de pertinence.

    Returns:
        Une chaîne de caractères avec un bloc par chunk, séparés par une
        ligne vide.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i} — {chunk['source_document']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def max_semantic_score(chunks: list[dict]) -> float:
    """
    Renvoie le meilleur score sémantique parmi une liste de chunks. Utilisé
    par le garde-fou de pertinence (voir src/rag/pipeline.py) pour décider
    si la question est suffisamment couverte par les documents indexés.

    Args:
        chunks: liste de chunks (résultat de retrieve()).

    Returns:
        Le score de similarité le plus élevé trouvé, ou 0.0 si la liste
        est vide (aucun chunk récupéré).
    """
    scores = [c.get("similarity_score", 0.0) for c in chunks]
    return max(scores) if scores else 0.0
