"""
Segmentation des documents nettoyés en chunks de 300-800 tokens
avec chevauchement (overlap), pour préserver le contexte entre segments.

Usage :
    python -m src.preprocessing.chunker
"""
import json
from pathlib import Path
from tqdm import tqdm
import tiktoken

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR, CHUNKS_DIR, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

# Tokenizer utilisé uniquement pour compter/découper (compatible modèles OpenAI)
ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Découpe un texte en chunks de `chunk_size` tokens avec un chevauchement
    de `overlap` tokens entre segments consécutifs.

    Le chevauchement évite de "couper" une idée en plein milieu entre deux
    chunks : la fin d'un chunk se retrouve aussi au début du suivant, ce qui
    préserve le contexte pour la recherche et la génération.

    Args:
        text: texte à découper (déjà nettoyé).
        chunk_size: taille maximale d'un chunk, en tokens.
        overlap: nombre de tokens partagés entre deux chunks consécutifs.

    Returns:
        Liste des segments de texte (chaînes de caractères), dans l'ordre
        d'apparition dans le document original.
    """
    # encode() transforme le texte en une liste d'entiers (tokens), la
    # même unité que celle utilisée par les modèles de langage pour
    # mesurer et facturer la longueur d'un texte.
    tokens = ENCODING.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    # "step" = de combien de tokens on avance à chaque nouveau chunk.
    # Si overlap >= chunk_size, on force un minimum de 1 pour éviter une
    # boucle infinie (l'avancement ne serait jamais négatif ou nul).
    step = max(chunk_size - overlap, 1)

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        # decode() reconvertit les tokens en texte lisible.
        chunk_str = ENCODING.decode(chunk_tokens)
        chunks.append(chunk_str.strip())

        # Si on a atteint la fin du texte, on arrête (sinon on créerait
        # un dernier chunk minuscule ou dupliqué).
        if end == len(tokens):
            break
        start += step

    return chunks


def process_all(processed_dir: Path = PROCESSED_DATA_DIR,
                 out_dir: Path = CHUNKS_DIR) -> None:
    """
    Découpe tous les documents nettoyés en chunks, et sauvegarde l'ensemble
    dans un seul fichier JSONL (un objet JSON par ligne), pratique pour être
    lu ligne par ligne sans charger tout le fichier en mémoire d'un coup.

    Args:
        processed_dir: dossier contenant les fichiers .txt nettoyés.
        out_dir: dossier où écrire le fichier chunks.jsonl final.
    """
    txt_files = list(processed_dir.glob("*.txt"))
    if not txt_files:
        print("⚠️  Aucun fichier nettoyé trouvé dans data/processed/.")
        return

    all_chunks = []
    # Identifiant unique et croissant, tous documents confondus (utile pour
    # retrouver un chunk précis dans l'index vectoriel par la suite).
    chunk_id = 0

    for txt_path in tqdm(txt_files, desc="Chunking des documents"):
        text = txt_path.read_text(encoding="utf-8")
        doc_chunks = chunk_text(text)

        for i, chunk in enumerate(doc_chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                # .stem = nom du fichier sans son extension, ex. "guide_risque_cardiovasculaire"
                "source_document": txt_path.stem,
                "chunk_index_in_doc": i,  # position du chunk DANS son document d'origine
                "text": chunk,
                "n_tokens": len(ENCODING.encode(chunk)),
            })
            chunk_id += 1

    out_path = out_dir / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            # ensure_ascii=False conserve les accents français lisibles
            # dans le fichier, plutôt que de les échapper en \uXXXX.
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✓ {len(all_chunks)} chunks créés → {out_path}")


if __name__ == "__main__":
    process_all()
