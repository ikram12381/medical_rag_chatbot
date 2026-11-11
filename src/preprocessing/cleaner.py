"""
Nettoyage des textes bruts : suppression des en-têtes/pieds de page,
du bruit HTML résiduel, normalisation des espaces et de la casse.

Usage :
    python -m src.preprocessing.cleaner
"""
import re
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Motifs fréquents de bruit à supprimer.
# Organisés par origine pour faciliter l'adaptation si de nouvelles sources
# sont ajoutées (il suffit de compléter la catégorie correspondante).
NOISE_PATTERNS = [
    # --- Générique (tous types de sources) ---
    r"Copyright ©.*",
    r"Tous droits réservés.*",
    r"^\s*Page \d+ sur \d+\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",              # pagination type "12 / 40"

    # --- Sites web gouvernementaux (sante.gov.ma et similaires) ---
    r"Menu\s*(Accueil|Contact|À propos).*",
    r"Cookies?.*politique.*",
    r"Aller au contenu principal.*",
    r"^\s*(Accueil|Contact|Recherche|Rechercher)\s*$",
    r"^\s*(Français|العربية|Arabic|English)\s*$",  # sélecteurs de langue isolés
    r"Suivez[- ]nous.*",
    r"Partager (cette page|sur).*",
    r"^\s*(Facebook|Twitter|LinkedIn|YouTube|Instagram)\s*$",
    r"Plan du site.*",
    r"Mentions légales.*",
    r"^\s*Précédent\s*\|?\s*Suivant\s*$",

    # --- PDF officiels (Ministère de la Santé / OMS) ---
    r"Ministère de la Santé.*Protection Sociale.*\d{4}",  # en-tête répété par page
    r"Organisation Mondiale de la Santé.*Bureau régional.*",
    r"^\s*ISBN[\s:].*",
    r"^\s*WHO[/-].*\d{4}.*",                # codes de référence OMS type "WHO/XXX/2020"
    r"^\s*Imprimé (au|en).*",
    r"Tous droits réservés.*Organisation mondiale de la [Ss]anté.*",
]


def remove_noise(text: str) -> str:
    """
    Supprime tous les motifs de bruit connus (headers, footers, cookies,
    menus, mentions légales...) définis dans NOISE_PATTERNS.

    Args:
        text: texte brut à nettoyer.

    Returns:
        Le texte avec chaque motif reconnu remplacé par une chaîne vide.
    """
    for pattern in NOISE_PATTERNS:
        # re.IGNORECASE : "Cookies" et "cookies" sont traités pareil.
        # re.MULTILINE : "^" et "$" correspondent au début/fin de CHAQUE
        # ligne (et non seulement du texte entier).
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces : réduit les espaces/tabulations multiples à un
    seul espace, et limite les lignes vides consécutives à deux maximum
    (évite les gros blocs vides laissés par la suppression du bruit).

    Args:
        text: texte à normaliser.

    Returns:
        Le texte normalisé, sans espaces en début/fin (strip()).
    """
    text = re.sub(r"[ \t]+", " ", text)       # espaces/tabs multiples -> 1 espace
    text = re.sub(r"\n{3,}", "\n\n", text)    # 3+ retours à la ligne -> 2 max
    return text.strip()


def remove_short_lines(text: str, min_length: int = 3) -> str:
    """
    Supprime les lignes trop courtes pour être du contenu médical utile
    (souvent des résidus de boutons, puces isolées, numéros de page seuls).

    Args:
        text: texte à filtrer.
        min_length: nombre minimal de caractères pour qu'une ligne soit
            conservée (les lignes vides sont toujours conservées, pour ne
            pas casser la mise en page avant normalize_whitespace()).

    Returns:
        Le texte sans les lignes trop courtes.
    """
    lines = text.split("\n")
    kept = [line for line in lines if len(line.strip()) >= min_length or line.strip() == ""]
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """
    Pipeline complet de nettoyage appliqué à un document, dans un ordre
    précis : d'abord retirer le bruit connu, puis les lignes trop courtes
    (souvent révélées une fois le bruit environnant supprimé), et enfin
    normaliser les espaces restants.

    Args:
        text: texte brut extrait d'une page web ou d'un PDF.

    Returns:
        Le texte nettoyé, prêt pour l'étape de chunking.
    """
    text = remove_noise(text)
    text = remove_short_lines(text)
    text = normalize_whitespace(text)
    return text


def find_frequent_lines(raw_dir: Path = RAW_DATA_DIR, min_occurrences: int = 3, top_n: int = 20) -> None:
    """
    Outil de diagnostic (pas utilisé dans le pipeline automatique) : affiche
    les lignes qui se répètent le plus souvent à travers tous les documents
    bruts. Une ligne très fréquente (en-tête, pied de page, mention légale...)
    est un bon candidat à ajouter à NOISE_PATTERNS.

    Args:
        raw_dir: dossier contenant les fichiers .txt bruts à analyser.
        min_occurrences: nombre minimal de répétitions pour qu'une ligne
            soit affichée (évite d'afficher du bruit statistique).
        top_n: nombre maximal de lignes affichées.

    Usage :
        python -c "from src.preprocessing.cleaner import find_frequent_lines; find_frequent_lines()"
    """
    from collections import Counter

    # Counter compte automatiquement combien de fois chaque ligne apparaît,
    # tous documents confondus.
    counter = Counter()
    for txt_path in raw_dir.glob("*.txt"):
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        counter.update(line.strip() for line in lines if line.strip())

    print(f"Top {top_n} lignes les plus fréquentes (bruit potentiel) :\n")
    # most_common(n) retourne les n éléments les plus fréquents, triés
    # par nombre d'occurrences décroissant.
    for line, count in counter.most_common(top_n):
        if count >= min_occurrences:
            # Tronque les lignes trop longues pour un affichage lisible.
            preview = line[:80] + ("..." if len(line) > 80 else "")
            print(f"  [{count}x] {preview}")


def process_all(raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR) -> None:
    """
    Nettoie tous les fichiers .txt présents dans data/raw/ (pages web
    scrapées + PDF déjà convertis en texte) et sauvegarde le résultat
    dans data/processed/, en gardant le même nom de fichier.

    Args:
        raw_dir: dossier source contenant les textes bruts.
        out_dir: dossier de destination pour les textes nettoyés.
    """
    txt_files = list(raw_dir.glob("*.txt"))
    if not txt_files:
        print("⚠️  Aucun fichier .txt trouvé dans data/raw/.")
        return

    for txt_path in tqdm(txt_files, desc="Nettoyage des textes"):
        raw_text = txt_path.read_text(encoding="utf-8")
        cleaned = clean_text(raw_text)
        out_path = out_dir / txt_path.name
        out_path.write_text(cleaned, encoding="utf-8")
        # Affiche le nombre de caractères avant/après pour visualiser
        # rapidement l'ampleur du nettoyage effectué sur chaque fichier.
        print(f"✓ {txt_path.name} nettoyé ({len(raw_text)} → {len(cleaned)} caractères)")


if __name__ == "__main__":
    process_all()
