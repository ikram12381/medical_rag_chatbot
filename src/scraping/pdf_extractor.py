"""
Extraction du texte brut à partir des PDF collectés (guides cliniques,
protocoles de prévention, documents du Ministère de la Santé, etc.)

Usage :
    python -m src.scraping.pdf_extractor
"""
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extrait tout le texte d'un fichier PDF, page par page, et les concatène.

    Args:
        pdf_path: chemin vers le fichier PDF à lire.

    Returns:
        Le texte complet du PDF, avec un retour à la ligne entre chaque page.

    Note:
        page.extract_text() peut retourner None sur certaines pages
        (ex: pages composées uniquement d'images scannées, sans couche de
        texte) — d'où le "or \"\"" qui remplace None par une chaîne vide
        pour éviter une erreur lors du join() final.
    """
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def process_all_pdfs(raw_dir: Path = RAW_DATA_DIR) -> None:
    """
    Parcourt tous les fichiers .pdf présents dans data/raw/, extrait leur
    texte, et sauvegarde un fichier .txt de même nom à côté de chaque PDF.

    Args:
        raw_dir: dossier contenant les PDF à traiter (par défaut data/raw/).
    """
    # glob("*.pdf") liste tous les fichiers se terminant par .pdf dans le dossier.
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  Aucun PDF trouvé dans data/raw/.")
        return

    for pdf_path in tqdm(pdf_files, desc="Extraction des PDF"):
        try:
            text = extract_text_from_pdf(pdf_path)
            # with_suffix(".txt") remplace l'extension .pdf par .txt tout
            # en gardant le même nom de fichier et le même dossier.
            out_path = pdf_path.with_suffix(".txt")
            out_path.write_text(text, encoding="utf-8")
            print(f"✓ {pdf_path.name} → {out_path.name}")
        except Exception as e:
            # Un PDF corrompu ou protégé ne doit pas arrêter le traitement
            # des autres fichiers.
            print(f"✗ Erreur pour {pdf_path.name} : {e}")


if __name__ == "__main__":
    process_all_pdfs()
