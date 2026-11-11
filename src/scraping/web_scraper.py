"""
Scraper pour les sources médicales marocaines officielles :
- Ministère de la Santé et de la Protection Sociale
- OMS Maroc
- CHU marocains
- Campagnes nationales de santé

Usage :
    python -m src.scraping.web_scraper
"""
import os
import time
import requests
import urllib3
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MedicalRAGBot/1.0; Projet académique UCD)"
}

# Sur certains PC (souvent à cause d'un antivirus qui inspecte le trafic HTTPS
# ou d'un magasin de certificats Windows incomplet), la vérification SSL peut
# échouer même pour un site légitime. Mettre SSL_VERIFY=false dans .env pour
# désactiver la vérification en dernier recours (usage académique uniquement).
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").strip().lower() != "false"
if not SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("⚠️  Vérification SSL désactivée (SSL_VERIFY=false dans .env).")

# Pages web du Ministère de la Santé à scraper (texte des pages elles-mêmes)
SOURCES = [
    {"name": "sante_gov_accueil", "url": "https://www.sante.gov.ma/Pages/Accueil.aspx"},
    {"name": "sante_gov_publications", "url": "https://www.sante.gov.ma/Publications/Pages/default.aspx"},
    {"name": "sante_gov_guides_manuels", "url": "https://www.sante.gov.ma/Publications/Guides-Manuels/Pages/default.aspx"},
]

# PDF officiels du Ministère de la Santé et de la Protection Sociale (vérifiés).
# Ces documents couvrent les maladies cardiovasculaires, l'hypertension,
# le diabète (pied diabétique), la vaccination et la surveillance épidémiologique.
PDF_SOURCES = [
    {
        "name": "guide_risque_cardiovasculaire",
        "url": "https://www.sante.gov.ma/Documents/2023/03/Guide%20risque%20cardiovasculaire%20VF%20%20AZ%2008%20juillet%202019.pdf",
    },
    {
        "name": "hta_modes_de_vie_sains",
        "url": "https://www.sante.gov.ma/Documents/2023/03/HTA%20et%20modes%20de%20vie%20sains%20VF%20janv%202020.pdf",
    },
    {
        "name": "hta_prevention_complications",
        "url": "https://www.sante.gov.ma/Documents/2023/03/HTA%20et%20pr%C3%A9vention%20des%20complications%20VF%20janv%202020.pdf",
    },
    {
        "name": "hypertension_arterielle_adulte",
        "url": "https://www.sante.gov.ma/Publications/Documents/Hypertention%20art%C3%A9rielle%20de%20l'adulte.pdf",
    },
    {
        "name": "calendrier_national_vaccination",
        "url": "https://www.sante.gov.ma/PublishingImages/2018/04/calendrier%20national%20de%20vaccination-%20(1).pdf",
    },
    {
        "name": "guide_vaccination_unicef",
        "url": "https://www.sante.gov.ma/Publications/Guides-Manuels/Documents/Guide%20de%20vaccination%20Unicef.pdf",
    },
    {
        "name": "manuel_surveillance_rougeole_2025",
        "url": "https://www.sante.gov.ma/Documents/2025/01/Circulaire_diffusion_manuel_rougeole.pdf",
    },
    {
        "name": "guide_urgences_pediatriques",
        "url": "https://www.sante.gov.ma/Documents/2018/01/Guid_Urgece_Ped_Valide_25Dec17_From_M.pdf",
    },
    {
        "name": "depliant_pied_diabetique",
        "url": "https://www.sante.gov.ma/Documents/2024/11/depliant_Pied.pdf",
    },
    {
        "name": "sante_en_chiffres_2024",
        "url": "https://www.sante.gov.ma/Documents/2026/06/Sante%20en%20chiffre%202024%20VF.pdf",
    },
]

# Documents OMS (Organisation Mondiale de la Santé) apportant une diversité
# de sources au-delà du seul Ministère de la Santé marocain.
WHO_PDF_SOURCES = [
    {
        "name": "oms_strategie_cooperation_maroc",
        "url": "https://iris.who.int/bitstream/handle/10665/272538/ccsbrief-mar-fre.pdf",
    },
    {
        "name": "oms_politique_sante_enfant_maroc",
        "url": "https://applications.emro.who.int/dsaf/libcat/EMRDOC_6_FR.pdf",
    },
]


def fetch_page(url: str, timeout: int = 15) -> str:
    """
    Télécharge le HTML brut d'une page web.

    Args:
        url: adresse de la page à récupérer.
        timeout: délai maximal d'attente en secondes avant d'abandonner.

    Returns:
        Le contenu HTML de la page sous forme de chaîne de caractères.

    Raises:
        requests.HTTPError: si le serveur répond avec un code d'erreur
            (404, 500, etc.), via raise_for_status().
    """
    response = requests.get(url, headers=HEADERS, timeout=timeout, verify=SSL_VERIFY)
    response.raise_for_status()  # lève une exception si le code HTTP n'est pas 2xx
    return response.text


def extract_text_from_html(html: str) -> str:
    """
    Extrait le texte "utile" d'une page HTML, en retirant les éléments
    qui ne contiennent jamais d'information médicale exploitable
    (menus de navigation, scripts JS, styles CSS, formulaires...).

    Args:
        html: contenu HTML brut de la page.

    Returns:
        Le texte visible de la page, une ligne par élément, sans les
        lignes vides ni les espaces superflus.
    """
    soup = BeautifulSoup(html, "html.parser")

    # decompose() supprime complètement ces balises et leur contenu de
    # l'arbre HTML avant l'extraction du texte (sinon on récupèrerait
    # le code JS/CSS brut comme si c'était du texte de la page).
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
        tag.decompose()

    # get_text(separator="\n") force un retour à la ligne entre chaque
    # élément HTML, plutôt que de tout coller sur une seule ligne.
    text = soup.get_text(separator="\n")

    # Nettoyage basique : on retire les espaces en début/fin de chaque
    # ligne, puis on élimine les lignes devenues vides après ce nettoyage.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def save_raw_text(name: str, text: str) -> Path:
    """
    Sauvegarde un texte extrait dans le dossier data/raw/, au format .txt.

    Args:
        name: nom du fichier à créer (sans extension), ex. "sante_gov_accueil".
        text: contenu texte à écrire.

    Returns:
        Le chemin complet du fichier créé.
    """
    out_path = RAW_DATA_DIR / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def scrape_all(sources: list[dict] = None, delay: float = 1.5) -> None:
    """
    Scrape toutes les pages web définies dans SOURCES (ou une liste fournie
    en paramètre), avec un délai entre chaque requête pour rester "poli"
    envers le serveur distant (évite de le surcharger de requêtes).

    Args:
        sources: liste de dictionnaires {"name": ..., "url": ...}.
            Si None, utilise la constante SOURCES du module.
        delay: pause en secondes entre deux téléchargements successifs.
    """
    sources = sources or SOURCES
    if not sources:
        print("⚠️  Aucune source définie. Complétez la liste SOURCES dans ce fichier.")
        return

    # tqdm affiche une barre de progression dans le terminal pendant la boucle.
    for source in tqdm(sources, desc="Scraping des sources"):
        try:
            html = fetch_page(source["url"])
            text = extract_text_from_html(html)
            path = save_raw_text(source["name"], text)
            print(f"✓ {source['name']} → {path}")
        except Exception as e:
            # On capture toute erreur (réseau, HTTP, parsing...) pour qu'un
            # échec sur UNE source n'interrompe pas le scraping des autres.
            print(f"✗ Erreur pour {source['name']} : {e}")
        time.sleep(delay)


def download_pdf(url: str, name: str, timeout: int = 20) -> Path:
    """
    Télécharge un fichier PDF distant et le sauvegarde tel quel (en binaire)
    dans data/raw/.

    Args:
        url: adresse du PDF à télécharger.
        name: nom du fichier à créer (sans extension .pdf).
        timeout: délai maximal d'attente en secondes.

    Returns:
        Le chemin complet du fichier PDF créé.
    """
    response = requests.get(url, headers=HEADERS, timeout=timeout, verify=SSL_VERIFY)
    response.raise_for_status()
    out_path = RAW_DATA_DIR / f"{name}.pdf"
    # write_bytes (et non write_text) car un PDF est un fichier binaire,
    # pas du texte encodé.
    out_path.write_bytes(response.content)
    return out_path


def download_all_pdfs(pdf_sources: list[dict] = None, delay: float = 1.5) -> None:
    """
    Télécharge tous les PDF officiels définis dans PDF_SOURCES (Ministère de
    la Santé) et WHO_PDF_SOURCES (OMS), sauf si une autre liste est fournie.

    Args:
        pdf_sources: liste de dictionnaires {"name": ..., "url": ...}.
            Si None, combine PDF_SOURCES + WHO_PDF_SOURCES.
        delay: pause en secondes entre deux téléchargements successifs.
    """
    pdf_sources = pdf_sources or (PDF_SOURCES + WHO_PDF_SOURCES)
    if not pdf_sources:
        print("⚠️  Aucune source PDF définie.")
        return

    for source in tqdm(pdf_sources, desc="Téléchargement des PDF officiels"):
        try:
            path = download_pdf(source["url"], source["name"])
            print(f"✓ {source['name']} → {path}")
        except Exception as e:
            print(f"✗ Erreur pour {source['name']} : {e}")
        time.sleep(delay)


# Ce bloc ne s'exécute que si on lance directement ce fichier
# (python -m src.scraping.web_scraper), pas s'il est importé ailleurs.
if __name__ == "__main__":
    scrape_all()
    download_all_pdfs()
