# Guide de démarrage — Chatbot Médical RAG (Équipe)

Ce guide explique comment installer et exécuter le projet sur ton ordinateur,
étape par étape. Compte environ 15-20 minutes.

---

## Prérequis

- **Python 3.10 ou plus récent** installé ([python.org](https://www.python.org/downloads/))
- **VS Code** installé, avec l'extension **Python** (Microsoft)
- Un compte **GitHub** (gratuit) pour obtenir une clé API

---

## Étape 1 — Récupérer le projet

### Si le projet est sur GitHub (recommandé)

```bash
git clone <URL_DU_DEPOT>
cd medical_rag_chatbot
```

### Si tu as reçu un fichier `.zip`

1. Dézippe-le dans un dossier de ton choix (ex: `Documents\medical_rag_chatbot`)
2. **Vérifie qu'il n'y a pas de double dossier imbriqué** :
   si en ouvrant le dossier dézippé tu retrouves un dossier du même nom à
   l'intérieur, ouvre celui-là. Tu dois arriver dans un dossier qui contient
   directement `config.py`, `requirements.txt`, `main.py`, `src/`, `app/`.
3. Ouvre ce dossier dans VS Code : **File → Open Folder**

---

## Étape 2 — Créer l'environnement virtuel

Ouvre un terminal dans VS Code (`` Ctrl+` ``), puis :

```powershell
python -m venv venv
```

Active-le :

```powershell
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Tu dois voir `(venv)` apparaître au début de la ligne du terminal.

⚠️ **Sur Windows**, vérifie régulièrement avec `pwd` (ou en regardant le chemin
avant `PS>`) que tu es bien dans le dossier du projet — il est facile de se
retrouver dans le mauvais dossier après certaines commandes.

---

## Étape 3 — Installer les dépendances

```powershell
pip install -r requirements.txt
```

---

## Étape 4 — Obtenir TA propre clé API (chacun la sienne)

Chaque membre de l'équipe doit créer **son propre token** — ne partagez jamais
un token entre vous (question de sécurité et de limites de débit individuelles).

1. Va sur **https://github.com/settings/tokens**
2. **Generate new token** → **Tokens (classic)**
3. Donne-lui un nom (ex: `medical-rag-projet`), choisis une expiration
4. **Ne coche aucune case** de permission particulière
5. Clique **Generate token**
6. **Copie le token immédiatement** (il ne sera plus jamais réaffiché)

---

## Étape 5 — Configurer le fichier `.env`

```powershell
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Ouvre le fichier avec :

```powershell
code .env
```

Remplace la ligne :
```
GITHUB_TOKEN=your_github_pat_here
```
par ton propre token, puis sauvegarde (`Ctrl+S`).

### En cas d'erreur SSL au scraping (fréquent sur certains PC/antivirus)

Si tu obtiens une erreur du type `SSL: CERTIFICATE_VERIFY_FAILED`, ajoute cette
ligne à la fin de ton `.env` :
```
SSL_VERIFY=false
```

---

## Étape 6 — Exécuter le pipeline (une seule fois, dans l'ordre)

```powershell
python -m src.scraping.web_scraper       # télécharge pages + PDF officiels
python -m src.scraping.pdf_extractor     # extrait le texte des PDF
python -m src.preprocessing.cleaner      # nettoie les textes
python -m src.preprocessing.chunker      # découpe en segments
python -m src.indexing.vector_store      # construit l'index vectoriel FAISS (utilise ta clé API)
python -m src.indexing.bm25_index        # construit l'index lexical BM25 (recherche hybride)
```

Vérifie qu'il n'y a pas d'erreur après chaque commande avant de passer à la suivante.

---

## Étape 7 — Lancer l'interface

```powershell
streamlit run app/streamlit_app.py
```

Ton navigateur s'ouvre automatiquement sur `http://localhost:8501`.
Pose une question médicale (ex: *"Quels sont les symptômes du diabète type 2 ?"*)
pour tester.

---

## Problèmes fréquents

| Symptôme | Solution |
|---|---|
| `ModuleNotFoundError` | Vérifie que `(venv)` est actif, relance `pip install -r requirements.txt` |
| `FileNotFoundError: ... requirements.txt` | Tu n'es pas dans le bon dossier — vérifie avec `pwd`/`dir` |
| `401 Unauthorized` (API) | Ton `GITHUB_TOKEN` dans `.env` est incorrect ou absent |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Ajoute `SSL_VERIFY=false` dans `.env` |
| `FileNotFoundError: Index introuvable` | L'étape 6 (`vector_store`) n'a pas été faite ou a échoué |
| Rate limit / 429 | GitHub Models a des limites gratuites : attends 1-2 minutes |

---

## Travailler à plusieurs sur le projet

- Chacun a **son propre `.env`** avec **son propre token** (le fichier `.env`
  n'est jamais partagé ni poussé sur Git — il est dans `.gitignore`)
- Le dossier `data/` (PDF téléchargés, index vectoriel) peut être régénéré par
  chacun localement en relançant l'étape 6 — pas besoin de le partager
- Pour contribuer au code : utilisez des branches Git et des pull requests
  si le dépôt est sur GitHub
