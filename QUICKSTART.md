# 🚀 Guide de démarrage — lancer le projet

Guide **pas à pas** pour installer et lancer l'assistant radiologue virtuel sur
n'importe quelle machine. Aucune connaissance préalable requise.

> ⚠️ **Prototype pédagogique. Non destiné au diagnostic.** Aucune sortie ne doit
> servir à une décision médicale réelle.

---

## 1. Prérequis

- **Python 3.11+** ([télécharger](https://www.python.org/downloads/)) — cocher
  « Add Python to PATH » à l'installation sous Windows.
- **Git** (facultatif, si vous clonez le dépôt).
- *(Facultatif, avancé)* un **GPU NVIDIA** + un **token Hugging Face** pour le vrai
  modèle médical MedGemma. **Pas nécessaire** pour la démo : le moteur « jouet »
  tourne partout, sans GPU ni token.

Vérifier Python :
```bash
python --version        # doit afficher 3.11 ou plus
```

## 2. Récupérer le code et se placer dans le dossier

```bash
# Si vous clonez :
git clone <URL_DU_DEPOT> assistant-radio
cd assistant-radio
# Sinon, ouvrez simplement un terminal DANS le dossier du projet.
```

## 3. Installer les dépendances (dans un environnement isolé)

**Windows (PowerShell) :**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux :**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 💡 Vous devez voir `(.venv)` au début de la ligne de commande. Pour ré-activer
> l'environnement plus tard, relancez la commande `activate`.

## 4. Lancer l'application web (la démo principale)

```bash
streamlit run app/streamlit_app.py
```
Le navigateur s'ouvre sur **http://localhost:8501**. Sinon, ouvrez ce lien à la
main. Pour **arrêter** : `Ctrl + C` dans le terminal.

**Que faire dans l'app :**
1. Onglet **🔬 Prédiction** → choisir *Mode* (Baseline / Amélioré) et *Moteur*
   (Jouet rapide).
2. Choisir un **cas d'exemple** dans le menu, ou **importer** une image.
3. Lire le **résultat** : classe, confiance, observations, justification, warning,
   et le **JSON** structuré.
4. Onglets **Données / Visualisations / Tableau de bord** pour explorer le jeu de
   données, les métriques (baseline vs amélioré) et le journal des prédictions.

## 5. Générer les preuves (métriques, erreurs) — optionnel

```bash
python eval/run_evaluation.py --mode toy      # métriques baseline vs amélioré -> eval/results/
python eval/build_error_register.py           # registre d'erreurs (30 cas) -> eval/error_register.csv
python eval/threshold_sweep.py                # analyse de sensibilité au seuil
```

## 6. Lancer les tests (vérifier que tout marche)

```bash
pip install -r requirements-test.txt
python -m pytest -q                           # 15 tests doivent passer
```

## 7. Autres interfaces (facultatif)

```bash
# API REST
uvicorn api.main:app --reload
#   -> http://localhost:8000/docs   (POST /predict, GET /history)

# Interface Gradio
python app/gradio_app.py
```

---

## 8. Avancé — le vrai modèle médical MedGemma (GPU requis)

Le moteur « MedGemma (réel) » de l'app et le script de comparaison utilisent un
vrai modèle. Il faut :
1. accepter la licence sur <https://huggingface.co/google/medgemma-4b-it> ;
2. fournir un token Hugging Face.

```powershell
# Windows
$env:HF_TOKEN="votre_token_hf"
python eval/run_vlm_comparison.py --model google/medgemma-4b-it
```
```bash
# macOS / Linux
export HF_TOKEN="votre_token_hf"
python eval/run_vlm_comparison.py --model google/medgemma-4b-it
```
Sans accès *gated*, un modèle ouvert valide le harnais :
`--model HuggingFaceTB/SmolVLM-256M-Instruct --limit 9`.

### Évaluer sur de vraies radios (dataset Kaggle)

Aucune image réelle n'est incluse dans le dépôt (licence + confidentialité). Pour
en ajouter un petit échantillon autorisé :

```bash
# 1. Télécharger le dataset public (compte Kaggle + kaggle.json requis)
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/real --unzip

# 2. Construire le catalogue (labels déduits des dossiers NORMAL/PNEUMONIA)
python scripts/prepare_real_dataset.py --images data/real/chest_xray --per-class 15

# 3. Évaluer MedGemma sur ces vraies radios
python eval/run_vlm_comparison.py --model google/medgemma-4b-it --cases data/real/real_cases.csv --limit 0
```

`NORMAL` → `normal`, `PNEUMONIA` → `suspected_opacity`. Les images restent
**gitignorées**. Sources, licences et détails : [`data/real/README.md`](data/real/README.md).
Pensez à **citer la source et la licence** du dataset dans le rapport.

---

## 9. Problèmes fréquents

| Symptôme | Solution |
|---|---|
| `python` introuvable | Réinstaller Python en cochant « Add to PATH », rouvrir le terminal |
| `ModuleNotFoundError` | L'environnement n'est pas activé, ou `pip install -r requirements.txt` a échoué |
| `streamlit` : commande inconnue | Activer `.venv` puis réinstaller les dépendances |
| Port 8501 déjà utilisé | `streamlit run app/streamlit_app.py --server.port 8502` |
| MedGemma : erreur **403 / gated** | Accepter la licence sur la page du modèle + définir `HF_TOKEN` |
| MedGemma très lent | Normal (~30 s/image sur GPU) ; rester sur le moteur « Jouet » pour une démo rapide |

---

Détails du projet, barème et architecture : voir le [README](README.md) et
[`docs/`](docs/). Rapport complet : [`docs/rapport.md`](docs/rapport.md).
