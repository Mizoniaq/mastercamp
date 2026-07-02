# Données réelles (à fournir par l'utilisateur)

Ce dossier est **vide par défaut**. Aucune image médicale réelle n'est
redistribuée dans ce dépôt (licences + dé-identification). Pour évaluer le
prototype sur de vraies radiographies, obtenez un **petit échantillon autorisé**
puis lancez le script d'ingestion.

## Sources publiques (accepter la licence AVANT tout téléchargement)

| Dataset | Accès | Licence / conditions |
|---|---|---|
| RSNA Pneumonia Detection | Kaggle (compte + acceptation des règles du challenge) | usage recherche, non redistribuable |
| NIH ChestX-ray14 | NIH Box / Kaggle | domaine public US, citer la source |
| CheXpert | Stanford AIMI (enregistrement + accord d'usage) | recherche uniquement, non redistribuable |
| MIMIC-CXR / -JPG | PhysioNet (accès crédentialé + formation CITI) | strictement contrôlé, non redistribuable |

> **Ne commitez jamais** d'images patient réelles, même pseudonymisées, sans
> autorisation explicite et traçable. Ce dossier (hors ce README) est ignoré par git.

## Chemin rapide recommandé (le plus simple)

Le dataset Kaggle **« Chest X-Ray Images (Pneumonia) »**
(`paultimothymooney/chest-xray-pneumonia`) est le plus adapté : images **JPEG**
(pas de DICOM à convertir), deux classes qui correspondent directement au projet.

**Mapping vers nos classes** : `NORMAL` → `normal`, `PNEUMONIA` → `suspected_opacity`.

```bash
# 1. Installer et configurer l'API Kaggle (compte Kaggle + token kaggle.json)
pip install kaggle
#    Déposez kaggle.json dans %USERPROFILE%\.kaggle\  (Windows)

# 2. Télécharger et décompresser (accepter les conditions du dataset sur le site)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/real --unzip

# 3. Prendre un petit échantillon (~15 normal + ~15 pneumonia) et le poser dans
#    data/real/images/ , puis créer data/real/labels.csv :
#        filename,label
#        NORMAL-xxxx.jpeg,normal
#        person-xxxx_bacteria.jpeg,suspected_opacity
```

> Sans compte Kaggle, téléchargez manuellement depuis la page du dataset. **Citez
> la source et la licence dans le rapport** (usage recherche, non redistribuable) —
> et ne commitez aucune image.

## Ingestion

1. Placez 20-30 images dé-identifiées dans `data/real/images/`.
2. (Optionnel) Fournissez un CSV `labels.csv` avec les colonnes
   `filename,label` (label ∈ `normal`, `suspected_opacity`, `uncertain`).
3. Générez le catalogue au schéma du projet :

```bash
python scripts/prepare_real_dataset.py --images data/real/images --labels data/real/labels.csv
```

4. Évaluez le pipeline sur ce jeu réel :

```bash
python eval/run_evaluation.py --mode toy --cases data/real/real_cases.csv
```

5. Faites tourner le **vrai modèle médical** sur ces images réelles :

```bash
$env:HF_TOKEN="ta_cle"
python eval/run_vlm_comparison.py --model google/medgemma-4b-it --cases data/real/real_cases.csv --limit 0
```

Le classifieur jouet fonctionne sur n'importe quelle image, mais ses features sont
calées sur le jeu synthétique : sur de vraies radios il constitue une **baseline
faible et honnête**. L'intérêt est surtout de faire tourner **MedGemma** : sur de
vraies opacités, il devrait produire une sensibilité **> 0** (contrairement au jeu
synthétique), ce qui démontre enfin un vrai signal médical.
