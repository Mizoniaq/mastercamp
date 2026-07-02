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

Le classifieur jouet fonctionne sur n'importe quelle image ; sur de vraies
radios, il constitue une **baseline faible et honnête** (les images réelles ne
sont pas séparables aussi trivialement que le jeu synthétique). Pour une analyse
médicale sérieuse, utilisez le connecteur MedGemma
(`eval/run_vlm_comparison.py --model google/medgemma-4b-it`).
