# Mini-rapport — Assistant radiologue virtuel responsable

> **Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.**
>
> EFREI — Solution Delivery, filière Data — 2025-2026

Ce rapport documente le prototype livré : objectif, périmètre, données, stratégie
baseline, amélioration mesurée, métriques, limites, risques, précautions éthiques
et dépendances. Il s'appuie sur des résultats **reproductibles** (voir §10 Preuves).

---

## 1. Objectif

Construire une chaîne d'analyse **prudente, traçable et évaluée** autour d'une
radiographie thoracique frontale. Le système reçoit une image et retourne une
sortie JSON structurée (classe, confiance, observations, justification, limites,
avertissement). Le but n'est **pas** le diagnostic mais la démonstration d'une
démarche d'ingénierie responsable : périmètre restreint, baseline reproductible,
garde-fous, métriques, journalisation et analyse d'erreurs.

## 2. Périmètre

- **Entrée** : une radiographie thoracique frontale (image PNG/JPG).
- **Sorties** : `normal`, `suspected_opacity`, `uncertain`.
- La classe `uncertain` est un **garde-fou méthodologique** : savoir ne pas
  conclure sur une image ambiguë ou de mauvaise qualité fait partie de la qualité
  attendue.
- **Hors périmètre** : diagnostic, tri, orientation patient, données réelles
  identifiantes, promesse de performance clinique.

## 3. Données

Jeu **synthétique jouet** : 30 images 512×512 (`data/sample_images/`) décrites
par `data/synthetic_cases.csv` (colonnes `case_id, image_path, source, label,
split, quality, notes`). Répartition équilibrée : 10 `normal`, 10
`suspected_opacity`, 10 `uncertain`. Splits `smoke` (20) et `final` (10).

Ces images **imitent grossièrement** une radiographie uniquement pour valider les
flux logiciels (chargement, inférence, JSON, logs, métriques, garde-fous). Un
score parfait sur ce jeu **ne constitue pas** une performance médicale. Pour un
vrai projet : RSNA Pneumonia, CheXpert, MIMIC-CXR ou NIH ChestX-ray, sous réserve
de licence et de conditions d'accès (voir §9).

Le signal exploité est réel et interprétable : les trois classes se distinguent
par le **contraste global** (écart-type des intensités) et la **taille d'une zone
focale brillante** (proxy d'opacité) :

| Classe | Contraste (std) | Pic lumineux (p99) | Zone brillante |
|---|---|---|---|
| `normal` | ~16.9 | ~64 | ~0.37 % |
| `suspected_opacity` | ~19.1 | ~86 | 1.2–2.5 % |
| `uncertain` (qualité limitée) | ~10.5 | ~44 | ~0.07 % |

## 4. Stratégie baseline

`toy_predict(mode="baseline")` (`src/inference.py`) est un **classifieur d'image
transparent** (pas un modèle appris) : il lit des statistiques de l'image (via
`src/preprocessing.py:extract_features`) et applique une règle unique — si la
fraction de pixels très brillants dépasse `OPACITY_BRIGHT_FRACTION` (0.8 %), la
classe est `suspected_opacity`, sinon `normal`.

**Limite volontaire de la baseline** : elle n'a **aucune notion de qualité
d'image** et **n'abstient jamais**. C'est exactement le comportement risqué que
l'amélioration corrige.

> Choix de conception : la baseline ne « triche » pas en lisant le label dans le
> nom de fichier. Elle décide à partir des pixels, ce qui rend l'évaluation et
> l'analyse d'erreurs honnêtes.

Un **connecteur VLM réel** (MedGemma via `transformers`) est disponible en option
(`medgemma_predict`), non requis pour la démo : il nécessite `HF_TOKEN`, un accès
réseau et un téléchargement de plusieurs Go. Il réutilise les prompts versionnés
`prompts/baseline_prompt.txt` et `prompts/improved_prompt.txt`.

## 5. Amélioration (règle d'incertitude explicite)

`toy_predict(mode="improved")` ajoute deux garde-fous :

1. **Qualité image** : si le contraste est faible (`limited`/`poor`), on bascule
   vers `uncertain` (on ne conclut pas sur une image peu lisible).
2. **Seuil de confiance** : si la confiance < **0.60**, on bascule vers
   `uncertain` (règle explicitement écrite dans `prompts/improved_prompt.txt`).

Les garde-fous transverses (`src/guardrails.py`) s'appliquent aux deux modes :
validation du schéma JSON, `warning` toujours présent, repli vers `uncertain` si
sortie invalide.

## 6. Métriques (résultats reproductibles)

Évaluation sur les 30 cas (`python eval/run_evaluation.py --mode toy`) :

| Métrique | Baseline | Amélioration |
|---|---:|---:|
| Accuracy | **0.667** | **1.000** |
| Macro-F1 | 0.556 | 1.000 |
| Sensibilité (`suspected_opacity`) | 1.000 | 1.000 |
| Spécificité (`normal`) | 1.000 | 1.000 |
| Taux d'incertitude | 0.000 | 0.333 |
| JSON valide | 1.000 | 1.000 |
| Warning présent | 1.000 | 1.000 |
| Latence médiane | ~3 ms | ~3 ms |

**Lecture** : le gain ne se limite pas à un chiffre. La baseline classe
parfaitement `normal` et `suspected_opacity` mais **force les 10 images de qualité
limitée en `normal`** avec une confiance élevée (~0.84) — un comportement
faussement rassurant. La règle d'incertitude fait passer ces 10 cas en
`uncertain` (confiance plafonnée à 0.5), portant l'accuracy de 0.667 à 1.0 et le
macro-F1 de 0.556 à 1.0. L'amélioration corrige donc un **mode d'échec précis et
médicalement pertinent**, pas seulement un score.

### 6.1 Sensibilité au seuil (analyse d'ablation)

`eval/threshold_sweep.py` fait varier le seuil de contraste qui déclenche
l'abstention (mode amélioré). La courbe montre trois régimes :

| Seuil `limited_contrast_std` | Accuracy | Taux d'incertitude | Régime |
|---|---:|---:|---|
| < 10.5 | 0.667 | 0.00 | aucune abstention (= baseline) |
| **11 – 16** | **1.000** | **0.333** | **zone sûre (valeur retenue : 13)** |
| ≥ 18 | 0.667 → 0.333 | 0.67 → 1.0 | sur-abstention (utilité détruite) |

Le seuil retenu (13) est volontairement au centre de la plage sûre [10.6 ; 16.9].
Cela prouve que le garde-fou a un **coût** et que sa valeur est justifiée, pas
arbitraire.

### 6.2 Comparaison de prompts sur un vrai VLM + métrique d'hallucination

`eval/run_vlm_comparison.py` exécute un **vrai** modèle vision-langage avec le
`baseline_prompt` puis l'`improved_prompt`, applique les mêmes garde-fous et
mesure en plus un **taux de sur-affirmation** (`detect_overclaim` : présence de
pathologies nommées / langage définitif = proxy d'hallucination) et un taux de
JSON invalide.

Résultats obtenus avec un modèle **ouvert et accessible** (SmolVLM-256M, 9 cas),
utilisé comme **substitut de validation du harnais** — ce n'est pas un modèle
médical, sa faible exactitude est attendue et n'est pas le sujet :

| Métrique | baseline_prompt | improved_prompt |
|---|---:|---:|
| Accuracy | 0.333 | 0.333 |
| JSON valide | 1.000 | 0.889 |
| Warning présent | 1.000 | 1.000 |
| Taux de sur-affirmation | 0.000 | 0.111 |
| JSON invalide (→ `uncertain`) | 0.000 | 0.111 |
| Latence médiane | ~6.0 s | ~6.4 s |

**Ce que ça prouve** : le pipeline fonctionne sur une **vraie** sortie de modèle,
les garde-fous rattrapent les cas réels (JSON invalide → `uncertain`), le warning
est toujours présent et l'hallucination est détectée automatiquement. Un modèle
généraliste ne sait pas lire une radio (d'où l'exactitude faible) — c'est
précisément pourquoi le prototype reste prudent et non clinique.

**Modèle médical visé (MedGemma)** : le connecteur est câblé
(`--model google/medgemma-4b-it`). Il est *gated* : il faut accepter la licence
sur la page du modèle et fournir `HF_TOKEN`. Une fois l'accès accordé, la même
commande produit la comparaison sur le vrai modèle médical.

## 7. Analyse d'erreurs

`eval/error_register.csv` (généré par `eval/build_error_register.py`) contient les
**30 cas commentés** avec la taxonomie du projet :

| Code | Signification | Occurrences (baseline) |
|---|---|---:|
| OK | Classification correcte | 20 |
| FN | Faux négatif / rassurance abusive | 10 |
| FP | Faux positif | 0 |
| UA | Incertitude acceptable | 0 (baseline n'abstient jamais) |
| JF | Erreur de format JSON | 0 |
| HT | Hallucination textuelle | évaluée manuellement, non détectée |

Les 10 FN correspondent aux images `uncertain` (qualité limitée) que la baseline
appelle `normal`. Chaque ligne du registre indique la prédiction baseline, la
prédiction améliorée, la sévérité et l'action corrective. En mode amélioré, ces
10 cas deviennent `uncertain` → 0 erreur résiduelle sur ce jeu.

Une **analyse commentée à la main** des cas les plus instructifs (succès, échecs,
sur-abstention) est disponible dans [`docs/error_analysis.md`](error_analysis.md).

## 8. Limites

- Données **synthétiques** non représentatives d'une population clinique.
- Sur ce jeu, les classes sont nettement séparables → un score parfait en mode
  amélioré est **attendu** et ne préjuge pas d'une performance médicale.
- La **confiance est un proxy heuristique, non calibré**.
- Sensibilité aux seuils (`OPACITY_BRIGHT_FRACTION`, seuil de confiance 0.60) et,
  pour le connecteur réel, au modèle et au prompt.
- Risque d'**hallucination textuelle** avec un VLM réel (non présent dans le mode
  jouet dont les observations sont dérivées des features).

## 9. Risques et précautions éthiques

- **Ligne rouge** : aucun usage clinique, aucun tri, aucune orientation patient.
- **Avertissement obligatoire** présent dans la sortie JSON, l'interface web, le
  README et ce rapport.
- **Données** : uniquement synthétiques ou publiques autorisées et dé-identifiées.
  Aucune donnée patient réelle, même pseudonymisée, sans autorisation traçable.
- **Journalisation** : chaque prédiction (image, modèle, prompt, classe,
  confiance, latence) est tracée en SQLite (`runs`) pour auditabilité.
- Détails complets dans [`docs/ethique_et_limites.md`](ethique_et_limites.md).

## 10. Dépendances et licences

- Code du dépôt : licence **MIT** (voir `LICENSE`).
- Dépendances Python : voir `requirements.txt` / `requirements-test.txt`. Le mode
  jouet n'utilise que `numpy`, `pillow`, `pandas` (+ `fastapi`/`streamlit` pour
  les interfaces). Le connecteur réel ajoute `transformers`, `torch`, `accelerate`.
- Datasets/modèles externes (**licences propres, à documenter avant tout usage**) :
  MedGemma (`google/medgemma-4b-*`), Gemma 4 / Unsloth, MIMIC-CXR, CheXpert. Voir
  le tableau des références dans le [README](../README.md) et `docs/appel_offre.md`.
- Variable d'environnement : `HF_TOKEN` (jamais commitée ; voir `.env.example`).

## 11. Preuves (commandes reproductibles)

```bash
# 1. Tests de fumée (structure, schéma, garde-fous, API, éval, robustesse)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q            # 14 tests

# 2. Évaluation baseline vs amélioration (écrit eval/results/ + logs SQLite)
python eval/run_evaluation.py --mode toy

# 3. Registre d'erreurs commenté (30 cas)
python eval/build_error_register.py

# 4. Analyse de sensibilité au seuil
python eval/threshold_sweep.py

# 5. Comparaison de prompts sur un vrai VLM (modèle ouvert accessible)
python eval/run_vlm_comparison.py --model HuggingFaceTB/SmolVLM-256M-Instruct --limit 9
#   Modèle médical visé (accès gated + HF_TOKEN) :
#   HF_TOKEN=... python eval/run_vlm_comparison.py --model google/medgemma-4b-it

# 6. (Optionnel) Évaluer sur un échantillon réel autorisé — voir data/real/README.md
python scripts/prepare_real_dataset.py --images data/real/images --labels data/real/labels.csv
python eval/run_evaluation.py --mode toy --cases data/real/real_cases.csv

# 7. Démo web (une interface au choix)
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload   # puis POST /predict, GET /history
```

Sorties de preuve : `eval/results/before_after_summary.csv`,
`eval/results/{baseline,improved}_metrics.json`, `eval/results/*_confusion.csv`,
`eval/results/threshold_sweep.csv`, `eval/results/vlm_before_after_summary.csv`,
`eval/error_register.csv`, table `runs` de `medical_ai_evidence.sqlite`.
