# Assistant radiologue virtuel responsable

> **Auteur :** Badr Tajini  
> **Solution Delivery - Filière Data**  
> **École :** EFREI  
> **Année académique :** 2025-2026

> ### 🚀 Pour installer et lancer le projet : **[→ Guide de démarrage (QUICKSTART.md)](QUICKSTART.md)**
> Guide pas à pas pour n'importe quelle machine (Windows / macOS / Linux), sans prérequis.

## Contexte

Prototype pédagogique d'IA médicale multimodale pour apprendre à construire une chaîne **prudente, traçable et évaluée** autour d'une radiographie thoracique frontale.

---

>  **Position non clinique.** Ce dépôt n'est pas un dispositif médical. Il ne doit jamais être utilisé pour diagnostiquer, trier ou orienter un patient. Toute sortie doit rester un résultat expérimental, vérifié par un professionnel qualifié.

---

## Contrat du projet

| Élément | Cadrage |
|---|---|
| Entrée | Une radiographie thoracique frontale |
| Sorties | `normal`, `suspected_opacity`, `uncertain` |
| Preuve minimale | JSON valide, warning, logs, métriques, cas d'erreur |
| Données | Synthétiques ou publiques, autorisées et dé-identifiées |
| Finalité | Prototype éducatif de data/IA, pas aide au diagnostic réelle |

Le bon rendu ne cherche pas à impressionner par un modèle spectaculaire. Il démontre une méthode : périmètre limité, baseline reproductible, garde-fous, évaluation, analyse d'erreurs et limites explicites.

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python eval/run_evaluation.py --mode toy
streamlit run app/streamlit_app.py
```

## Exécution du prototype livré (V1)

Le **mode jouet est le mode par défaut** : c'est un classifieur d'image
transparent (statistiques de contraste et de zone brillante), **sans GPU, sans
réseau, sans token**. Il valide toute la chaîne (JSON, garde-fous, logs,
métriques).

```bash
# Environnement
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt

# Démo web (au choix)
streamlit run app/streamlit_app.py            # UI + onglet "Tableau de bord & logs"
uvicorn api.main:app --reload                 # API : POST /predict, GET /history

# Évaluation baseline vs amélioration -> eval/results/ + logs SQLite
python eval/run_evaluation.py --mode toy

# Registre d'erreurs commenté (30 cas) -> eval/error_register.csv
python eval/build_error_register.py

# Analyse de sensibilité au seuil de qualité
python eval/threshold_sweep.py

# Comparaison de prompts sur un VRAI VLM (modèle ouvert accessible, GPU conseillé)
python eval/run_vlm_comparison.py --model HuggingFaceTB/SmolVLM-256M-Instruct --limit 9
```

**Preuves générées** : `eval/results/before_after_summary.csv`,
`eval/results/*_metrics.json`, `eval/results/*_confusion.csv`,
`eval/error_register.csv`, table `runs` de `medical_ai_evidence.sqlite`.
Synthèse chiffrée et analyse : [`docs/rapport.md`](docs/rapport.md).

### Modèle réel MedGemma (optionnel, niveau Could)

Le connecteur `src.inference.vlm_predict` appelle un vrai VLM via `transformers`.
Il n'est **pas** utilisé par la démo par défaut (téléchargement de plusieurs Go,
GPU recommandé). Il lit le token depuis l'environnement — jamais en dur :

```bash
export HF_TOKEN="<votre-token-huggingface>"   # Windows: $env:HF_TOKEN="..."
python eval/run_vlm_comparison.py --model google/medgemma-4b-it
```

MedGemma est **gated** : il faut d'abord accepter la licence sur
<https://huggingface.co/google/medgemma-4b-it> (approbation automatique), sinon le
téléchargement renvoie une erreur 403. Un modèle **ouvert** (`HuggingFaceTB/SmolVLM-256M-Instruct`)
permet de valider le harnais sans accès *gated*. Pour évaluer sur de vraies
radios, voir [`data/real/README.md`](data/real/README.md). Variables
d'environnement : [`.env.example`](.env.example).

## Smoke test du dépôt

Avant une soutenance, un push ou une livraison, lancer le contrôle court :

```bash
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src api app eval finetuning tests
python eval/run_evaluation.py --mode toy \
  --out-dir /tmp/assistant-radio-eval \
  --db-path /tmp/assistant-radio-evidence.sqlite
```

Ce smoke test vérifie la structure du dépôt, le contrat du dataset synthétique, le schéma de sortie, les garde-fous, l'API de démonstration, la compilation Python et l'évaluation jouet.

## API de démonstration

```bash
uvicorn api.main:app --reload
```

Exemple :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```

La réponse doit contenir une classe, une confiance, des observations visuelles, une justification, des limites et l'avertissement non clinique.

## Organisation

```text
assistant-radiologue-virtuel/
├── README.md
├── docs/          # appel d'offre, architecture, éthique, évaluation
├── data/          # cas synthétiques et images jouet
├── prompts/       # prompt baseline, prompt amélioré, schéma JSON
├── src/           # inférence jouet, garde-fous, métriques, SQLite
├── api/           # FastAPI
├── app/           # Streamlit / Gradio
├── eval/          # évaluation, sorties CSV/JSON, registre d'erreurs
├── tests/         # smoke tests et contrat minimal
├── notebooks/     # notebooks de démarrage
└── finetuning/    # stubs expérimentaux, non obligatoires
```

## Livrables attendus

> Niveaux de réalisation attendus : **Must / Should / Could**

Cette section sert à clarifier ce qui est attendu des étudiants selon trois niveaux de maturité : `Must have`, `Should have` et `Could have`. Le projet doit rester centré sur un prototype pédagogique d’IA médicale, non clinique, limité à une radiographie thoracique frontale et à trois classes :

* `normal` : aucun signe évident d’opacité sur l’image analysée ;
* `suspected_opacity` : présence possible d’une opacité ou d’un signal visuel suspect ;
* `uncertain` : qualité d’image insuffisante, signes ambigus ou confiance trop faible pour conclure.

La classe `uncertain` est volontairement conservée. Elle ne doit pas être considérée comme un échec du modèle, mais comme un garde-fou méthodologique. Dans un contexte médical, savoir ne pas conclure lorsqu’une image est ambiguë fait partie de la qualité attendue.

### Niveau 1 (Must have) : socle minimal obligatoire

Le niveau Must correspond au minimum attendu pour considérer le projet comme fonctionnel, compréhensible et soutenable.

À ce niveau, les étudiants doivent livrer une première chaîne complète, même simple. L’objectif n’est pas encore d’obtenir le meilleur score possible, mais de prouver que le système fonctionne de bout en bout : une image entre, une analyse est produite, la sortie est structurée, les résultats sont sauvegardés, et les limites sont clairement affichées.

**Fonctionnalités attendues :**

* mettre en place un dépôt Git propre, avec une structure claire et documentée ;
* préparer un petit jeu de données initial, synthétique ou public autorisé ;
* utiliser environ 20 images pour vérifier la chaîne complète ;
* construire une baseline reproductible par prompting ;
* accepter une radiographie thoracique frontale en entrée ;
* produire une sortie JSON structurée ;
* afficher une classe parmi `normal`, `suspected_opacity` ou `uncertain` ;
* fournir un score de confiance ;
* produire une justification courte, fondée uniquement sur les éléments visibles ;
* ajouter un avertissement non clinique obligatoire ;
* journaliser les résultats : image, prompt, modèle, prédiction, latence, sortie JSON ;
* fournir une première interface web simple avec Streamlit, Gradio ou FastAPI ;
* présenter une première évaluation simple avec quelques exemples de réussite et d’échec.

Champs minimaux attendus dans la sortie JSON :

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["observation courte"],
  "justification": "justification synthétique et prudente",
  "limitations": ["limite identifiée"],
  "warning": "Prototype pédagogique uniquement. Non destiné au diagnostic médical."
}
```

**Critères d’acceptation :**

* le projet peut être installé et lancé à partir du README ;
* une image peut être envoyée dans l’application ;
* une réponse JSON est produite ;
* le warning médical est toujours présent ;
* les résultats sont sauvegardés ;
* le groupe peut expliquer la baseline, ses limites et ses premières erreurs.

Un projet qui atteint correctement ce niveau est déjà un projet recevable, à condition qu’il soit clair, reproductible et honnête sur ses limites.

**Diagramme :** 
```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MUST HAVE - SOCLE MINIMAL                            │
└──────────────────────────────────────────────────────────────────────────────┘

      ┌──────────────────────┐
      │  Dépôt Git propre     │
      │  README, docs, src,   │
      │  app, data, prompts   │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Jeu initial          │
      │  20 images environ    │
      │  synthétiques ou      │
      │  publiques autorisées │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Entrée image         │
      │  radiographie         │
      │  thoracique frontale  │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Baseline prompting   │
      │  1 modèle             │
      │  1 prompt stable      │
      │  inférence rejouable  │
      └──────────┬───────────┘
                 │
                 ▼
      ┌────────────────────────────────────────────────────────────┐
      │                    Sortie JSON structurée                   │
      │                                                            │
      │  {                                                         │
      │    "image_quality": "...",                                 │
      │    "predicted_class": "normal | suspected_opacity |         │
      │                        uncertain",                         │
      │    "confidence": 0.0,                                      │
      │    "visual_evidence": ["observations visibles"],           │
      │    "justification": "courte et prudente",                  │
      │    "limitations": ["limites connues"],                     │
      │    "warning": "prototype pédagogique, non clinique"         │
      │  }                                                         │
      └──────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Interface web simple │
      │  Streamlit / Gradio   │
      │  ou FastAPI minimal   │
      └──────────┬───────────┘
                 │
                 ▼
      ┌────────────────────────────────────────────────────────────┐
      │                    Journalisation obligatoire               │
      │  image | prompt | modèle | prédiction | latence | JSON      │
      └──────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Première évaluation  │
      │  réussites, échecs,   │
      │  limites observées    │
      └──────────────────────┘
```

### Niveau 2 (Should have) : amélioration mesurée et analyse critique

Le niveau Should correspond à un projet plus solide, capable de démontrer une véritable progression d’ingénierie.

À ce niveau, les étudiants ne doivent pas seulement montrer que le système fonctionne. Ils doivent montrer qu’ils savent l’évaluer, l’améliorer et interpréter ses erreurs. L’amélioration doit être mesurée, et non simplement déclarée.

**Améliorations attendues :**

* comparer au moins deux versions du système :

  * une baseline par prompt simple ;
  * une version améliorée par prompt renforcé, garde-fou, ensemble de prompts ou règle d’incertitude ;
* tester plusieurs prompts sur le même ensemble d’images ;
* ajouter une règle d’incertitude explicite ;
* refuser ou basculer vers `uncertain` si le JSON est invalide, si la confiance est faible ou si les signes visuels sont ambigus ;
* calculer des métriques adaptées :

  * accuracy ;
  * macro-F1 ;
  * sensibilité ;
  * spécificité ;
  * validité JSON ;
  * latence ;
  * taux d’incertitude ;
  * taux d’hallucination ou de justification non fondée ;
* construire un tableau de bord simple de suivi des résultats ;
* documenter 20 à 30 cas finaux commentés ;
* créer un registre d’erreurs ;
* distinguer les faux positifs, faux négatifs, incertitudes acceptables, erreurs de format et hallucinations textuelles ;
* rédiger un mini-rapport expliquant les choix de dataset, les prompts, les métriques, les erreurs et les limites.

**Critères d’acceptation :**

* la comparaison baseline vs amélioration est chiffrée ;
* les erreurs sont classées et commentées ;
* les étudiants peuvent expliquer pourquoi une amélioration est conservée ou rejetée ;
* le rapport ne montre pas seulement les réussites ;
* la soutenance présente des cas de succès, des cas d’échec et des cas incertains ;
* la prudence médicale reste visible dans l’interface, dans les sorties et dans le discours.

Le niveau Should est le niveau recommandé pour une bonne soutenance. Il montre que le groupe ne s’est pas limité à une démonstration technique, mais qu’il a construit une démarche expérimentale.

**Diagramme :** 

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SHOULD HAVE - AMÉLIORATION MESURÉE                       │
└──────────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────────────────┐
                     │  Même ensemble d'images   │
                     │  données fixes, labels,   │
                     │  cas de test identiques   │
                     └────────────┬─────────────┘
                                  │
             ┌────────────────────┴────────────────────┐
             │                                         │
             ▼                                         ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│ Version A — Baseline      │              │ Version B — Améliorée     │
│ prompt simple             │              │ prompt renforcé           │
│ sortie JSON               │              │ garde-fous                │
│ warning obligatoire       │              │ ensemble de prompts       │
└────────────┬─────────────┘              │ règle d'incertitude       │
             │                            └────────────┬─────────────┘
             │                                         │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Contrôles de cohérence       │
                   │  - JSON valide ?              │
                   │  - confiance suffisante ?     │
                   │  - signes visuels clairs ?    │
                   │  - justification fondée ?     │
                   └──────────────┬───────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
┌──────────────────────┐                         ┌──────────────────────┐
│ Sortie acceptée       │                         │ Bascule vers          │
│ normal ou             │                         │ uncertain             │
│ suspected_opacity     │                         │ si doute ou anomalie  │
│ si cohérente          │                         │ de format             │
└──────────┬───────────┘                         └──────────┬───────────┘
           │                                                │
           └──────────────────────┬─────────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │              Métriques                │
                │ accuracy | macro-F1 | sensibilité     │
                │ spécificité | JSON valide | latence   │
                │ incertitude | hallucination           │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │        Tableau de bord simple         │
                │ comparaison baseline vs improved      │
                │ graphiques, scores, exemples          │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │        Registre d'erreurs             │
                │ faux positifs | faux négatifs         │
                │ incertitudes acceptables              │
                │ erreurs de format | hallucinations    │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │      Mini-rapport critique            │
                │ dataset, prompts, métriques,          │
                │ erreurs, limites, décisions           │
                └──────────────────────────────────────┘
```

### Niveau 3 (Could have) : extensions avancées

Le niveau Could correspond aux extensions ambitieuses. Elles ne doivent être abordées qu’après validation du niveau Must et du niveau Should.

Une extension avancée ne doit jamais remplacer la baseline. Elle doit venir après une première chaîne stable, des données propres, des métriques disponibles et une analyse d’erreurs déjà engagée.

**Extensions possibles :**

* expérimenter une adaptation LoRA ou QLoRA sur un modèle multimodal ;
* tester Gemma 4 avec Unsloth pour une expérimentation rapide ;
* tester MedGemma via Hugging Face, PEFT ou QLoRA, sous réserve des conditions d’accès et d’usage ;
* ajouter un classifieur léger CNN ou ViT pour produire un score de confiance plus contrôlable ;
* comparer plusieurs familles de modèles ;
* ajouter une localisation visuelle approximative d’une zone suspecte si les données le permettent ;
* enrichir la base SQLite avec les cas, prompts, runs, évaluations et erreurs ;
* ajouter des tests automatiques pour vérifier la validité JSON, la présence du warning et la cohérence des sorties ;
* construire une API FastAPI plus complète autour de l’endpoint `/predict` ;
* améliorer l’interface avec un dashboard interactif ;
* réaliser une ablation systématique des prompts ;
* analyser l’impact de différents seuils de confiance.

**Critères d’acceptation :**

* l’extension est justifiée par une hypothèse claire ;
* le gain est mesuré ;
* la comparaison avec la baseline est conservée ;
* les conditions d’usage des modèles et datasets sont citées ;
* les coûts et limites sont documentés ;
* le système reste explicitement non clinique ;
* aucune sortie ne doit être présentée comme un diagnostic.

Un bonus technique n’est pertinent que s’il améliore la compréhension, la robustesse ou la qualité de l’évaluation. Un fine-tuning non mesuré, une interface trop complexe ou une extension non reliée aux métriques ne doivent pas être considérés comme une amélioration réelle.

**Diagramme :**

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    COULD HAVE - EXTENSIONS AVANCÉES                          │
└──────────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────────────┐
                           │  Base validée         │
                           │  Must + Should        │
                           │  déjà fonctionnels    │
                           └──────────┬───────────┘
                                      │
                                      ▼
          ┌──────────────────────────────────────────────────────┐
          │              Choix d'une extension contrôlée          │
          └──────────────────────────────────────────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
┌──────────────────────┐   ┌──────────────────────┐      ┌──────────────────────┐
│ Fine-tuning léger     │   │ Score plus contrôlé   │      │ Meilleure API         │
│ LoRA / QLoRA          │   │ CNN ou ViT léger      │      │ FastAPI /predict      │
│ Gemma 4 + Unsloth     │   │ calibration           │      │ validation entrée     │
│ MedGemma + PEFT       │   │ seuils de confiance   │      │ contrats de sortie    │
└──────────┬───────────┘   └──────────┬───────────┘      └──────────┬───────────┘
           │                          │                             │
           ▼                          ▼                             ▼
┌──────────────────────┐   ┌──────────────────────┐      ┌──────────────────────┐
│ Comparaison modèles   │   │ Localisation visuelle │      │ Dashboard avancé      │
│ MedGemma, Gemma 4,    │   │ zone suspecte         │      │ filtres, graphiques,  │
│ VLM fallback, CNN/ViT │   │ si labels disponibles │      │ suivi des erreurs     │
└──────────┬───────────┘   └──────────┬───────────┘      └──────────┬───────────┘
           │                          │                             │
           └──────────────┬───────────┴──────────────┬──────────────┘
                          │                          │
                          ▼                          ▼
            ┌──────────────────────────┐   ┌──────────────────────────┐
            │ SQLite enrichi            │   │ Tests automatiques        │
            │ cases, prompts, runs,     │   │ JSON valide               │
            │ évaluations, erreurs      │   │ warning présent           │
            │ versions modèles          │   │ classes autorisées        │
            └────────────┬─────────────┘   │ cohérence des sorties     │
                         │                 └────────────┬─────────────┘
                         │                              │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────┐
                   │  Analyse avancée                  │
                   │  ablation de prompts              │
                   │  impact des seuils de confiance   │
                   │  gains, coûts, limites            │
                   └──────────────────────────────────┘
```

#### Synthèse des attentes

| Niveau      | Intention pédagogique                           | Résultat attendu                                              |
| ----------- | ----------------------------------------------- | ------------------------------------------------------------- |
| Must have   | Construire une chaîne fonctionnelle et prudente | Baseline, JSON, warning, logs, interface simple               |
| Should have | Mesurer, comparer et analyser                   | Baseline vs amélioration, métriques, erreurs commentées       |
| Could have  | Explorer une extension avancée                  | LoRA, MedGemma, classifieur, localisation ou dashboard avancé |

La réussite du projet ne dépend pas du nombre de fonctionnalités ajoutées. Elle dépend de la capacité du groupe à défendre une progression claire : un périmètre maîtrisé, une baseline reproductible, une amélioration mesurée, des erreurs documentées et des limites assumées.



## Références techniques

Les pistes avancées doivent rester expérimentales, traçables et justifiées. En particulier, un groupe qui mobilise Gemma, MedGemma, Unsloth, MIMIC-CXR ou CheXpert doit citer la source exacte, la version, les conditions d'accès et les limites d'usage.

| Ressource | Usage possible | Référence à citer |
|---|---|---|
| Unsloth - Gemma 4 | Fine-tuning LoRA/QLoRA expérimental, uniquement après une baseline simple | [Guide Gemma 4](https://unsloth.ai/docs/models/gemma-4/train), [catalogue des modèles](https://unsloth.ai/docs/get-started/unsloth-model-catalog), [blog Unsloth](https://unsloth.ai/blog) |
| MedGemma | Baseline ou adaptation médicale image-texte, avec prudence sur les conditions d'accès | [Model card Hugging Face](https://huggingface.co/google/medgemma-4b-pt) |
| MIMIC-CXR / MIMIC-CXR-JPG | Jeu de données de radiographies thoraciques, accès contrôlé et non redistribuable | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) |
| CheXpert | Jeu de données public de radiographies thoraciques avec rapports associés | [Stanford AIMI - CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) |

## Points de vigilance

- Ne pas inventer d'information clinique absente de l'image.
- Ne pas supprimer la classe `uncertain`; elle est un garde-fou, pas un échec.
- Ne pas afficher uniquement des réussites en soutenance.
- Ne jamais commiter de données patient réelles, identifiantes ou ambiguës.
- Ne pas présenter le prototype comme validé médicalement.

## Licence et sources externes

Le code pédagogique du dépôt est publié sous licence MIT. 

**Les datasets externes, modèles et bibliothèques utilisés conservent leurs licences propres** : les étudiants doivent vérifier et documenter les droits d'usage avant toute expérimentation.

> **Exigence minimale :** indiquer dans le rapport la source, la version, la licence ou les conditions d'accès, les restrictions de redistribution, les traitements d'anonymisation et les limites d'interprétation. Aucun fichier patient réel, même pseudonymisé, ne doit être ajouté au dépôt sans autorisation explicite et traçable.
