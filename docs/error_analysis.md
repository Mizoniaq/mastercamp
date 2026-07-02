# Analyse d'erreurs commentée

> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.

Ce document accompagne le registre automatique [`eval/error_register.csv`](../eval/error_register.csv)
(30 cas, taxonomie FN/FP/UA/JF/HT). Il détaille à la main les cas les plus
instructifs — succès, échecs et incertitudes — comme l'exige la règle de
soutenance « ne jamais montrer seulement des réussites ».

Rappel des features utilisées : `contrast` (écart-type des intensités) et
`bright_fraction` (taille d'une zone focale brillante, proxy d'opacité).

## Vue d'ensemble (mode baseline)

| Type | Cas | Lecture |
|---|---:|---|
| OK (correct) | 20 | `normal` et `suspected_opacity` bien classés |
| FN (faux négatif) | 10 | images `uncertain` (qualité limitée) forcées en `normal` |
| FP | 0 | aucune sur-détection d'opacité sur ce jeu |
| UA | 0 | la baseline n'abstient jamais (limite voulue) |
| JF / HT | 0 | JSON toujours valide ; pas d'hallucination en mode jouet |

Toutes les FN de la baseline sont corrigées en `uncertain` par la règle
d'incertitude (mode amélioré) → 0 erreur résiduelle sur ce jeu.

## Cas commentés

### CXR_SYN_003 — `uncertain` · le cas le plus important
- **Baseline** : `normal` avec **confiance 0.835**. **Amélioration** : `uncertain` (0.5).
- **Type** : FN, sévérité haute.
- **Pourquoi** : l'image a un contraste faible (~10.5) et aucune zone brillante
  (bright_fraction ~0.07 %). La baseline n'a pas de notion de qualité : elle voit
  « pas d'opacité » et conclut `normal`, **avec une confiance élevée**. C'est le
  comportement le plus dangereux d'un point de vue médical : rassurer sur une
  image en réalité illisible.
- **Correction** : la règle qualité (contraste < 13) bascule le cas en `uncertain`
  et plafonne la confiance à 0.5. C'est le cœur de la démonstration.

### CXR_SYN_002 — `suspected_opacity` · vrai positif
- **Baseline = Amélioration** : `suspected_opacity` (0.72).
- **Type** : OK.
- **Pourquoi** : zone brillante marquée (bright_fraction ~1.2 %, p99 ~86) au-dessus
  du seuil d'opacité (0.8 %). Les deux modes concordent ; la règle d'incertitude
  ne se déclenche pas car la qualité est bonne et la confiance ≥ 0.60.

### CXR_SYN_005 — `suspected_opacity` · confiance plus élevée
- **Baseline = Amélioration** : `suspected_opacity` (~0.85).
- **Type** : OK.
- **Pourquoi** : zone brillante plus étendue que 002 → confiance plus haute. Montre
  que la confiance (proxy heuristique) suit la force du signal, sans être calibrée.

### CXR_SYN_001 — `normal` · vrai négatif
- **Baseline = Amélioration** : `normal` (~0.78).
- **Type** : OK.
- **Pourquoi** : contraste correct (~16.9), pas de zone focale brillante
  (bright_fraction ~0.37 % < 0.8 %). Cas de contrôle : la règle qualité ne doit
  **pas** se déclencher ici, et c'est bien le cas (16.9 > 13).

### CXR_SYN_006 / 009 / 012 … — `uncertain` · même schéma que 003
- **Baseline** : `normal` (~0.83) · **Amélioration** : `uncertain` (0.5).
- **Type** : FN → corrigé.
- **Pourquoi** : ce sont des répliques du cas 003 (jeu synthétique déterministe).
  Elles confirment que l'échec baseline est **systématique** sur la mauvaise
  qualité, pas accidentel — donc réellement corrigé par une règle, pas par chance.

### Cas de sur-abstention (analyse de seuil, `eval/threshold_sweep.py`)
- Si on relève le seuil qualité au-delà de ~17, les images `normal` (contraste
  16.9) sont à leur tour marquées « limitées » et basculent en `uncertain` :
  l'accuracy retombe à 0.667 puis 0.333. C'est le **risque inverse** : trop
  d'abstention détruit l'utilité. Le seuil retenu (13) est volontairement au
  centre de la zone sûre [10.6 ; 16.9].

## Ce que l'analyse démontre
1. L'amélioration corrige un **mode d'échec précis et médicalement pertinent**
   (rassurance abusive sur image illisible), pas seulement un score.
2. L'échec baseline est **systématique et explicable** par une feature (contraste),
   donc l'amélioration est une vraie règle, pas un ajustement opportuniste.
3. Le garde-fou a un **coût** (analyse de seuil) : abstenir trop nuit aussi. Le
   choix du seuil est donc justifié, pas arbitraire.
