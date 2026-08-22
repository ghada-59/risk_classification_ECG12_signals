# Modèle A — Model Card

## Identification

- **Nom** : Modèle A — Classifieur d'état cardiaque ECG 12 leads
- **Version** : 0.1.0
- **Date** : 25 mai 2026
- **Auteurs** : projet académique Ghada-AI
- **Licence** : usage académique uniquement

## Objectif clinique

Classer une fenêtre ECG 12 leads de 10 s en trois états — `normal`,
`suspect`, `critique` — avec un score de risque continu. Système d'**aide
à la décision** ; ne remplace pas l'avis d'un cardiologue.

**Limite explicite assumée** : ce n'est **pas** une prédiction de
décompensation HF à 30 s – 1 min (PTB-XL ne contient pas d'événements
temporels). Le modèle évalue l'état cardiaque sur un snapshot 10 s.

## Données d'entraînement

- **Dataset** : [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
  (21 799 ECG, 18 869 patients, 1989–2001).
- **Résolution** : 100 Hz, 12 leads, 10 s par enregistrement.
- **Anonymisation** : ECG entièrement anonymisés à la source (RGPD).
- **Labels** : mapping déterministe SCP-ECG → 3 classes, défini dans
  [scripts/modelA/label_mapping.py](../../scripts/modelA/label_mapping.py).

## Architecture

- **Backbone CNN1D** : 3 blocs `Conv1d (k=7→5→3) + BN + ReLU + MaxPool/2`
  (canaux 12 → 64 → 128 → 256)
- **Tête séquentielle** : BiLSTM 128 cachées, 1 couche
- **Branche auxiliaire** : FC(12 → 32) sur features HRV/morpho (neurokit2)
- **Tête finale** : Concat(256 + 32) → FC(128) → Dropout(0.3) → FC(3)
- **Paramètres entraînables** : 578 659

## Prétraitement

- Bandpass Butterworth 0.5–40 Hz (ordre 4, filtfilt zero-phase)
- Notch 50 Hz (secteur Europe)
- Z-score par lead, clip ±5σ

## Entraînement

- **Splits** : strat_fold PTB-XL — folds 1–8 train, 9 val, 10 test (patient-indépendant)
- **Tailles** : train 17 418 / val 2 183 / test 2 198
- **Loss** : Focal loss γ=2 + class weights inverse-fréquence
- **Optimizer** : AdamW, lr=1e-3, weight_decay=1e-4
- **Scheduler** : CosineAnnealingLR (T_max=30)
- **Batch** : 64, seed 42
- **Augmentations train** : time shift ±50 samples, lead dropout p=0.1
- **Arrêt** : early stopping patience=6 — arrêté à l'epoch 17, best epoch 11
- **Durée totale** : ~60 minutes CPU (Intel 8 cores)

## Résultats — test fold 10 (n=2 198)

| classe       | support | sens. | spéc. | PPV  | F1   | AUC  | AUC IC95%        | FP  | FN  |
|--------------|--------:|------:|------:|-----:|-----:|-----:|:----------------:|----:|----:|
| **normal**   |     738 | 0.900 | 0.716 | 0.615| 0.731| 0.884| [0.872, 0.899]  | 415 |  74 |
| **suspect**  |    1200 | 0.466 | 0.971 | 0.951| 0.625| 0.855| [0.840, 0.870]  |  29 | 641 |
| **critique** |     260 | 0.789 | 0.957 | 0.709| 0.747| 0.955| [0.943, 0.965]  |  84 |  55 |

- **Macro AUC-ROC** : **0.898** (cible ≥ 0.85 ✓)
- **Weighted AUC-ROC** : 0.877
- **Accuracy** : 0.686

## Latence d'inférence (CPU)

100 inférences synthétiques `(12, 1000)` sur CPU :

| percentile | latence (ms) |
|------------|-------------:|
| p50        |        177.3 |
| p95        |        227.5 |
| p99        |        263.5 |
| max        |        270.4 |

**Cible** p95 < 3000 ms — **atteint largement** (p95 = 227 ms, soit 13× sous la cible).

## Analyse des erreurs

- **`critique`** : très bonne séparation (AUC 0.955). Sensibilité 0.79 légèrement
  sous la cible interne de 0.80 — la majorité des manqués proviennent de cas
  avec `STACH` ou tachycardies frontières au seuil 0.5. Abaisser le seuil
  améliore la sensibilité au prix de la PPV.
- **`suspect`** : sensibilité la plus basse (0.47). C'est une classe
  hétérogène ("substrat HF") qui inclut blocs, hypertrophies, vieux MI ;
  le modèle a tendance à reclasser ces cas en `normal` à seuil 0.5
  (spécificité 0.97 — peu de faux positifs).
- **`normal`** : sensibilité 0.90 mais PPV 0.62 — la classe sur-prédite,
  attendu compte tenu des poids de classes et du déséquilibre.

## Utilisation prévue

- Recherche académique, prototypage, démonstrations cliniques (dashboard
  web) avec affichage explicite de la classe + score de risque.

## Utilisation NON prévue

- Diagnostic clinique direct sans supervision médicale.
- Décision de triage automatique.
- Suivi temps réel hors snapshot 10 s.
- Détection de décompensation HF à horizon 30 s – 1 min (limitation
  fondamentale des données d'entraînement).

## Reproductibilité

- Seed déterministe (42) sur Python / NumPy / PyTorch.
- Hyperparams figés dans [scripts/modelA/config.py](../../scripts/modelA/config.py).
- Splits déterministes (PTB-XL `strat_fold`).
- Tous les artefacts (checkpoint, cache, prédictions) sont produits
  par `python -m scripts.modelA.train` puis `python -m scripts.modelA.evaluate`.

## Conformité

- Données PTB-XL anonymisées conformément RGPD.
- Aucune donnée personnelle traitée ou stockée par le système.
- API stateless (pas de log des inférences côté serveur).
