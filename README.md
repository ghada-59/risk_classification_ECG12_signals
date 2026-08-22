# risk_classification_ECG12_signals-main — PTB-XL EDA + Modèle A (ECG 12 leads)

Projet académique en deux parties :

1. **Pipeline EDA** — analyse exploratoire du métadonnées PTB-XL (`ptbxl_database.csv`).
2. **Modèle A** — classifieur CNN1D + BiLSTM sur signaux ECG 12 leads (10 s, 100 Hz), avec API FastAPI et dashboard web.

> **Usage** : aide à la décision uniquement. Ne remplace pas un avis médical. Données PTB-XL anonymisées (RGPD).

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Données utilisées](#2-données-utilisées)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation](#4-installation)
5. [Guide complet — étape par étape](#5-guide-complet--étape-par-étape)
   - [Étape 0 — EDA (métadonnées)](#étape-0--eda-métadonnées)
   - [Étape 1 — Télécharger les signaux WFDB](#étape-1--télécharger-les-signaux-wfdb)
   - [Étape 2 — Construire le cache](#étape-2--construire-le-cache)
   - [Étape 3 — Entraîner Modèle A](#étape-3--entraîner-modèle-a)
   - [Étape 4 — Évaluer le modèle](#étape-4--évaluer-le-modèle)
   - [Étape 5 — Benchmark latence](#étape-5--benchmark-latence)
   - [Étape 6 — API + dashboard](#étape-6--api--dashboard)
6. [Résultats obtenus](#6-résultats-obtenus)
7. [Modèle A — détails techniques](#7-modèle-a--détails-techniques)
8. [Dashboard web](#8-dashboard-web)
9. [Référence API](#9-référence-api)
10. [Limites et conformité](#10-limites-et-conformité)
11. [Dépannage](#11-dépannage)

---

## 1. Vue d'ensemble

| Pipeline | Entrée | Sortie principale |
|----------|--------|-------------------|
| **EDA** | `data/ptbxl_database.csv` | Rapport Markdown, graphiques PNG, CSV nettoyé |
| **Modèle A** | CSV + `data/ptbxl/records100/` (WFDB) | Checkpoint `.pt`, métriques, dashboard temps réel |

**Parcours minimal** (si le modèle est déjà entraîné) :

```powershell
cd D:\stage\risk_classification_ECG12_signals-main
.\.venv\Scripts\Activate.ps1
python -m scripts.modelA.api
# Ouvrir http://localhost:8000
```

**Parcours complet** (première fois) : étapes 0 → 6 ci-dessous (~2–3 h sur CPU 8 cœurs, dont ~25 min de cache + ~60 min d'entraînement).

---

## 2. Données utilisées

| Fichier / dossier | Taille | Rôle |
|-------------------|--------|------|
| `data/ptbxl_database.csv` | ~6,3 MB | Index PTB-XL : 21 799 enregistrements × 28 colonnes (âge, sexe, `scp_codes`, `strat_fold`, chemins WFDB, etc.) |
| `data/ptbxl/records100/` | ~2,5 GB | Signaux ECG 100 Hz, 12 leads, 10 s — fichiers `.hea` + `.dat` (21 799 records) |

**Labels Modèle A** : dérivés de la colonne `scp_codes` via un mapping 3 classes défini dans `scripts/modelA/label_mapping.py` (pas de label « décompensation HF » dans PTB-XL).

| Classe | Signification (résumé) |
|--------|------------------------|
| `normal` | Rythme sinusal / NORM sans code significatif |
| `suspect` | Substrat chronique, blocs, vieux MI, ectopies isolées |
| `critique` | Tachyarythmies, blocs AV haut grade, WPW, tachycardie soutenue |

**Score de risque** : `P(critique) + 0.5 × P(suspect)` (probabilités softmax).

---

## 3. Structure du projet

```
risk_classification_ECG12_signals-main/
├── data/
│   ├── ptbxl_database.csv          # Métadonnées PTB-XL (EDA + index Modèle A)
│   └── ptbxl/records100/           # Signaux WFDB 100 Hz (Modèle A)
├── scripts/
│   ├── config.py                   # Registre EDA (ptbxl_database)
│   ├── run_eda.py                  # Entrée EDA
│   ├── data_loader.py …            # Modules EDA
│   └── modelA/                     # Téléchargement, cache, train, API, …
├── webapp/                         # Dashboard (HTML/CSS/JS)
├── processed_data/                 # CSV nettoyés (EDA)
├── visualizations/
│   ├── ptbxl_database/             # Graphiques EDA
│   └── modelA/                     # ROC, matrice, PR, risque
├── reports/
│   ├── ptbxl_database/             # Rapport EDA
│   └── modelA/                     # Validation, métriques, model card
├── models/modelA/
│   ├── modelA_best.pt              # Checkpoint (~2,3 MB)
│   ├── signals_cache.npy           # Cache signaux (~1 GB)
│   └── features_cache.npz          # Cache features HRV
├── requirements.txt
└── README.md
```

---

## 4. Installation

**Prérequis** : Python 3.11+, ~5 Go d'espace disque (signaux + cache).

**Windows (PowerShell)** :

```powershell
cd D:\stage\risk_classification_ECG12_signals-main
.\.venv\Scripts\Activate.ps1
python -m scripts.modelA.api
python -m pip install -r requirements.txt
```

**Linux / macOS** :

```bash
cd /chemin/vers/risk_classification_ECG12_signals-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dépendances principales** : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `wfdb`, `torch`, `neurokit2`, `scikit-learn`, `fastapi`, `uvicorn`, `requests`, `tqdm`.

---

## 5. Guide complet — étape par étape

### Étape 0 — EDA (métadonnées)

**Commande** :

```powershell
python -m scripts.run_eda
# ou uniquement PTB-XL :
python -m scripts.run_eda ptbxl_database
```

**Ce que fait le pipeline** : chargement → nettoyage → statistiques → analyse des features → visualisations → rapport Markdown.

**Fichiers produits** :

| Sortie | Chemin |
|--------|--------|
| CSV nettoyé | `processed_data/ptbxl_database_cleaned.csv` |
| Rapport | `reports/ptbxl_database/ptbxl_database_eda_report.md` |
| Graphiques | `visualizations/ptbxl_database/*.png` |

**Graphiques EDA générés** (11 fichiers) :

- `missing_values.png` — valeurs manquantes par colonne
- `dtype_distribution.png` — types de colonnes
- `numeric_distributions.png` — distributions numériques
- `outliers_boxplot.png` — boxplots (z-score)
- `correlation_heatmap.png` — corrélations numériques
- `class_balance_sex.png`, `class_balance_heart_axis.png`
- `multilabel_top_scp_codes.png` — codes SCP les plus fréquents
- `timeseries_recording_date.png`, `timeseries_year_recording_date.png`
- `age_sex_pyramid.png`

**Faits clés (rapport EDA)** :

| Indicateur | Valeur |
|------------|--------|
| Lignes × colonnes | 21 799 × 28 |
| Taille fichier source | 6,29 MB |
| Cellules manquantes | 33,99 % (surtout annotations qualité signal) |
| Doublons de lignes | 0 |
| `scp_codes` | présent sur 100 % des lignes (colonne utilisée pour les labels Modèle A) |
| Âge moyen | ~62,8 ans (écart-type ~32) |

**Durée typique** : quelques minutes.

---

### Étape 1 — Télécharger les signaux WFDB

**Commande** :

```powershell
python -m scripts.modelA.download_ptbxl              # 21 799 records (~10–15 min, mirror S3)
python -m scripts.modelA.download_ptbxl --limit 500  # sous-ensemble pour tests
python -m scripts.modelA.download_ptbxl --workers 24 --source s3
```

**Résultat attendu** : `data/ptbxl/records100/XXXXX/XXXXX_lr.hea` et `.dat` pour chaque enregistrement.

| Paramètre | Détail |
|-----------|--------|
| Fréquence | 100 Hz |
| Durée | 10 s → 1 000 échantillons par lead |
| Leads | 12 (I, II, III, aVR, aVL, aVF, V1–V6) |
| Mirror par défaut | AWS S3 PhysioNet (rapide, sans compte) |

**Échecs** : journalisés dans `data/ptbxl/download_failures.log`.

---

### Étape 2 — Construire le cache

**Commande** :

```powershell
python -m scripts.modelA.build_cache --workers 6 --save-every 2000
```

**Ce que fait le script** : pour chaque record, lecture WFDB → prétraitement (bandpass, notch 50 Hz, z-score, clip) → extraction HRV/morpho (neurokit2, lead II) → sauvegarde.

| Fichier cache | Taille | Contenu |
|---------------|--------|---------|
| `models/modelA/signals_cache.npy` | ~1,0 GB | `(21799, 12, 1000)` float32 |
| `models/modelA/features_cache.npz` | ~0,7 MB | `(21799, 12)` features + `ecg_ids` |

**Durée typique** : ~26 min (6 workers, 21 799 records). **Reprise** : si interrompu, relancer la même commande — sauvegarde partielle tous les 2 000 records.

---

### Étape 3 — Entraîner Modèle A

**Commande** :

```powershell
python -m scripts.modelA.train                    # entraînement complet
python -m scripts.modelA.train --epochs 5           # court
python -m scripts.modelA.train --max-samples 2000   # smoke test
```

**Split patient-indépendant** (`strat_fold` PTB-XL) :

| Jeu | Folds | Enregistrements |
|-----|-------|-----------------|
| Train | 1–8 | 17 418 |
| Validation | 9 | 2 183 |
| Test | 10 | 2 198 |

**Hyperparamètres** (défaut, `scripts/modelA/config.py`) :

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64 |
| Optimiseur | AdamW, lr = 1e-3, weight decay = 1e-4 |
| Loss | Focal loss (γ = 2) + poids inverse-fréquence |
| Scheduler | CosineAnnealingLR (T_max = 30) |
| Early stopping | patience = 6 (sur val macro-AUC) |
| Augmentations (train) | time shift ±50 samples, lead dropout p = 0,1 |
| Seed | 42 |

**Fichiers produits** :

| Fichier | Description |
|---------|-------------|
| `models/modelA/modelA_best.pt` | Meilleur checkpoint (max val macro-AUC) |
| `reports/modelA/training_log.csv` | Loss / AUC par epoch |
| `reports/modelA/training_summary.json` | Résumé final |
| `reports/modelA/test_predictions.npz` | Probas et labels sur le test |

**Entraînement réalisé sur ce dépôt** :

| Métrique | Valeur |
|----------|--------|
| Epochs jusqu'au meilleur checkpoint | **11** (early stop à l'epoch 17) |
| Meilleur val macro-AUC | **0,9107** |
| Durée par epoch | ~3–4 min (CPU 8 cœurs) |
| Durée totale entraînement | ~60 min |

**Courbe d'apprentissage (extrait `training_log.csv`)** :

| Epoch | train_loss | val_loss | val_auc |
|------:|-----------:|---------:|--------:|
| 1 | 0,270 | 0,701 | 0,857 |
| 5 | 0,177 | 0,613 | 0,891 |
| 8 | 0,158 | 0,591 | **0,903** |
| 11 | 0,149 | 0,595 | **0,911** ← meilleur checkpoint |

---

### Étape 4 — Évaluer le modèle

**Commande** :

```powershell
python -m scripts.modelA.evaluate
```

**Fichiers produits** :

| Fichier | Description |
|---------|-------------|
| `reports/modelA/validation_report.md` | Rapport complet |
| `reports/modelA/metrics.json` | Métriques + IC95 % bootstrap |
| `reports/modelA/model_card.md` | Fiche modèle (architecture, limites) |
| `visualizations/modelA/confusion_matrix.png` | Matrice de confusion |
| `visualizations/modelA/roc_curves.png` | Courbes ROC (one-vs-rest) |
| `visualizations/modelA/pr_curves.png` | Courbes précision-rappel |
| `visualizations/modelA/risk_distribution.png` | Distribution du score de risque |

*(Les métriques détaillées sont dans la [section 6](#6-résultats-obtenus).)*

---

### Étape 5 — Benchmark latence

**Commande** :

```powershell
python -m scripts.modelA.inference --benchmark 100 --output reports/modelA/latency_benchmark.json
```

**Résultats mesurés (CPU, 100 inférences synthétiques 12×1000)** :

| Percentile | Latence | Cible |
|------------|--------:|------:|
| p50 | 177 ms | — |
| p95 | **227 ms** | < 3 000 ms ✓ |
| p99 | 264 ms | — |
| moyenne | 140 ms | — |
| max | 270 ms | — |

> La première inférence sur un ECG réel via l'API peut être plus lente (~1–2 s) à cause du chargement neurokit2 ; les suivantes sont plus rapides.

---

### Étape 6 — API + dashboard

**Commande** :

```powershell
python -m scripts.modelA.api
# équivalent :
uvicorn scripts.modelA.api:app --host 0.0.0.0 --port 8000
```

**URL** : [http://localhost:8000](http://localhost:8000)

Le serveur sert à la fois l'API REST et le dashboard (`webapp/`).

---

## 6. Résultats obtenus

### 6.1 Performance globale (test fold 10, n = 2 198)

| Métrique | Valeur | Objectif interne |
|----------|-------:|-----------------:|
| **Macro AUC-ROC** | **0,898** | ≥ 0,85 ✓ |
| Weighted AUC-ROC | 0,877 | — |
| Accuracy (seuil 0,5) | 0,686 | — |
| Paramètres du modèle | 578 659 | — |
| Test loss | 0,622 | — |

### 6.2 Métriques par classe (seuil 0,5)

| Classe | Support | Sensibilité | Spécificité | PPV | F1 | AUC | IC95 % AUC | FP | FN |
|--------|--------:|------------:|------------:|----:|---:|----:|:----------|---:|---:|
| **normal** | 738 | 0,900 | 0,716 | 0,615 | 0,731 | 0,884 | [0,872 – 0,899] | 415 | 74 |
| **suspect** | 1 200 | 0,466 | 0,971 | 0,951 | 0,625 | 0,855 | [0,840 – 0,870] | 29 | 641 |
| **critique** | 260 | 0,789 | 0,957 | 0,709 | 0,747 | **0,955** | [0,943 – 0,965] | 84 | 55 |

**Lecture rapide** :

- **`critique`** : excellente séparation (AUC 0,955), peu de faux positifs (FP = 84), sensibilité légèrement sous 0,80.
- **`suspect`** : classe hétérogène — forte spécificité (0,97) mais beaucoup de faux négatifs (FN = 641) au seuil 0,5.
- **`normal`** : bonne sensibilité (0,90), PPV plus basse (sur-prédiction relative de `normal`).

### 6.3 Objectifs vs résultats

| Objectif | Cible | Résultat | Statut |
|----------|------:|---------:|:------:|
| Macro AUC test | ≥ 0,85 | 0,898 | ✓ |
| AUC classe critique | ≥ 0,90 | 0,955 | ✓ |
| Sensibilité critique | ≥ 0,80 | 0,789 | ~ |
| Spécificité critique | ≥ 0,90 | 0,957 | ✓ |
| Latence p95 inférence | < 3 s | 227 ms | ✓ |

### 6.4 Artefacts à consulter

| Document | Chemin |
|----------|--------|
| Rapport validation | `reports/modelA/validation_report.md` |
| Métriques JSON | `reports/modelA/metrics.json` |
| Model card | `reports/modelA/model_card.md` |
| Log entraînement | `reports/modelA/training_log.csv` |
| Benchmark latence | `reports/modelA/latency_benchmark.json` |
| Rapport EDA | `reports/ptbxl_database/ptbxl_database_eda_report.md` |

---

## 7. Modèle A — détails techniques

### 7.1 Chaîne de traitement

```
WFDB 12×1000 (100 Hz)
  → bandpass 0,5–40 Hz + notch 50 Hz + z-score par lead + clip ±5σ
  → features HRV/morpho (neurokit2, lead II) — 12 scalaires
  → CNN1D : 3 blocs Conv (k=7,5,3) → canaux 64→128→256 + MaxPool
  → BiLSTM bidirectionnel (hidden 128)
  → concat features aux (FC 12→32)
  → FC 128 + dropout 0,3 → softmax 3 classes
  → classe + score de risque
```

### 7.2 Architecture

| Composant | Détail |
|-----------|--------|
| Entrée signal | `(batch, 12, 1000)` |
| Entrée aux | `(batch, 12)` features HRV/morpho |
| CNN | 3 blocs, kernels 7 / 5 / 3, pool /2 |
| LSTM | 1 couche, 128 hidden, bidirectionnel |
| Tête | FC 128 → 3 classes |
| Paramètres | **578 659** |

### 7.3 Modules Python

| Module | Rôle |
|--------|------|
| `label_mapping.py` | SCP → normal / suspect / critique |
| `signal_preprocessing.py` | Filtres + normalisation |
| `feature_extraction.py` | QRS, HRV, morphologie |
| `dataset.py` | Dataset PyTorch + cache |
| `model.py` | Réseau CNN1D + BiLSTM |
| `train.py` | Boucle d'entraînement |
| `evaluate.py` | Métriques + graphiques |
| `inference.py` | `ECGPredictor` + benchmark |
| `api.py` | Serveur FastAPI |

---

## 8. Dashboard web

Interface servie sur `http://localhost:8000` (dossier `webapp/`).

| Fonction | Description |
|----------|-------------|
| Liste d'échantillons | Jusqu'à 100 ECG du fold test, filtres par classe |
| Visualisation 12 leads | Canvas HiDPI, grille type papier ECG |
| Analyse | Bouton ou **Espace** → appel `POST /predict` |
| Résultats | Classe, probabilités, score de risque, anneau de confiance, comparaison label réel |
| Historique | 12 dernières prédictions (localStorage) |
| Import fichier | JSON ou CSV `12×1000` (glisser-déposer) |
| Raccourcis | `←` `→` navigation, `R` rafraîchir, `Espace` prédire |
| Paramètres | URL API, analyse auto, grille ECG |

**État API** : pastille en haut à droite (`modèle prêt · cpu` si le checkpoint est chargé).

---

## 9. Référence API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/health` | Statut, device, leads, checkpoint |
| `GET` | `/samples?limit=N` | Liste d'ECG test (id, label réel) |
| `GET` | `/samples/{ecg_id}` | Signal 12×1000 + métadonnées |
| `POST` | `/predict` | Classification |

**Exemple `POST /predict`** :

```json
{
  "signal": [[0.1, 0.2, ...], ...],
  "sampling_rate": 100
}
```

**Réponse** :

```json
{
  "label_id": 1,
  "label": "suspect",
  "probabilities": {
    "normal": 0.342,
    "suspect": 0.648,
    "critique": 0.010
  },
  "risk_score": 0.334,
  "timestamp": "2026-05-25T17:26:39.612+00:00",
  "latency_ms": 170.2
}
```

---

## 10. Limites et conformité

| Point | Détail |
|-------|--------|
| **Aide à la décision** | Usage académique / recherche uniquement |
| **RGPD** | Données PTB-XL anonymisées à la source |
| **Horizon temporel** | Le modèle classifie un **snapshot 10 s** ; ce n'est **pas** une prédiction de décompensation HF à 30 s – 1 min (absent de PTB-XL) |
| **Labels** | Mapping SCP → 3 classes ; discutable aux frontières — voir `scripts/modelA/label_mapping.py` |
| **API** | Stateless : pas de journal des inférences côté serveur |
| **Classe suspect** | Sensibilité modérée au seuil 0,5 ; ajuster le seuil si besoin clinique |

---

## 11. Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | Activer le venv et `pip install -r requirements.txt` |
| API « modèle non entraîné » | Lancer `python -m scripts.modelA.train` ou vérifier `models/modelA/modelA_best.pt` |
| Cache manquant | `python -m scripts.modelA.build_cache` |
| Téléchargement WFDB lent / échecs | `--source s3` et moins de `--workers` si instable |
| `Connection refused` sur :8000 | Démarrer `python -m scripts.modelA.api` |
| Erreur prédiction « NaN » | Corrigé dans `feature_extraction.py` — relancer l'API |
| EDA : colonnes dict/list | Géré automatiquement dans `data_cleaner.py` |

---

## Ajouter un dataset EDA

Ajouter une entrée `DatasetConfig` dans `scripts/config.py`, puis :

```powershell
python -m scripts.run_eda mon_dataset
```

---

*Dernière mise à jour des résultats Modèle A : entraînement fold PTB-XL, checkpoint epoch 11, test fold 10.*
