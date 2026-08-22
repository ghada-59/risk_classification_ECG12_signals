"""Post-training evaluation: metrics, ROC/PR curves, confusion matrix, report.

Usage
-----
    python -m scripts.modelA.evaluate
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import CLASS_NAMES, N_CLASSES, REPORTS_DIR, VIS_DIR  # noqa: E402


logger = logging.getLogger("evaluate")


def per_class_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute one-vs-rest sensitivity, specificity, AUC for each class."""
    out: dict = {"per_class": {}, "global": {}}
    for k, name in enumerate(CLASS_NAMES):
        y_t = (y_true == k).astype(int)
        y_p = y_prob[:, k]
        y_hat = (y_p >= threshold).astype(int)
        tp = int(((y_hat == 1) & (y_t == 1)).sum())
        fp = int(((y_hat == 1) & (y_t == 0)).sum())
        fn = int(((y_hat == 0) & (y_t == 1)).sum())
        tn = int(((y_hat == 0) & (y_t == 0)).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        ppv = tp / max(1, tp + fp)
        f1 = (
            2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
        )
        try:
            auc_v = float(roc_auc_score(y_t, y_p))
        except ValueError:
            auc_v = float("nan")
        out["per_class"][name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv": round(ppv, 4),
            "f1": round(f1, 4),
            "auc": round(auc_v, 4),
            "support": int(y_t.sum()),
        }
    try:
        out["global"]["macro_auc"] = round(
            float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4
        )
        out["global"]["weighted_auc"] = round(
            float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")), 4
        )
    except ValueError:
        out["global"]["macro_auc"] = float("nan")
    y_pred = y_prob.argmax(axis=1)
    out["global"]["accuracy"] = round(float((y_pred == y_true).mean()), 4)
    return out


def bootstrap_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n: int = 200,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    aucs = {name: [] for name in CLASS_NAMES}
    n_samples = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt = y_true[idx]
        yp = y_prob[idx]
        for k, name in enumerate(CLASS_NAMES):
            yt_k = (yt == k).astype(int)
            if yt_k.sum() == 0 or yt_k.sum() == len(yt_k):
                continue
            try:
                aucs[name].append(float(roc_auc_score(yt_k, yp[:, k])))
            except ValueError:
                continue
    ci = {}
    for name, vals in aucs.items():
        if vals:
            ci[name] = (
                round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4),
            )
        else:
            ci[name] = (float("nan"), float("nan"))
    return ci


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path
) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion matrix (row-normalized)")
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, name in enumerate(CLASS_NAMES):
        y_t = (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(y_t, y_prob[:, k])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, name in enumerate(CLASS_NAMES):
        y_t = (y_true == k).astype(int)
        precision, recall, _ = precision_recall_curve(y_t, y_prob[:, k])
        ax.plot(recall, precision, label=name, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (one-vs-rest)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_risk_distribution(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path
) -> str:
    risk = y_prob[:, 2] + 0.5 * y_prob[:, 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, name in enumerate(CLASS_NAMES):
        sns.kdeplot(risk[y_true == k], label=name, ax=ax, fill=True, alpha=0.3)
    ax.set_xlabel("risk score = P(critique) + 0.5·P(suspect)")
    ax.set_ylabel("density")
    ax.set_title("Risk-score distribution by true class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def write_report(
    metrics: dict,
    ci: dict,
    figures: list[str],
    out_path: Path,
    training_summary_path: Path | None = None,
) -> str:
    training = {}
    if training_summary_path and Path(training_summary_path).exists():
        training = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))

    lines: list[str] = ["# Modèle A — Validation Report\n"]

    lines.append("## 1. Modèle et entraînement\n")
    if training:
        lines.append(f"- **Paramètres** : {training.get('n_params', '?'):,}")
        lines.append(f"- **Epochs entraînés** : {training.get('epochs_trained', '?')}")
        lines.append(f"- **Best val macro-AUC** : {training.get('best_val_auc', '?'):.4f}")
        lines.append(f"- **Checkpoint** : `{training.get('checkpoint', '')}`")
    lines.append("")

    lines.append("## 2. Performance globale (test fold 10)\n")
    g = metrics["global"]
    lines.append(
        f"- **Macro AUC-ROC** : {g.get('macro_auc', float('nan')):.4f}"
    )
    lines.append(
        f"- **Weighted AUC-ROC** : {g.get('weighted_auc', float('nan')):.4f}"
    )
    lines.append(f"- **Accuracy** : {g.get('accuracy', float('nan')):.4f}")
    lines.append("")

    lines.append("## 3. Métriques par classe (seuil 0.5)\n")
    header = (
        "| classe | support | sensibilité | spécificité | PPV | F1 | "
        "AUC | AUC IC95% | FP | FN |"
    )
    lines.append(header)
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|"
    )
    for name, m in metrics["per_class"].items():
        ci_lo, ci_hi = ci.get(name, (float("nan"), float("nan")))
        lines.append(
            f"| **{name}** | {m['support']} | {m['sensitivity']:.3f} | "
            f"{m['specificity']:.3f} | {m['ppv']:.3f} | {m['f1']:.3f} | "
            f"{m['auc']:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {m['fp']} | {m['fn']} |"
        )
    lines.append("")

    lines.append("## 4. Visualisations\n")
    for fig in figures:
        title = Path(fig).stem.replace("_", " ").title()
        rel = Path(fig).resolve().relative_to(out_path.parent.parent.parent.resolve())
        lines.append(f"### {title}\n")
        lines.append(f"![{title}](../../{rel.as_posix()})\n")
    lines.append("")

    lines.append("## 5. Conformité & limites\n")
    lines.extend([
        "- Système d'aide à la décision — usage académique uniquement.",
        "- Données PTB-XL anonymisées (RGPD compliant).",
        "- Le modèle classifie l'**état cardiaque sur un snapshot 10 s** ; "
        "ce n'est pas une prédiction de décompensation HF à 30 s–1 min "
        "(absence de labels temporels dans PTB-XL).",
        "- Le mapping SCP→classe est documenté dans `scripts/modelA/label_mapping.py` "
        "et reste discutable aux frontières.",
    ])
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    pred_path = REPORTS_DIR / "test_predictions.npz"
    if not pred_path.exists():
        logger.error("Predictions file missing: %s — run training first.", pred_path)
        return 2

    data = np.load(pred_path)
    y_true: np.ndarray = data["y_true"]
    y_prob: np.ndarray = data["y_prob"]
    y_pred = y_prob.argmax(axis=1)

    metrics = per_class_metrics(y_true, y_prob)
    ci = bootstrap_auc(y_true, y_prob, n=200)

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    figures = [
        plot_confusion(y_true, y_pred, VIS_DIR / "confusion_matrix.png"),
        plot_roc(y_true, y_prob, VIS_DIR / "roc_curves.png"),
        plot_pr(y_true, y_prob, VIS_DIR / "pr_curves.png"),
        plot_risk_distribution(y_true, y_prob, VIS_DIR / "risk_distribution.png"),
    ]

    out = write_report(
        metrics, ci, figures,
        REPORTS_DIR / "validation_report.md",
        REPORTS_DIR / "training_summary.json",
    )
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "auc_ci95": ci}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
