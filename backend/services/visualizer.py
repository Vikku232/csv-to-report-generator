# backend/services/visualizer.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import uuid

def _savefig(fig, tmp_dir: Path) -> Path:
    path = tmp_dir / f"{uuid.uuid4().hex}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path

def plot_numeric_summary_heatmaps(df: pd.DataFrame, numeric_cols: list[str], tmp_dir: Path, cols_per_page: int = 5) -> list[Path]:
    paths = []
    if not numeric_cols:
        return paths

    stats = df[numeric_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats["missing"] = df[numeric_cols].isnull().sum()
    stats = stats.T

    for i in range(0, len(numeric_cols), cols_per_page):
        chunk_cols = numeric_cols[i:i + cols_per_page]
        chunk_data = stats[chunk_cols]
        page_num = (i // cols_per_page) + 1

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        sns.heatmap(chunk_data, annot=True, fmt=".2g", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
        ax.set_title(f"Descriptive Statistics (Numeric) - page {page_num}", fontsize=12, pad=10)
        ax.set_xlabel("column", fontsize=10)
        paths.append(_savefig(fig, tmp_dir))
    return paths

def plot_correlation_heatmap(corr: pd.DataFrame, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap", fontsize=13, pad=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    return _savefig(fig, tmp_dir)

def plot_target_distribution(value_counts: pd.Series, target_column: str, tmp_dir: Path) -> Path:
    value_counts = value_counts.sort_values(ascending=False)
    if len(value_counts) > 20:
        top_values = value_counts.head(20).copy()
        top_values["Other"] = value_counts.iloc[20:].sum()
        value_counts = top_values

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
    sns.barplot(x=value_counts.index.astype(str), y=value_counts.to_numpy(), ax=ax, color="#3182bd")
    ax.set_title(f"Distribution of {target_column}", fontsize=12)
    ax.set_xlabel(target_column, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    return _savefig(fig, tmp_dir)

def plot_kde_grid(df: pd.DataFrame, numeric_cols: list[str], tmp_dir: Path, cols_per_page: int = 6) -> list[Path]:
    paths = []
    for i in range(0, len(numeric_cols), cols_per_page):
        chunk = numeric_cols[i:i + cols_per_page]
        n_rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(8.5, 3.2 * n_rows))
        axes = np.array(axes).flatten()

        for ax, col in zip(axes, chunk):
            sns.kdeplot(df[col].dropna(), fill=True, color="#3182bd", ax=ax)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)

        for ax in axes[len(chunk):]:
            ax.axis("off")

        page_idx = (i // cols_per_page) + 1
        fig.suptitle(f"Feature vs Target (page {page_idx})\nKDE/Scatter Plots", fontsize=12, y=1.02)
        plt.tight_layout()
        paths.append(_savefig(fig, tmp_dir))
    return paths

def plot_outlier_proportion(outlier_df: pd.DataFrame, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(outlier_df) * 0.22)))
    y = np.arange(len(outlier_df))
    height = 0.38
    ax.barh(y - height/2, outlier_df["pct_outliers_3sigma"], height=height, label=r"3$\sigma$", color="#3182bd")
    ax.barh(y + height/2, outlier_df["pct_outliers_iqr"], height=height, label="IQR", color="#fc8d62")
    ax.set_yticks(y)
    ax.set_yticklabels(outlier_df["column"], fontsize=8)
    ax.set_xlabel("% Outliers", fontsize=10)
    ax.set_title(r"Outlier Proportion per Column (3$\sigma$ vs IQR)", fontsize=12)
    ax.legend(loc="upper right")
    ax.invert_yaxis()
    return _savefig(fig, tmp_dir)

def plot_duplicate_distribution(dup_df: pd.DataFrame, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(dup_df) * 0.22)))
    colors = sns.color_palette("Blues_r", len(dup_df))
    sns.barplot(y="column", x="duplicate_value_count", data=dup_df, ax=ax, palette=colors)
    ax.set_title("Duplicate Distribution per Column", fontsize=12)
    ax.set_xlabel("Count of Duplicate Values", fontsize=10)
    ax.set_ylabel("Column", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return _savefig(fig, tmp_dir)

def plot_feature_importance(fi_df: pd.DataFrame, target_name: str, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(fi_df) * 0.28)))
    colors = sns.color_palette("Blues_r", len(fi_df))
    sns.barplot(y="feature", x="importance", data=fi_df, ax=ax, palette=colors)
    ax.set_title(f"Top Feature Importance for {target_name}", fontsize=12)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return _savefig(fig, tmp_dir)

def plot_pca_clusters(pca_result: dict, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = ["#66c2a5", "#fc8d62"]
    sns.scatterplot(
        x=pca_result["pc1"], y=pca_result["pc2"],
        hue=[f"Cluster {l}" for l in pca_result["labels"]],
        palette=palette, alpha=0.8, s=25, ax=ax
    )
    ax.set_title(f"KMeans Clustering (k=2, silhouette={pca_result['silhouette']})", fontsize=12)
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.legend(loc="upper right")
    return _savefig(fig, tmp_dir)

def plot_predicted_vs_actual(y_test, y_proba_or_pred, task_type: str, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    if task_type == "classification":
        sns.stripplot(x=y_test, y=y_proba_or_pred, jitter=0.2, alpha=0.6, color="#3182bd", ax=ax)
        ax.set_title("Predicted Probability vs Actual Class", fontsize=12)
        ax.set_ylabel("Predicted Probability (Class=1)", fontsize=10)
        ax.set_xlabel("Actual Class", fontsize=10)
    else:
        ax.scatter(y_test, y_proba_or_pred, alpha=0.6, color="#3182bd")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
        ax.set_title("Predicted vs Actual", fontsize=12)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
    return _savefig(fig, tmp_dir)

def plot_model_summary_and_roc(model_result: dict, task_type: str, tmp_dir: Path) -> Path:
    fig, (ax_tbl, ax_plot) = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1, 1.2]})

    ax_tbl.axis("off")
    if task_type == "classification":
        table_data = [
            ["train accuracy", f"{model_result.get('train_accuracy', 0):.4f}"],
            ["test accuracy", f"{model_result.get('test_accuracy', 0):.4f}"],
            ["roc auc", f"{model_result.get('roc_auc', 0):.4f}" if "roc_auc" in model_result else "N/A"],
        ]
    else:
        table_data = [
            ["R2 score", f"{model_result.get('r2', 0):.4f}"],
            ["RMSE", f"{model_result.get('rmse', 0):.4f}"],
        ]

    table = ax_tbl.table(
        cellText=table_data,
        colLabels=["metric", "value"],
        loc="center",
        cellLoc="left"
    )
    table.scale(1, 1.5)
    ax_tbl.set_title("Model Summary", fontsize=11, pad=12)

    if task_type == "classification" and "fpr" in model_result:
        ax_plot.plot(model_result["fpr"], model_result["tpr"], color="#fc8d62", label=f"AUC = {model_result['roc_auc']:.4f}")
        ax_plot.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_plot.set_xlabel("FPR", fontsize=9)
        ax_plot.set_ylabel("TPR", fontsize=9)
        ax_plot.set_title("ROC Curve", fontsize=11)
        ax_plot.legend(loc="lower right")
    elif task_type == "classification":
        ax_plot.text(0.5, 0.5, "ROC curve unavailable\nfor multiclass data", ha="center", va="center")
        ax_plot.set_title("ROC Curve", fontsize=11)
        ax_plot.axis("off")
    else:
        residuals = model_result["y_test"] - model_result["y_pred"]
        ax_plot.scatter(model_result["y_pred"], residuals, alpha=0.7, color="#3182bd")
        ax_plot.axhline(0, linestyle="--", color="gray")
        ax_plot.set_xlabel("Predicted", fontsize=9)
        ax_plot.set_ylabel("Residual", fontsize=9)
        ax_plot.set_title("Residual Plot", fontsize=11)

    plt.tight_layout()
    return _savefig(fig, tmp_dir)

def plot_roc_curve(model_result: dict, tmp_dir: Path) -> Path | None:
    if "fpr" not in model_result or "tpr" not in model_result:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(model_result["fpr"], model_result["tpr"], color="#fc8d62", linewidth=2,
            label=f"AUC = {model_result['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12)
    ax.legend(loc="lower right")
    return _savefig(fig, tmp_dir)