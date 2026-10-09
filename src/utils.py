"""
Utility functions for plotting, metrics export, and directory helpers.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.config import FIGURES_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DiseasePrediction")

# Set global matplotlib style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid", palette="muted")


def plot_target_distribution(df: pd.DataFrame, target_col: str, title: str, filename: str):
    """Plot target variable distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts()
    percentage = df[target_col].value_counts(normalize=True) * 100
    
    bars = sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_title(f"Target Distribution: {title}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Target Class", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    
    for i, (count, pct) in enumerate(zip(counts.values, percentage.values)):
        ax.text(i, count / 2, f"{count}\n({pct:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        
    plt.tight_layout()
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list, title: str, filename: str):
    """Plot correlation heatmap for numerical features."""
    if len(num_cols) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols].corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title(f"Feature Correlation Heatmap: {title}", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")


def plot_confusion_matrices(cm_dict: dict, dataset_name: str, filename: str):
    """Plot side-by-side confusion matrices for all models on a dataset."""
    num_models = len(cm_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 3.5))
    if num_models == 1:
        axes = [axes]
        
    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        
    fig.suptitle(f"Confusion Matrices - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")


def plot_roc_curves(roc_data_dict: dict, dataset_name: str, filename: str):
    """Plot ROC curves for all models on a single plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, (fpr, tpr, roc_auc) in roc_data_dict.items():
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
        
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(f"ROC Curves - {dataset_name}", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")


def plot_feature_importance(importance_df: pd.DataFrame, title: str, filename: str, top_n: int = 15):
    """Plot bar chart of top feature importances or coefficients."""
    fig, ax = plt.subplots(figsize=(9, 6))
    top_df = importance_df.head(top_n).sort_values(by="Importance", ascending=True)
    
    colors = ["#3498db" if val >= 0 else "#e74c3c" for val in top_df["Importance"]]
    ax.barh(top_df["Feature"], top_df["Importance"], color=colors)
    ax.set_title(f"Top {top_n} Feature Importance / Impact: {title}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance / Coefficient Magnitude", fontsize=11)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")


def plot_model_comparison_chart(results_df: pd.DataFrame, filename: str = "overall_model_comparison.png"):
    """Plot metric comparison across datasets and models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Melt dataframe for grouped bar chart
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    melted = pd.melt(results_df, id_vars=["Dataset", "Model"], value_vars=metrics_to_plot,
                     var_name="Metric", value_name="Score")
    
    sns.barplot(data=melted, x="Dataset", y="Score", hue="Model", ax=ax, palette="Set2")
    ax.set_title("Model Performance Comparison Across Datasets", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylim([0.5, 1.05])
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="ML Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")
