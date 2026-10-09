"""
Evaluation Module for Disease Prediction Models.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, and Feature Importances.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from src.config import DATASETS_CONFIG, REPORTS_DIR
from src.utils import (
    logger, plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance
)


def evaluate_pipeline(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a fitted pipeline on test dataset.
    
    Returns metrics dict including accuracy, precision, recall, f1, roc_auc, cm, fpr, tpr.
    """
    y_pred = pipeline.predict(X_test)
    
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
    except Exception:
        roc_auc = 0.5
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "Confusion Matrix": cm,
        "FPR": fpr,
        "TPR": tpr,
        "Y_Pred": y_pred,
        "Y_Proba": y_proba
    }


def get_transformed_feature_names(preprocessor, num_features: list, cat_features: list) -> list:
    """Extract feature names out from fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is unavailable
        names = list(num_features)
        if cat_features and hasattr(preprocessor, "named_transformers_"):
            cat_tr = preprocessor.named_transformers_.get("cat")
            if cat_tr and hasattr(cat_tr, "named_steps"):
                encoder = cat_tr.named_steps.get("encoder")
                if encoder and hasattr(encoder, "get_feature_names_out"):
                    cat_names = encoder.get_feature_names_out(cat_features)
                    names.extend(cat_names)
        return names


def extract_feature_importance(pipeline, dataset_key: str) -> pd.DataFrame:
    """
    Extract feature importances or coefficients from a fitted model pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    config = DATASETS_CONFIG[dataset_key]
    feature_names = get_transformed_feature_names(
        preprocessor, config["num_features"], config["cat_features"]
    )
    
    # Strip prefix added by ColumnTransformer (e.g., 'num__age' -> 'age')
    clean_names = [f.split("__")[-1] for f in feature_names]
    
    importances = None
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = classifier.coef_[0]
        
    if importances is None or len(importances) != len(clean_names):
        return pd.DataFrame()
        
    df_imp = pd.DataFrame({
        "Feature": clean_names,
        "Importance": importances
    })
    df_imp["AbsImportance"] = df_imp["Importance"].abs()
    df_imp = df_imp.sort_values(by="AbsImportance", ascending=False).drop(columns=["AbsImportance"])
    return df_imp


def generate_evaluation_summary(all_results: dict) -> pd.DataFrame:
    """
    Consolidate metrics from all datasets and models into a single DataFrame.
    """
    rows = []
    for dataset_key, models_dict in all_results.items():
        ds_name = DATASETS_CONFIG[dataset_key]["name"]
        for model_name, res in models_dict.items():
            rows.append({
                "Dataset": ds_name,
                "Model": model_name,
                "Accuracy": res["Accuracy"],
                "Precision": res["Precision"],
                "Recall": res["Recall"],
                "F1 Score": res["F1 Score"],
                "ROC-AUC": res["ROC-AUC"]
            })
            
    summary_df = pd.DataFrame(rows)
    # Save CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved model comparison table to {csv_path}")
    return summary_df


def save_markdown_report(summary_df: pd.DataFrame, best_models: dict):
    """Generate professional evaluation report in markdown format."""
    report_path = REPORTS_DIR / "evaluation_report.md"
    
    md_content = ["# Disease Prediction Model Evaluation Report\n\n"]
    md_content.append("## Executive Summary\n\n")
    md_content.append("This report summarizes the empirical performance of four Machine Learning models across three distinct medical datasets (Heart Disease, Diabetes, and Breast Cancer).\n\n")
    
    md_content.append("### Best Performing Models\n\n")
    for dataset_key, info in best_models.items():
        ds_name = DATASETS_CONFIG[dataset_key]["name"]
        md_content.append(f"- **{ds_name}**: `{info['model_name']}` (Selected based on {info['criterion']} = {info['score']:.4f})\n")
        
    md_content.append("\n## Comprehensive Model Performance Comparison Table\n\n")
    try:
        table_md = summary_df.to_markdown(index=False)
    except Exception:
        # Fallback manual table formatting
        header = "| " + " | ".join(summary_df.columns) + " |\n"
        separator = "| " + " | ".join(["---"] * len(summary_df.columns)) + " |\n"
        rows_str = ""
        for _, row in summary_df.iterrows():
            rows_str += "| " + " | ".join([str(val) for val in row.values]) + " |\n"
        table_md = header + separator + rows_str

    md_content.append(table_md)
    
    md_content.append("\n\n## Medical Domain Metric Interpretations\n\n")
    md_content.append("- **Recall (Sensitivity)**: Critical for disease screening. High recall minimizes **False Negatives** (failing to identify a patient with a condition).\n")
    md_content.append("- **Precision**: Minimizes **False Positives** (unnecessary anxiety and follow-up medical testing).\n")
    md_content.append("- **F1-Score & ROC-AUC**: Measures balanced diagnostic capability across varied risk thresholds.\n")
    
    md_content.append("\n## Disclaimer\n\n")
    md_content.append("> **Important Educational Disclaimer**: This evaluation is conducted for educational machine learning demonstration purposes only. Models must NOT be used for clinical diagnosis.\n")
    
    with open(report_path, "w") as f:
        f.writelines(md_content)
        
    logger.info(f"Saved evaluation markdown report to {report_path}")

