"""
Training Pipeline for Disease Prediction ML System.
Trains, hyperparameter tunes, evaluates, and serializes models for Heart Disease, Diabetes, and Breast Cancer.
"""
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.config import DATASETS_CONFIG, MODEL_NAMES, HYPERPARAMETER_GRIDS, RANDOM_STATE
from src.data_loader import load_dataset, get_dataset_summary
from src.preprocessing import get_train_test_data, build_model_pipeline
from src.evaluate import (
    evaluate_pipeline, extract_feature_importance, generate_evaluation_summary,
    save_markdown_report
)
from src.utils import (
    logger, plot_target_distribution, plot_correlation_heatmap,
    plot_confusion_matrices, plot_roc_curves, plot_feature_importance,
    plot_model_comparison_chart
)


def tune_model(dataset_key: str, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Perform 5-Fold GridSearchCV hyperparameter tuning on a model pipeline.
    """
    pipeline = build_model_pipeline(dataset_key, model_name)
    param_grid = HYPERPARAMETER_GRIDS.get(model_name, {})
    
    logger.info(f"[{dataset_key}] Tuning {model_name} with GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    logger.info(f"[{dataset_key}] {model_name} Best CV F1 Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_all():
    """
    Execute end-to-end training workflow across all datasets and models.
    """
    all_dataset_results = {}
    best_models_info = {}
    
    for dataset_key in DATASETS_CONFIG:
        config = DATASETS_CONFIG[dataset_key]
        dataset_name = config["name"]
        logger.info(f"\n==========================================")
        logger.info(f"Processing Dataset: {dataset_name} ({dataset_key})")
        logger.info(f"==========================================")
        
        # 1. Load Data
        df = load_dataset(dataset_key)
        
        # 2. Generate Exploratory Plots
        plot_target_distribution(df, config["target_col"], dataset_name, f"{dataset_key}_target_distribution.png")
        plot_correlation_heatmap(df, config["num_features"], dataset_name, f"{dataset_key}_correlation_heatmap.png")
        
        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = get_train_test_data(df, dataset_key)
        
        # 4. Train & Tune Models
        fitted_pipelines = {}
        dataset_eval_results = {}
        cm_dict = {}
        roc_dict = {}
        
        for model_name in MODEL_NAMES:
            best_pipeline, best_params, cv_score = tune_model(dataset_key, model_name, X_train, y_train)
            fitted_pipelines[model_name] = best_pipeline
            
            # Evaluate on Test Set
            eval_res = evaluate_pipeline(best_pipeline, X_test, y_test)
            eval_res["best_params"] = best_params
            eval_res["cv_f1_score"] = cv_score
            dataset_eval_results[model_name] = eval_res
            
            cm_dict[model_name] = eval_res["Confusion Matrix"]
            roc_dict[model_name] = (eval_res["FPR"], eval_res["TPR"], eval_res["ROC-AUC"])
            
            logger.info(f"[{dataset_key} | {model_name}] Test Acc: {eval_res['Accuracy']:.4f} | Recall: {eval_res['Recall']:.4f} | F1: {eval_res['F1 Score']:.4f} | ROC-AUC: {eval_res['ROC-AUC']:.4f}")
            
        all_dataset_results[dataset_key] = dataset_eval_results
        
        # 5. Generate Dataset Evaluation Plots
        plot_confusion_matrices(cm_dict, dataset_name, f"{dataset_key}_confusion_matrices.png")
        plot_roc_curves(roc_dict, dataset_name, f"{dataset_key}_roc_curves.png")
        
        # 6. Select Best Model (Primary criteria: F1 Score & ROC-AUC balanced for medical risk)
        best_model_name = max(
            dataset_eval_results.keys(),
            key=lambda m: (dataset_eval_results[m]["F1 Score"] + dataset_eval_results[m]["ROC-AUC"]) / 2
        )
        best_pipeline = fitted_pipelines[best_model_name]
        best_res = dataset_eval_results[best_model_name]
        
        logger.info(f"🏆 Best Model for {dataset_name}: {best_model_name} (F1: {best_res['F1 Score']:.4f}, ROC-AUC: {best_res['ROC-AUC']:.4f})")
        
        # Save Best Pipeline
        model_save_path = config["model_path"]
        joblib.dump(best_pipeline, model_save_path)
        logger.info(f"Saved fitted best model pipeline to {model_save_path}")
        
        # Also save feature importance plot for best model
        imp_df = extract_feature_importance(best_pipeline, dataset_key)
        if not imp_df.empty:
            plot_feature_importance(imp_df, f"{dataset_name} ({best_model_name})", f"{dataset_key}_feature_importance.png")
            
        best_models_info[dataset_key] = {
            "model_name": best_model_name,
            "criterion": "Composite F1 & ROC-AUC",
            "score": (best_res["F1 Score"] + best_res["ROC-AUC"]) / 2
        }
        
    # 7. Overall Reports and Metrics Summary
    summary_df = generate_evaluation_summary(all_dataset_results)
    save_markdown_report(summary_df, best_models_info)
    plot_model_comparison_chart(summary_df, "overall_model_comparison.png")
    
    logger.info("\n🎉 Training pipeline completed successfully across all datasets!")
    return summary_df, best_models_info


if __name__ == "__main__":
    train_all()
