"""
Prediction Inference Module for Disease Risk Assessment.
Loads pre-trained pipelines, validates patient feature input, and returns risk assessments.
"""
import joblib
import pandas as pd
import numpy as np
from src.config import DATASETS_CONFIG, MODELS_DIR
from src.evaluate import extract_feature_importance
from src.utils import logger


def load_model_pipeline(dataset_key: str):
    """Load pre-trained scikit-learn pipeline for a specific dataset."""
    if dataset_key not in DATASETS_CONFIG:
        raise ValueError(f"Invalid dataset_key: {dataset_key}")
        
    model_path = DATASETS_CONFIG[dataset_key]["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run training pipeline first.")
        
    pipeline = joblib.load(model_path)
    return pipeline


def validate_and_format_input(dataset_key: str, input_dict: dict) -> pd.DataFrame:
    """Validate user input fields and format into a pandas DataFrame."""
    config = DATASETS_CONFIG[dataset_key]
    num_features = config["num_features"]
    cat_features = config["cat_features"]
    
    expected_cols = num_features + cat_features
    df_input = pd.DataFrame([input_dict])
    
    missing = [col for col in expected_cols if col not in df_input.columns]
    if missing:
        raise ValueError(f"Missing required input features for dataset '{dataset_key}': {missing}")
        
    # Reorder columns to match expected schema
    df_input = df_input[expected_cols]
    
    # Cast numerical columns
    for col in num_features:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")
        
    return df_input


def predict_disease_risk(dataset_key: str, input_dict: dict) -> dict:
    """
    Predict disease risk for a single patient sample.
    
    Parameters
    ----------
    dataset_key : str
        'heart', 'diabetes', or 'breast_cancer'
    input_dict : dict
        Mapping of feature names to user input values
        
    Returns
    -------
    dict
        Structured prediction output containing:
        - prediction (0 or 1)
        - prediction_label (human readable string)
        - probability (float between 0 and 1)
        - probability_percent (formatted string)
        - model_name (class name of classifier)
        - feature_importances (DataFrame of feature weights)
    """
    if dataset_key not in DATASETS_CONFIG:
        raise ValueError(f"Invalid dataset_key: {dataset_key}. Must be one of {list(DATASETS_CONFIG.keys())}")
        
    config = DATASETS_CONFIG[dataset_key]
    pipeline = load_model_pipeline(dataset_key)

    
    df_input = validate_and_format_input(dataset_key, input_dict)
    
    # Make prediction
    pred = pipeline.predict(df_input)[0]
    
    if hasattr(pipeline, "predict_proba"):
        prob = pipeline.predict_proba(df_input)[0, 1]
    else:
        prob = float(pred)
        
    target_labels = config["target_label_map"]
    label = target_labels.get(int(pred), "Unknown Risk Level")
    
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__
    
    # Feature importances
    imp_df = extract_feature_importance(pipeline, dataset_key)
    
    result = {
        "dataset_key": dataset_key,
        "dataset_name": config["name"],
        "prediction": int(pred),
        "prediction_label": label,
        "probability": float(prob),
        "probability_percent": f"{prob * 100:.1f}%",
        "model_name": classifier_name,
        "feature_importances": imp_df
    }
    return result
