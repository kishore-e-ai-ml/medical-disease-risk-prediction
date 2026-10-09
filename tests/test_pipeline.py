"""
Automated Pytest Suite for Disease Prediction ML Pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from src.config import DATASETS_CONFIG
from src.data_loader import load_dataset
from src.preprocessing import get_train_test_data, build_model_pipeline
from src.predict import predict_disease_risk, load_model_pipeline


def test_data_loader():
    """Verify loading of Heart Disease, Diabetes, and Breast Cancer datasets."""
    for key in ["heart", "diabetes", "breast_cancer"]:
        df = load_dataset(key)
        assert isinstance(df, pd.DataFrame), f"Dataset '{key}' is not a pandas DataFrame"
        assert not df.empty, f"Dataset '{key}' is empty"
        target_col = DATASETS_CONFIG[key]["target_col"]
        assert target_col in df.columns, f"Target column '{target_col}' missing from dataset '{key}'"


def test_preprocessing_splits():
    """Verify train/test data splitting and column transformers."""
    for key in ["heart", "diabetes", "breast_cancer"]:
        df = load_dataset(key)
        X_train, X_test, y_train, y_test = get_train_test_data(df, key, test_size=0.2, random_state=42)
        
        assert len(X_train) > 0 and len(X_test) > 0
        assert len(X_train) + len(X_test) <= len(df)
        assert DATASETS_CONFIG[key]["target_col"] not in X_train.columns


def test_pipeline_construction():
    """Test sklearn model pipeline construction."""
    for key in ["heart", "diabetes", "breast_cancer"]:
        pipeline = build_model_pipeline(key, "Logistic Regression")
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")


def test_saved_model_loading():
    """Test loading trained model pipelines from disk."""
    for key in ["heart", "diabetes", "breast_cancer"]:
        pipeline = load_model_pipeline(key)
        assert pipeline is not None
        assert hasattr(pipeline, "predict")


def test_heart_prediction_function():
    """Test inference prediction for Heart Disease sample."""
    sample_patient = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0, "oldpeak": 1.4,
        "slope": 1, "ca": 0, "thal": 2
    }
    result = predict_disease_risk("heart", sample_patient)
    assert result["dataset_key"] == "heart"
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0
    assert "%" in result["probability_percent"]
    assert isinstance(result["prediction_label"], str)


def test_diabetes_prediction_function():
    """Test inference prediction for Diabetes sample."""
    sample_patient = {
        "Pregnancies": 3, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
        "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    result = predict_disease_risk("diabetes", sample_patient)
    assert result["dataset_key"] == "diabetes"
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0


def test_breast_cancer_prediction_function():
    """Test inference prediction for Breast Cancer sample."""
    sample_patient = {
        "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0, "mean smoothness": 0.1184,
        "mean compactness": 0.2776, "mean concavity": 0.3001, "mean concave points": 0.1471, "mean symmetry": 0.2419, "mean fractal dimension": 0.07871,
        "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006399,
        "compactness error": 0.04904, "concavity error": 0.05373, "concave points error": 0.01587, "symmetry error": 0.03003, "fractal dimension error": 0.006193,
        "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0, "worst smoothness": 0.1622,
        "worst compactness": 0.6656, "worst concavity": 0.7119, "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
    }
    result = predict_disease_risk("breast_cancer", sample_patient)
    assert result["dataset_key"] == "breast_cancer"
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0


def test_invalid_input_handling():
    """Verify graceful handling of invalid dataset key or missing features."""
    with pytest.raises(ValueError):
        predict_disease_risk("invalid_key", {})
        
    with pytest.raises(ValueError):
        predict_disease_risk("heart", {"age": 50})  # Missing required columns
