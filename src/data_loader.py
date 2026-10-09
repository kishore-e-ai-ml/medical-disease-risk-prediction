"""
Data Loader Module for Disease Prediction System.
Handles fetching, saving, and standardizing medical datasets.
"""
import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from src.config import DATASETS_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import logger


def download_heart_disease_dataset() -> pd.DataFrame:
    """Download and clean the UCI Cleveland Heart Disease dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    config = DATASETS_CONFIG["heart"]
    raw_path = config["raw_file"]
    
    if not raw_path.exists():
        logger.info(f"Downloading Heart Disease dataset from {url}...")
        try:
            df = pd.read_csv(url, header=None, names=columns, na_values="?")
        except Exception as e:
            logger.warning(f"Failed to download from UCI direct link ({e}). Attempting secondary URL...")
            backup_url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/heart.csv"
            df = pd.read_csv(backup_url)
            # Standardize column names if needed
            if "target" not in df.columns and "output" in df.columns:
                df.rename(columns={"output": "target"}, inplace=True)
                
        # Clean missing values and target
        df.replace("?", np.nan, inplace=True)
        # In UCI Cleveland dataset: 0 = no disease, 1,2,3,4 = heart disease
        if "target" in df.columns:
            df["target"] = (df["target"] > 0).astype(int)
            
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw Heart Disease dataset to {raw_path}")
    else:
        df = pd.read_csv(raw_path)
        
    return df


def download_diabetes_dataset() -> pd.DataFrame:
    """Download and format the PIMA Indian Diabetes dataset."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    
    config = DATASETS_CONFIG["diabetes"]
    raw_path = config["raw_file"]
    
    if not raw_path.exists():
        logger.info(f"Downloading Diabetes dataset from {url}...")
        try:
            df = pd.read_csv(url, header=None, names=columns)
        except Exception as e:
            logger.error(f"Error downloading diabetes dataset: {e}")
            raise e
            
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw Diabetes dataset to {raw_path}")
    else:
        df = pd.read_csv(raw_path)
        
    return df


def download_breast_cancer_dataset() -> pd.DataFrame:
    """Load and format Breast Cancer Wisconsin dataset from scikit-learn."""
    config = DATASETS_CONFIG["breast_cancer"]
    raw_path = config["raw_file"]
    
    if not raw_path.exists():
        logger.info("Loading Breast Cancer dataset from sklearn...")
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        
        # In sklearn: 0 = Malignant, 1 = Benign.
        # We invert so 1 = High Risk (Malignant) and 0 = Low Risk (Benign) for clinical consistency.
        df["target"] = (df["target"] == 0).astype(int)
        
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw Breast Cancer dataset to {raw_path}")
    else:
        df = pd.read_csv(raw_path)
        
    return df


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """
    Unified dataset loader function.
    
    Parameters
    ----------
    dataset_key : str
        One of 'heart', 'diabetes', 'breast_cancer'
        
    Returns
    -------
    pd.DataFrame
        Loaded raw dataset
    """
    if dataset_key not in DATASETS_CONFIG:
        raise ValueError(f"Invalid dataset_key '{dataset_key}'. Must be one of {list(DATASETS_CONFIG.keys())}")
        
    if dataset_key == "heart":
        return download_heart_disease_dataset()
    elif dataset_key == "diabetes":
        return download_diabetes_dataset()
    elif dataset_key == "breast_cancer":
        return download_breast_cancer_dataset()
    else:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")


def get_dataset_summary(df: pd.DataFrame, dataset_key: str) -> dict:
    """Generate basic dataset information dictionary for reports and notebooks."""
    config = DATASETS_CONFIG[dataset_key]
    target_col = config["target_col"]
    
    summary = {
        "dataset_key": dataset_key,
        "name": config["name"],
        "shape": df.shape,
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[target_col].value_counts().to_dict(),
        "target_percentages": (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
    }
    return summary
