"""
Preprocessing Pipeline Builders for Disease Prediction.
Ensures zero data leakage by combining Imputation, Scaling, Encoding, and Model into sklearn Pipelines.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import DATASETS_CONFIG, RANDOM_STATE
from src.utils import logger


def clean_raw_dataframe(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Perform dataset-specific cleaning like converting zeros to NaN where appropriate."""
    df_clean = df.copy()
    config = DATASETS_CONFIG[dataset_key]
    
    # Handle zeros as missing values for physiological metrics in Diabetes dataset
    if "zero_as_missing" in config:
        for col in config["zero_as_missing"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)
                
    # Drop any duplicate rows if they exist
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped = initial_rows - len(df_clean)
    if dropped > 0:
        logger.info(f"[{dataset_key}] Dropped {dropped} duplicate rows.")
        
    return df_clean


def get_train_test_data(df: pd.DataFrame, dataset_key: str, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Clean dataset and split into train and test sets with stratification.
    """
    config = DATASETS_CONFIG[dataset_key]
    target_col = config["target_col"]
    
    df_clean = clean_raw_dataframe(df, dataset_key)
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"[{dataset_key}] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_preprocessor(dataset_key: str) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for a given dataset specification.
    """
    config = DATASETS_CONFIG[dataset_key]
    num_features = config["num_features"]
    cat_features = config["cat_features"]
    
    transformers = []
    
    if num_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_features))
        
    if cat_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_features))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return preprocessor


def get_model_instance(model_name: str, random_state: int = RANDOM_STATE):
    """Factory function for instantiating model classifiers."""
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_name == "Support Vector Machine (SVM)":
        return SVC(random_state=random_state, probability=True)
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=random_state)
    elif model_name == "XGBoost":
        return XGBClassifier(random_state=random_state, eval_metric="logloss")
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def build_model_pipeline(dataset_key: str, model_name: str) -> Pipeline:
    """
    Construct a complete end-to-end scikit-learn Pipeline containing preprocessor and model.
    """
    preprocessor = build_preprocessor(dataset_key)
    classifier = get_model_instance(model_name)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline
