"""
Central configuration for Disease Prediction ML pipeline.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure all directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Global Seed
RANDOM_STATE = 42

# Dataset Configuration Schema
DATASETS_CONFIG = {
    "heart": {
        "name": "Heart Disease",
        "description": "Predict heart disease risk using demographic and diagnostic parameters.",
        "target_col": "target",
        "raw_file": RAW_DATA_DIR / "heart.csv",
        "processed_file": PROCESSED_DATA_DIR / "heart_processed.csv",
        "model_path": MODELS_DIR / "heart_best_model.pkl",
        "num_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "cat_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "feature_descriptions": {
            "age": "Age in years",
            "sex": "Sex (0 = Female, 1 = Male)",
            "cp": "Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic)",
            "trestbps": "Resting Blood Pressure (mmHg)",
            "chol": "Serum Cholestoral (mg/dl)",
            "fbs": "Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)",
            "restecg": "Resting ECG Results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
            "thalach": "Maximum Heart Rate Achieved",
            "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
            "oldpeak": "ST Depression Induced by Exercise Relative to Rest",
            "slope": "Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)",
            "ca": "Number of Major Vessels (0-3) Colored by Flourosopy",
            "thal": "Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversable Defect)"
        },
        "target_label_map": {0: "Low Risk (No Heart Disease)", 1: "High Risk (Heart Disease Present)"}
    },
    "diabetes": {
        "name": "Diabetes",
        "description": "Predict diabetes onset based on diagnostic medical measurements.",
        "target_col": "Outcome",
        "raw_file": RAW_DATA_DIR / "diabetes.csv",
        "processed_file": PROCESSED_DATA_DIR / "diabetes_processed.csv",
        "model_path": MODELS_DIR / "diabetes_best_model.pkl",
        "num_features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "cat_features": [],
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "feature_descriptions": {
            "Pregnancies": "Number of times pregnant",
            "Glucose": "Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
            "BloodPressure": "Diastolic blood pressure (mmHg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (weight in kg / height in m^2)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function (hereditary score)",
            "Age": "Age in years"
        },
        "target_label_map": {0: "Low Risk (Non-Diabetic)", 1: "High Risk (Diabetic)"}
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "description": "Predict breast tumor malignancy from cell nuclei characteristics.",
        "target_col": "target",
        "raw_file": RAW_DATA_DIR / "breast_cancer.csv",
        "processed_file": PROCESSED_DATA_DIR / "breast_cancer_processed.csv",
        "model_path": MODELS_DIR / "breast_cancer_best_model.pkl",
        "num_features": [
            "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
            "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
            "radius error", "texture error", "perimeter error", "area error", "smoothness error",
            "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
            "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
            "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
        ],
        "cat_features": [],
        "feature_descriptions": {
            "mean radius": "Mean radius of lobes",
            "mean texture": "Standard deviation of gray-scale values",
            "mean perimeter": "Mean size of core tumor",
            "mean area": "Mean tumor area",
            "mean smoothness": "Mean local variation in radius lengths",
            "worst radius": "Largest radius",
            "worst texture": "Worst texture score",
            "worst perimeter": "Largest perimeter",
            "worst area": "Largest area"
        },
        "target_label_map": {0: "Low Risk (Benign)", 1: "High Risk (Malignant)"}
    }
}

# Machine Learning Algorithms to train and compare
MODEL_NAMES = [
    "Logistic Regression",
    "Support Vector Machine (SVM)",
    "Random Forest",
    "XGBoost"
]

# Hyperparameter Tuning Grids
HYPERPARAMETER_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__max_iter": [1000]
    },
    "Support Vector Machine (SVM)": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"]
    },
    "Random Forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    },
    "XGBoost": {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0]
    }
}
