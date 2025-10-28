"""
Advanced ML Pipeline with sklearn Pipeline and ColumnTransformer
Production-ready preprocessing and model training pipeline
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
import joblib
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TenureGroupTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to create tenure groups from tenure feature"""

    def __init__(self):
        self.labels = ["{0}-{1}".format(i, i + 11) for i in range(1, 72, 12)]
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        # Fit the label encoder on the labels
        self.label_encoder.fit(self.labels)
        return self

    def transform(self, X):
        X = X.copy()
        if "tenure" in X.columns:
            # Create tenure groups
            X["tenure_group"] = pd.cut(
                X["tenure"], bins=range(1, 80, 12), right=False, labels=self.labels
            )
            # Handle any out-of-range values
            X["tenure_group"] = X["tenure_group"].fillna(self.labels[0])
            # Convert to string and encode
            X["tenure_group"] = X["tenure_group"].astype(str)
            X["tenure_group"] = self.label_encoder.transform(X["tenure_group"])
            # Drop original tenure column
            X = X.drop("tenure", axis=1)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create additional engineered features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Average monthly charges
        if "TotalCharges" in X.columns and "tenure" in X.columns:
            # Avoid division by zero: replace tenure=0 with 1
            tenure_safe = X["tenure"].replace(0, 1)
            X["AvgChargesPerMonth"] = X["TotalCharges"] / tenure_safe

        # Service usage score (count of services used)
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        for col in service_cols:
            if col in X.columns:
                X[f"{col}_binary"] = (X[col] != 0).astype(int)

        binary_cols = [
            f"{col}_binary" for col in service_cols if f"{col}_binary" in X.columns
        ]
        if binary_cols:
            X["ServiceUsageScore"] = X[binary_cols].sum(axis=1)

        # Charge-to-service ratio
        if "ServiceUsageScore" in X.columns and "MonthlyCharges" in X.columns:
            X["ChargeToServiceRatio"] = X["MonthlyCharges"] / (
                X["ServiceUsageScore"] + 1
            )

        return X


class ChurnPredictionPipeline:
    """
    Complete production-ready ML pipeline for churn prediction
    """

    def __init__(self, model_type="ensemble", resampling_method="smoteenn"):
        """
        Initialize the pipeline

        Args:
            model_type: 'rf', 'xgb', 'lgbm', 'voting', 'stacking', 'ensemble'
            resampling_method: 'smoteenn', 'adasyn', or None
        """
        self.model_type = model_type
        self.resampling_method = resampling_method
        self.pipeline = None
        self.feature_names = None
        self.label_encoders = {}

    def _get_model(self):
        """Get the specified model"""
        models = {
            "rf": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "xgb": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "lgbm": LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                min_child_samples=20,
                class_weight="balanced",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }

        if self.model_type == "voting":
            # Soft voting classifier
            return VotingClassifier(
                estimators=[
                    ("rf", models["rf"]),
                    ("xgb", models["xgb"]),
                    ("lgbm", models["lgbm"]),
                ],
                voting="soft",
                n_jobs=-1,
            )

        elif self.model_type == "stacking":
            # Stacking classifier
            return StackingClassifier(
                estimators=[
                    ("rf", models["rf"]),
                    ("xgb", models["xgb"]),
                    ("lgbm", models["lgbm"]),
                ],
                final_estimator=LGBMClassifier(
                    n_estimators=100, random_state=42, verbose=-1
                ),
                cv=5,
                n_jobs=-1,
            )

        elif self.model_type == "ensemble":
            # Best performing ensemble
            return models["lgbm"]  # Can be switched to stacking for production

        else:
            return models.get(self.model_type, models["rf"])

    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build the complete preprocessing and modeling pipeline"""

        # Identify column types
        numerical_features = X.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove target if present
        if "Churn" in numerical_features:
            numerical_features.remove("Churn")
        if "Churn" in categorical_features:
            categorical_features.remove("Churn")

        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")

        # Numerical pipeline
        numerical_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
        )

        # Categorical pipeline
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                ),
            ]
        )

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            [
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="passthrough",
        )

        # Get resampling method
        resampler = None
        if self.resampling_method == "smoteenn":
            resampler = SMOTEENN(random_state=42)
        elif self.resampling_method == "adasyn":
            resampler = ADASYN(random_state=42)

        # Build complete pipeline
        if resampler:
            pipeline = ImbPipeline(
                [
                    ("tenure_transformer", TenureGroupTransformer()),
                    ("feature_engineer", FeatureEngineer()),
                    ("preprocessor", preprocessor),
                    ("resampler", resampler),
                    ("classifier", self._get_model()),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("tenure_transformer", TenureGroupTransformer()),
                    ("feature_engineer", FeatureEngineer()),
                    ("preprocessor", preprocessor),
                    ("classifier", self._get_model()),
                ]
            )

        self.pipeline = pipeline
        return pipeline

    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the pipeline with cross-validation

        Args:
            X: Features dataframe
            y: Target series
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing training metrics
        """
        logger.info(
            f"Training {self.model_type} model with {self.resampling_method} resampling"
        )

        # Build pipeline if not already built
        if self.pipeline is None:
            self.build_pipeline(X)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_scores = {}

        for metric in scoring_metrics:
            scores = cross_val_score(
                self.pipeline, X, y, cv=cv, scoring=metric, n_jobs=-1
            )
            cv_scores[metric] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "scores": scores.tolist(),
            }
            logger.info(
                f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std():.4f})"
            )

        # Fit on full training data
        self.pipeline.fit(X, y)
        logger.info("Model training completed successfully")

        return cv_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.pipeline.predict_proba(X)

    def save_model(self, filepath: str):
        """Save the complete pipeline"""
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "resampling_method": self.resampling_method,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved pipeline"""
        model_data = joblib.load(filepath)

        instance = cls(
            model_type=model_data["model_type"],
            resampling_method=model_data["resampling_method"],
        )
        instance.pipeline = model_data["pipeline"]
        instance.feature_names = model_data["feature_names"]

        logger.info(f"Model loaded from {filepath}")
        return instance


def train_multiple_models(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    """
    Train and compare multiple models

    Args:
        df: Preprocessed dataframe
        test_size: Test set size ratio

    Returns:
        Dictionary with results for each model
    """
    # Prepare data
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Define models to train
    model_configs = [
        ("Random Forest", "rf", "smoteenn"),
        ("XGBoost", "xgb", "smoteenn"),
        ("LightGBM", "lgbm", "smoteenn"),
        ("Voting Ensemble", "voting", "smoteenn"),
        ("Stacking Ensemble", "stacking", "smoteenn"),
    ]

    results = {}

    for name, model_type, resampling in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {name}")
        logger.info(f"{'='*60}")

        try:
            pipeline = ChurnPredictionPipeline(
                model_type=model_type, resampling_method=resampling
            )
            cv_scores = pipeline.train(X_train, y_train)

            # Test set evaluation
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                confusion_matrix,
            )

            test_scores = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

            results[name] = {
                "cv_scores": cv_scores,
                "test_scores": test_scores,
                "pipeline": pipeline,
            }

            logger.info(f"\nTest Set Performance:")
            logger.info(f"Accuracy: {test_scores['accuracy']:.4f}")
            logger.info(f"Precision: {test_scores['precision']:.4f}")
            logger.info(f"Recall: {test_scores['recall']:.4f}")
            logger.info(f"F1-Score: {test_scores['f1']:.4f}")
            logger.info(f"ROC-AUC: {test_scores['roc_auc']:.4f}")

        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue

    return results


if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_clean

    # Load data
    df = load_and_clean("../data/Telco_customer_churn.csv")

    # Train models
    results = train_multiple_models(df)

    # Save best model
    best_model_name = max(results.keys(), key=lambda k: results[k]["test_scores"]["f1"])
    best_pipeline = results[best_model_name]["pipeline"]
    best_pipeline.save_model("../models/churn_model_pipeline.pkl")

    print(f"\nBest model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['test_scores']['f1']:.4f}")
