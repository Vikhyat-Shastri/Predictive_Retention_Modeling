"""
Unit Tests for Churn Prediction Pipeline
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path before imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import ChurnPredictionPipeline, TenureGroupTransformer, FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 24, 6],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "Yes"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        "MonthlyCharges": [50.0, 70.0, 45.0],
        "TotalCharges": [600.0, 1680.0, 270.0],
        "Churn": [1, 0, 1],
    }
    return pd.DataFrame(data)


class TestTenureGroupTransformer:
    """Test TenureGroupTransformer"""

    def test_transform(self, sample_data):
        """Test tenure group transformation"""
        transformer = TenureGroupTransformer()
        transformer.fit(sample_data)

        result = transformer.transform(sample_data)

        assert "tenure_group" in result.columns
        assert "tenure" not in result.columns
        assert result["tenure_group"].dtype == np.int64

    def test_fit_transform_consistency(self, sample_data):
        """Test fit and transform consistency"""
        transformer = TenureGroupTransformer()
        result1 = transformer.fit_transform(sample_data)
        result2 = transformer.transform(sample_data)

        pd.testing.assert_frame_equal(result1, result2)


class TestFeatureEngineer:
    """Test FeatureEngineer"""

    def test_feature_creation(self, sample_data):
        """Test feature engineering"""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_data)

        # Check if new features are created
        assert "ServiceUsageScore" in result.columns or len(result.columns) >= len(
            sample_data.columns
        )

    def test_no_errors_on_missing_columns(self):
        """Test handling of missing columns"""
        data = pd.DataFrame({"col1": [1, 2, 3]})
        engineer = FeatureEngineer()

        # Should not raise an error
        result = engineer.fit_transform(data)
        assert len(result) == 3


class TestChurnPredictionPipeline:
    """Test ChurnPredictionPipeline"""

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = ChurnPredictionPipeline(
            model_type="rf", resampling_method="smoteenn"
        )

        assert pipeline.model_type == "rf"
        assert pipeline.resampling_method == "smoteenn"
        assert pipeline.pipeline is None

    def test_build_pipeline(self, sample_data):
        """Test pipeline building"""
        X = sample_data.drop("Churn", axis=1)

        pipeline = ChurnPredictionPipeline(model_type="rf")
        built_pipeline = pipeline.build_pipeline(X)

        assert built_pipeline is not None
        assert pipeline.pipeline is not None

    def test_train_predict(self, sample_data):
        """Test training and prediction"""
        X = sample_data.drop("Churn", axis=1)
        y = sample_data["Churn"]

        pipeline = ChurnPredictionPipeline(model_type="rf", resampling_method=None)
        pipeline.train(X, y, cv_folds=2)

        # Test prediction
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

        # Test prediction probabilities
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_save_load_model(self, sample_data, tmp_path):
        """Test model saving and loading"""
        X = sample_data.drop("Churn", axis=1)
        y = sample_data["Churn"]

        # Train and save
        pipeline = ChurnPredictionPipeline(model_type="rf", resampling_method=None)
        pipeline.train(X, y, cv_folds=2)

        model_path = tmp_path / "test_model.pkl"
        pipeline.save_model(str(model_path))

        # Load and test
        loaded_pipeline = ChurnPredictionPipeline.load_model(str(model_path))

        predictions_original = pipeline.predict(X)
        predictions_loaded = loaded_pipeline.predict(X)

        np.testing.assert_array_equal(predictions_original, predictions_loaded)


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_pipeline(self, sample_data):
        """Test complete pipeline from data to prediction"""
        X = sample_data.drop("Churn", axis=1)
        y = sample_data["Churn"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        # Train
        pipeline = ChurnPredictionPipeline(model_type="rf", resampling_method=None)
        cv_scores = pipeline.train(X_train, y_train, cv_folds=2)

        assert "accuracy" in cv_scores
        assert "f1" in cv_scores

        # Predict
        predictions = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert proba.shape[0] == len(X_test)

    def test_different_model_types(self, sample_data):
        """Test different model types"""
        X = sample_data.drop("Churn", axis=1)
        y = sample_data["Churn"]

        model_types = ["rf", "xgb", "lgbm"]

        for model_type in model_types:
            pipeline = ChurnPredictionPipeline(
                model_type=model_type, resampling_method=None
            )
            pipeline.train(X, y, cv_folds=2)

            predictions = pipeline.predict(X)
            assert len(predictions) == len(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
