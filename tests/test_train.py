"""
Unit tests for the training pipeline.
"""
import pytest
import numpy as np
import os
import sys
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    load_california_housing_data,
    evaluate_model,
    save_model,
    load_model
)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_california_housing_data(self):
        """Test that data loading returns correct shapes and types."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes
        assert X_train.shape[1] == X_test.shape[1]  # Same number of features
        assert X_train.shape[0] == y_train.shape[0]  # Same number of samples
        assert X_test.shape[0] == y_test.shape[0]  # Same number of samples
        
        # Check that we have data
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] > 0
        
        # Check that features are 2D and targets are 1D
        assert len(X_train.shape) == 2
        assert len(y_train.shape) == 1
        
        # With polynomial features, we should have more than 8 features
        assert X_train.shape[1] > 8
        
        # Check that test size is approximately 20%
        total_samples = X_train.shape[0] + X_test.shape[0]
        test_ratio = X_test.shape[0] / total_samples
        assert 0.15 <= test_ratio <= 0.25  # Allow some tolerance


class TestModelCreation:
    """Test model creation and training."""
    
    def test_ridge_regression_creation(self):
        """Test that Ridge model can be created."""
        model = Ridge(alpha=1.0, random_state=42)
        assert isinstance(model, Ridge)
    
    def test_model_training(self):
        """Test that model can be trained on the data."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Check that model has been trained
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape matches number of features
        assert model.coef_.shape[0] == X_train.shape[1]
        
        # Check that intercept is a scalar
        assert np.isscalar(model.intercept_)
    
    def test_model_prediction(self):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Check prediction shape
        assert predictions.shape == y_test.shape
        
        # Check that predictions are numeric
        assert np.issubdtype(predictions.dtype, np.number)
        
        # Check that predictions are not all the same (model learned something)
        assert not np.allclose(predictions, predictions[0])


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def test_evaluate_model(self):
        """Test model evaluation returns correct metrics."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test)
        
        # Check that metrics dictionary contains expected keys
        assert 'r2_score' in metrics
        assert 'mse' in metrics
        
        # Check that metrics are numeric
        assert isinstance(metrics['r2_score'], (int, float))
        assert isinstance(metrics['mse'], (int, float))
        
        # Check that R² score is in valid range [-inf, 1]
        assert metrics['r2_score'] <= 1.0
        
        # Check that MSE is non-negative
        assert metrics['mse'] >= 0.0
    
    def test_r2_score_threshold(self):
        """Test that R² score meets minimum threshold requirement."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test)
        
        # Minimum R² threshold as specified in requirements
        min_r2_threshold = 0.6
        assert metrics['r2_score'] >= min_r2_threshold, \
            f"R² score {metrics['r2_score']:.4f} is below minimum threshold {min_r2_threshold}"


class TestModelPersistence:
    """Test model saving and loading functionality."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test that model can be saved and loaded correctly."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Train a model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        save_model(model, str(model_path))
        
        # Check that file was created
        assert model_path.exists()
        
        # Load model
        loaded_model = load_model(str(model_path))
        
        # Check that loaded model is the same type
        assert isinstance(loaded_model, Ridge)
        
        # Check that coefficients are the same
        np.testing.assert_array_almost_equal(model.coef_, loaded_model.coef_)
        np.testing.assert_almost_equal(model.intercept_, loaded_model.intercept_)
        
        # Check that predictions are the same
        original_predictions = model.predict(X_test)
        loaded_predictions = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_data_quality(self):
        """Test that data doesn't contain invalid values."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check for NaN values
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_train).any()
        assert not np.isnan(y_test).any()
        
        # Check for infinite values
        assert not np.isinf(X_train).any()
        assert not np.isinf(X_test).any()
        assert not np.isinf(y_train).any()
        assert not np.isinf(y_test).any()
        
        # Check that target values are positive (house prices)
        assert (y_train > 0).all()
        assert (y_test > 0).all()
    
    def test_feature_scales(self):
        """Test that features are in reasonable ranges."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check that features are not all zeros
        assert not np.allclose(X_train, 0)
        assert not np.allclose(X_test, 0)
        
        # Check that features have some variance
        for i in range(X_train.shape[1]):
            assert np.std(X_train[:, i]) > 0
            assert np.std(X_test[:, i]) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 
