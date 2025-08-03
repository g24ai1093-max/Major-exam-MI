"""
Utility functions for the MLOps pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib
import os


def load_california_housing_data():
    """
    Load California Housing dataset from sklearn with preprocessing.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and test data
    """
    # Load the California Housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Apply feature scaling to improve model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Add polynomial features to capture non-linear relationships
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    return X_train_poly, X_test_poly, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        dict: Dictionary containing R² score and MSE
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return {
        'r2_score': r2,
        'mse': mse
    }


def save_model(model, filepath):
    """
    Save model using joblib.
    
    Args:
        model: Model to save
        filepath: Path where to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load model using joblib.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def print_model_info(model, X_test, y_test):
    """
    Print model information and performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    """
    # Get model performance
    metrics = evaluate_model(model, X_test, y_test)
    
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of Features: {X_test.shape[1]}")
    print(f"Number of Test Samples: {X_test.shape[0]}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(metrics['mse']):.4f}")
    print("=" * 50)
    
    return metrics 