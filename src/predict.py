"""
Prediction script for the trained Linear Regression model.
"""
import numpy as np
from utils import load_model, load_california_housing_data, evaluate_model


def run_predictions():
    """
    Load trained model and run predictions on test data.
    """
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")
    
    print("Loading test data...")
    _, X_test, _, y_test = load_california_housing_data()
    
    print(f"Test data shape: {X_test.shape}")
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = model.predict(X_test)
    
    # Print sample predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    
    num_samples = min(10, len(predictions))
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        print(f"  Actual: {y_test[i]:.4f}")
        print(f"  Predicted: {predictions[i]:.4f}")
        print(f"  Difference: {abs(y_test[i] - predictions[i]):.4f}")
        print()
    
    # Calculate and print performance metrics
    metrics = evaluate_model(model, X_test, y_test)
    
    print("=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(metrics['mse']):.4f}")
    print("=" * 50)
    
    # Verify R² score meets minimum threshold
    min_r2_threshold = 0.6
    if metrics['r2_score'] >= min_r2_threshold:
        print(f"✅ Model meets minimum R² threshold ({min_r2_threshold})")
    else:
        print(f"❌ Model does not meet minimum R² threshold ({min_r2_threshold})")
    
    return predictions, metrics


if __name__ == "__main__":
    run_predictions() 
