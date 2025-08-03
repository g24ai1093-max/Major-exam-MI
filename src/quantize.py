"""
Quantization script for Linear Regression model parameters.
"""
import numpy as np
import joblib
import os
from utils import load_model, evaluate_model, load_california_housing_data


def quantize_parameters(coef, intercept, scale_factor=10000):
    """
    Manually quantize model parameters to 8-bit unsigned integers.
    
    Args:
        coef: Model coefficients
        intercept: Model intercept
        scale_factor: Scaling factor for quantization
        
    Returns:
        tuple: (quantized_coef, quantized_intercept, scale_factor)
    """
    # Quantize coefficients with clipping to prevent overflow
    coef_scaled = coef * scale_factor
    coef_clipped = np.clip(coef_scaled, 0, 255)
    quantized_coef = np.round(coef_clipped).astype(np.uint8)
    
    # Quantize intercept
    intercept_scaled = intercept * scale_factor
    intercept_clipped = np.clip(intercept_scaled, 0, 255)
    quantized_intercept = np.round(intercept_clipped).astype(np.uint8)
    
    return quantized_coef, quantized_intercept, scale_factor


def dequantize_parameters(quantized_coef, quantized_intercept, scale_factor):
    """
    Dequantize parameters back to float values.
    
    Args:
        quantized_coef: Quantized coefficients
        quantized_intercept: Quantized intercept
        scale_factor: Scaling factor used for quantization
        
    Returns:
        tuple: (dequantized_coef, dequantized_intercept)
    """
    dequantized_coef = quantized_coef.astype(np.float64) / scale_factor
    dequantized_intercept = quantized_intercept.astype(np.float64) / scale_factor
    
    return dequantized_coef, dequantized_intercept


def create_quantized_model(original_model, quantized_coef, quantized_intercept, scale_factor):
    """
    Create a new model with quantized parameters.
    
    Args:
        original_model: Original trained model
        quantized_coef: Quantized coefficients
        quantized_intercept: Quantized intercept
        scale_factor: Scaling factor
        
    Returns:
        Ridge: Model with quantized parameters
    """
    from sklearn.linear_model import Ridge
    
    # Dequantize parameters
    dequantized_coef, dequantized_intercept = dequantize_parameters(
        quantized_coef, quantized_intercept, scale_factor
    )
    
    # Create new model with dequantized parameters
    quantized_model = Ridge(alpha=original_model.alpha)
    quantized_model.coef_ = dequantized_coef
    quantized_model.intercept_ = dequantized_intercept
    
    return quantized_model


def quantize_model():
    """
    Main function to quantize the trained model.
    """
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")
    
    # Extract model parameters
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save unquantized parameters
    print("\nSaving unquantized parameters...")
    unquantized_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(unquantized_params, "models/unquant_params.joblib")
    print("Unquantized parameters saved!")
    
    # Quantize parameters with better scaling
    print("\nQuantizing parameters...")
    scale_factor = 10000  # Increased scale factor for better precision
    quantized_coef, quantized_intercept, scale_factor = quantize_parameters(
        coef, intercept, scale_factor
    )
    
    print(f"Quantized coefficients range: [{quantized_coef.min()}, {quantized_coef.max()}]")
    print(f"Quantized intercept: {quantized_intercept}")
    print(f"Scale factor: {scale_factor}")
    
    # Save quantized parameters
    print("\nSaving quantized parameters...")
    quantized_params = {
        'coef': quantized_coef,
        'intercept': quantized_intercept,
        'scale_factor': scale_factor
    }
    joblib.dump(quantized_params, "models/quant_params.joblib")
    print("Quantized parameters saved!")
    
    # Create quantized model and test performance
    print("\nCreating quantized model...")
    quantized_model = create_quantized_model(
        model, quantized_coef, quantized_intercept, scale_factor
    )
    
    # Load test data for comparison
    _, X_test, _, y_test = load_california_housing_data()
    
    # Compare performance
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    
    original_metrics = evaluate_model(model, X_test, y_test)
    quantized_metrics = evaluate_model(quantized_model, X_test, y_test)
    
    print(f"Original Model R² Score: {original_metrics['r2_score']:.6f}")
    print(f"Quantized Model R² Score: {quantized_metrics['r2_score']:.6f}")
    print(f"Original Model MSE: {original_metrics['mse']:.6f}")
    print(f"Quantized Model MSE: {quantized_metrics['mse']:.6f}")
    
    # Calculate size reduction
    original_size = os.path.getsize("models/linear_regression_model.joblib")
    quantized_size = os.path.getsize("models/quant_params.joblib")
    size_reduction = ((original_size - quantized_size) / original_size) * 100
    
    print(f"\nModel size reduction: {size_reduction:.1f}%")
    print("=" * 50)
    
    return quantized_model


if __name__ == "__main__":
    quantize_model() 