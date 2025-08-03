#!/usr/bin/env python3
"""
Complete MLOps Pipeline Runner
Runs the entire pipeline locally: train -> quantize -> predict
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_pipeline():
    """Run the complete MLOps pipeline."""
    print("🚀 Starting MLOps Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Training
        print("📊 Step 1: Training Linear Regression Model")
        print("-" * 40)
        start_time = time.time()
        
        from train import train_model
        model, X_test, y_test = train_model()
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Step 2: Quantization
        print("\n🔢 Step 2: Model Quantization")
        print("-" * 40)
        start_time = time.time()
        
        from quantize import quantize_model
        quantized_model = quantize_model()
        
        quantization_time = time.time() - start_time
        print(f"✅ Quantization completed in {quantization_time:.2f} seconds")
        
        # Step 3: Prediction
        print("\n🎯 Step 3: Model Prediction")
        print("-" * 40)
        start_time = time.time()
        
        from predict import run_predictions
        predictions, metrics = run_predictions()
        
        prediction_time = time.time() - start_time
        print(f"✅ Prediction completed in {prediction_time:.2f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total execution time: {training_time + quantization_time + prediction_time:.2f} seconds")
        print(f"Final R² Score: {metrics['r2_score']:.4f}")
        print(f"Final MSE: {metrics['mse']:.4f}")
        
        if metrics['r2_score'] >= 0.6:
            print("✅ Model meets minimum performance threshold!")
        else:
            print("⚠️  Model below minimum performance threshold")
        
        print("\n📁 Generated files:")
        print("- models/linear_regression_model.joblib")
        print("- models/unquant_params.joblib")
        print("- models/quant_params.joblib")
        
        print("\n🚀 Ready for GitHub Actions deployment!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1) 