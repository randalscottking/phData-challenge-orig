"""
Model Evaluation Script
=======================
Evaluates the performance of the trained housing price prediction model.
"""

import json
import pickle
import pathlib
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn import metrics, model_selection


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the data the same way as create_model.py"""
    SALES_PATH = "data/kc_house_data.csv"
    SALES_COLUMN_SELECTION = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'sqft_above', 'sqft_basement', 'zipcode'
    ]

    # Load sales data
    data = pd.read_csv(SALES_PATH,
                       usecols=SALES_COLUMN_SELECTION,
                       dtype={'zipcode': str})

    # Load demographics data
    demographics = pd.read_csv("data/zipcode_demographics.csv",
                               dtype={'zipcode': str})

    # Merge and prepare features
    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    X = merged_data

    return X, y


def load_model():
    """Load the trained model from pickle file"""
    model_path = pathlib.Path("model/model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_model():
    """Evaluate model performance with various metrics"""
    print("="*80)
    print("Housing Price Model Evaluation")
    print("="*80)

    # Load data
    print("\nLoading data...")
    X, y = load_data()
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    # Split data (same as training)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=42, test_size=0.25
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Load model
    print("\nLoading trained model...")
    model = load_model()

    # Make predictions
    print("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    print("\n" + "="*80)
    print("Performance Metrics")
    print("="*80)

    # R² Score
    train_r2 = metrics.r2_score(y_train, y_train_pred)
    test_r2 = metrics.r2_score(y_test, y_test_pred)

    print(f"\nR² Score:")
    print(f"  Training:   {train_r2:.4f}")
    print(f"  Test:       {test_r2:.4f}")
    print(f"  Difference: {abs(train_r2 - test_r2):.4f}")

    # Mean Absolute Error
    train_mae = metrics.mean_absolute_error(y_train, y_train_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_test_pred)

    print(f"\nMean Absolute Error (MAE):")
    print(f"  Training:   ${train_mae:,.2f}")
    print(f"  Test:       ${test_mae:,.2f}")

    # Root Mean Squared Error
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

    print(f"\nRoot Mean Squared Error (RMSE):")
    print(f"  Training:   ${train_rmse:,.2f}")
    print(f"  Test:       ${test_rmse:,.2f}")

    # Mean Absolute Percentage Error
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print(f"\nMean Absolute Percentage Error (MAPE):")
    print(f"  Training:   {train_mape:.2f}%")
    print(f"  Test:       {test_mape:.2f}%")

    # Residual analysis
    print("\n" + "="*80)
    print("Residual Analysis")
    print("="*80)

    test_residuals = y_test - y_test_pred

    print(f"\nResidual Statistics (Test Set):")
    print(f"  Mean:       ${test_residuals.mean():,.2f}")
    print(f"  Std Dev:    ${test_residuals.std():,.2f}")
    print(f"  Min:        ${test_residuals.min():,.2f}")
    print(f"  Max:        ${test_residuals.max():,.2f}")

    # Prediction range analysis
    print("\n" + "="*80)
    print("Prediction Range Analysis")
    print("="*80)

    print(f"\nActual Prices (Test Set):")
    print(f"  Mean:       ${y_test.mean():,.2f}")
    print(f"  Median:     ${y_test.median():,.2f}")
    print(f"  Min:        ${y_test.min():,.2f}")
    print(f"  Max:        ${y_test.max():,.2f}")

    print(f"\nPredicted Prices (Test Set):")
    print(f"  Mean:       ${y_test_pred.mean():,.2f}")
    print(f"  Median:     ${np.median(y_test_pred):,.2f}")
    print(f"  Min:        ${y_test_pred.min():,.2f}")
    print(f"  Max:        ${y_test_pred.max():,.2f}")

    # Model fitness assessment
    print("\n" + "="*80)
    print("Model Fitness Assessment")
    print("="*80)

    # Check for overfitting
    r2_diff = abs(train_r2 - test_r2)
    if r2_diff < 0.05:
        fitness = "Good fit"
        explanation = "Training and test R² scores are very close."
    elif r2_diff < 0.10:
        fitness = "Acceptable fit"
        explanation = "Slight difference between training and test scores."
    else:
        if train_r2 > test_r2:
            fitness = "Possible overfitting"
            explanation = "Training score is significantly higher than test score."
        else:
            fitness = "Underfitting"
            explanation = "Test score is higher than training score."

    print(f"\nModel Status: {fitness}")
    print(f"Explanation: {explanation}")

    # Performance evaluation
    if test_r2 > 0.80:
        performance = "Excellent"
    elif test_r2 > 0.70:
        performance = "Good"
    elif test_r2 > 0.60:
        performance = "Fair"
    else:
        performance = "Poor"

    print(f"\nTest Performance: {performance}")
    print(f"The model explains {test_r2*100:.1f}% of the variance in house prices.")

    # Recommendations
    print("\n" + "="*80)
    print("Recommendations for Improvement")
    print("="*80)

    print("\n1. Feature Engineering:")
    print("   - The model currently uses only a subset of available features")
    print("   - Consider adding: waterfront, view, condition, grade")
    print("   - Consider adding: yr_built, yr_renovated, lat, long")
    print("   - Consider feature interactions (e.g., sqft_living * grade)")

    print("\n2. Model Selection:")
    print("   - Current: K-Nearest Neighbors (KNN)")
    print("   - Try: Random Forest, Gradient Boosting, or XGBoost")
    print("   - These may capture non-linear relationships better")

    print("\n3. Hyperparameter Tuning:")
    print("   - Optimize number of neighbors (n_neighbors)")
    print("   - Try different distance metrics")
    print("   - Use cross-validation for robust evaluation")

    print("\n4. Data Quality:")
    print("   - Check for outliers in predictions")
    print("   - Validate demographic data join completeness")
    print("   - Consider handling missing values differently")

    print("\n" + "="*80)


if __name__ == "__main__":
    evaluate_model()
