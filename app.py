"""
Housing Price Prediction API
=============================
Flask REST API for serving housing price predictions.
"""

import json
import pickle
import pathlib
from typing import Dict, Any, List

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and metadata
MODEL = None
MODEL_FEATURES = None
DEMOGRAPHICS_DATA = None


def load_model():
    """Load the trained model from pickle file"""
    global MODEL
    if MODEL is None:
        model_path = pathlib.Path("model/model.pkl")
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
    return MODEL


def load_model_features():
    """Load the list of required features"""
    global MODEL_FEATURES
    if MODEL_FEATURES is None:
        features_path = pathlib.Path("model/model_features.json")
        with open(features_path, 'r') as f:
            MODEL_FEATURES = json.load(f)
    return MODEL_FEATURES


def load_demographics():
    """Load the demographics data for joining"""
    global DEMOGRAPHICS_DATA
    if DEMOGRAPHICS_DATA is None:
        DEMOGRAPHICS_DATA = pd.read_csv("data/zipcode_demographics.csv",
                                        dtype={'zipcode': str})
    return DEMOGRAPHICS_DATA


def get_demographics_for_zipcode(zipcode: str) -> Dict[str, Any]:
    """
    Get demographic features for a given zipcode.

    Args:
        zipcode: The zipcode to lookup

    Returns:
        Dictionary of demographic features, or None if not found
    """
    demographics = load_demographics()
    zipcode_data = demographics[demographics['zipcode'] == str(zipcode)]

    if zipcode_data.empty:
        return None

    # Convert to dictionary and remove zipcode
    demo_dict = zipcode_data.iloc[0].to_dict()
    demo_dict.pop('zipcode', None)

    return demo_dict


def prepare_features(house_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare features for model prediction by joining with demographics.

    Args:
        house_data: Dictionary with house features including zipcode

    Returns:
        DataFrame with all features required by the model
    """
    # Get demographics for the zipcode
    zipcode = house_data.get('zipcode')
    if not zipcode:
        raise ValueError("zipcode is required")

    demographics = get_demographics_for_zipcode(zipcode)
    if demographics is None:
        raise ValueError(f"No demographic data found for zipcode: {zipcode}")

    # Combine house features with demographics
    features = {}

    # Add house features (excluding zipcode)
    for key in ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'sqft_above', 'sqft_basement']:
        if key in house_data:
            features[key] = house_data[key]

    # Add all demographic features
    features.update(demographics)

    # Create DataFrame with single row
    df = pd.DataFrame([features])

    # Ensure columns are in the correct order
    model_features = load_model_features()
    df = df[model_features]

    return df


def prepare_minimal_features(house_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare features using only the minimum required inputs.
    Uses only the features that the model was actually trained on.

    Args:
        house_data: Dictionary with minimal house features

    Returns:
        DataFrame with required features
    """
    return prepare_features(house_data)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Housing Price Prediction API',
        'version': '1.0.0'
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that accepts house features and returns price prediction.

    Expected JSON input (all features from future_unseen_examples.csv):
    {
        "bedrooms": 4,
        "bathrooms": 1.0,
        "sqft_living": 1680,
        "sqft_lot": 5043,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 4,
        "grade": 6,
        "sqft_above": 1680,
        "sqft_basement": 0,
        "yr_built": 1911,
        "yr_renovated": 0,
        "zipcode": "98118",
        "lat": 47.5354,
        "long": -122.273,
        "sqft_living15": 1560,
        "sqft_lot15": 5765
    }

    Note: The model only uses a subset of these features.
    Demographic data is added automatically based on zipcode.

    Returns:
        JSON response with prediction and metadata
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data'
            }), 400

        # Validate required fields (only those used by the model)
        required_fields = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                          'floors', 'sqft_above', 'sqft_basement', 'zipcode']
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400

        # Prepare features (including demographic data)
        features_df = prepare_features(data)

        # Load model and make prediction
        model = load_model()
        prediction = model.predict(features_df)[0]

        # Return prediction with metadata
        response = {
            'predicted_price': float(prediction),
            'currency': 'USD',
            'model_version': '1.0',
            'input_features': {
                'bedrooms': data.get('bedrooms'),
                'bathrooms': data.get('bathrooms'),
                'sqft_living': data.get('sqft_living'),
                'sqft_lot': data.get('sqft_lot'),
                'floors': data.get('floors'),
                'sqft_above': data.get('sqft_above'),
                'sqft_basement': data.get('sqft_basement'),
                'zipcode': data.get('zipcode')
            },
            'demographic_data_included': True,
            'message': 'Prediction successful'
        }

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({
            'error': 'Invalid input',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/predict/minimal', methods=['POST'])
def predict_minimal():
    """
    Prediction endpoint that requires only the essential features.

    Expected JSON input (minimal required features):
    {
        "bedrooms": 4,
        "bathrooms": 1.0,
        "sqft_living": 1680,
        "sqft_lot": 5043,
        "floors": 1.5,
        "sqft_above": 1680,
        "sqft_basement": 0,
        "zipcode": "98118"
    }

    Returns:
        JSON response with prediction and metadata
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data'
            }), 400

        # Validate required fields
        required_fields = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                          'floors', 'sqft_above', 'sqft_basement', 'zipcode']
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400

        # Prepare features
        features_df = prepare_minimal_features(data)

        # Load model and make prediction
        model = load_model()
        prediction = model.predict(features_df)[0]

        # Return prediction with metadata
        response = {
            'predicted_price': float(prediction),
            'currency': 'USD',
            'model_version': '1.0',
            'input_features': data,
            'demographic_data_included': True,
            'message': 'Prediction successful',
            'note': 'This endpoint uses only the minimal required features'
        }

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({
            'error': 'Invalid input',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/model/info', methods=['GET'])
def model_info():
    """Get information about the deployed model"""
    try:
        features = load_model_features()

        return jsonify({
            'model_name': 'Housing Price Predictor',
            'model_type': 'K-Nearest Neighbors Regression',
            'version': '1.0',
            'description': 'Predicts house prices in the Seattle area using house features and demographic data',
            'required_features': [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'sqft_above', 'sqft_basement', 'zipcode'
            ],
            'total_features': len(features),
            'automatic_features': [
                'demographic_data (26 features joined automatically by zipcode)'
            ],
            'endpoints': {
                '/api/v1/predict': 'Full prediction with all available features',
                '/api/v1/predict/minimal': 'Prediction with only required features',
                '/api/v1/model/info': 'Model information (this endpoint)'
            }
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /health',
            'GET /api/v1/model/info',
            'POST /api/v1/predict',
            'POST /api/v1/predict/minimal'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Pre-load model and data at startup
    print("Loading model and data...")
    load_model()
    load_model_features()
    load_demographics()
    print("Model loaded and ready!")

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
