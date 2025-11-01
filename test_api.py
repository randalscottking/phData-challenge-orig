"""
API Test Script
===============
Tests the housing price prediction API endpoints using examples from
future_unseen_examples.csv
"""

import json
import sys
import time
import requests
import pandas as pd


# API base URL
API_BASE_URL = "http://localhost:5000"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def test_health_check():
    """Test the health check endpoint"""
    print_section("Test 1: Health Check")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print("✗ Health check failed")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        print("  Make sure the API is running:")
        print("  - With Flask: python app.py")
        print("  - With Docker: docker-compose up")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_model_info():
    """Test the model info endpoint"""
    print_section("Test 2: Model Information")

    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model/info", timeout=5)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            print("✓ Model info retrieved successfully")
            return True
        else:
            print("✗ Model info request failed")
            return False

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_single_prediction():
    """Test a single prediction with the full endpoint"""
    print_section("Test 3: Single Prediction (Full Endpoint)")

    # Sample data from future_unseen_examples.csv
    test_data = {
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

    print(f"Input Data: {json.dumps(test_data, indent=2)}")

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            data = response.json()
            price = data.get('predicted_price')
            print(f"\n✓ Prediction successful: ${price:,.2f}")
            return True
        else:
            print("✗ Prediction failed")
            return False

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_minimal_prediction():
    """Test prediction with minimal endpoint"""
    print_section("Test 4: Single Prediction (Minimal Endpoint)")

    # Minimal required features
    test_data = {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2220,
        "sqft_lot": 6380,
        "floors": 1.5,
        "sqft_above": 1660,
        "sqft_basement": 560,
        "zipcode": "98115"
    }

    print(f"Input Data: {json.dumps(test_data, indent=2)}")

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict/minimal",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            data = response.json()
            price = data.get('predicted_price')
            print(f"\n✓ Prediction successful: ${price:,.2f}")
            return True
        else:
            print("✗ Prediction failed")
            return False

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_multiple_predictions():
    """Test multiple predictions from future_unseen_examples.csv"""
    print_section("Test 5: Multiple Predictions from Test Data")

    try:
        # Load test examples
        df = pd.read_csv("data/future_unseen_examples.csv")

        # Test first 5 examples
        num_tests = min(5, len(df))
        print(f"Testing {num_tests} examples from future_unseen_examples.csv\n")

        successes = 0
        failures = 0

        for i in range(num_tests):
            row = df.iloc[i]

            # Prepare minimal data
            test_data = {
                "bedrooms": int(row['bedrooms']),
                "bathrooms": float(row['bathrooms']),
                "sqft_living": int(row['sqft_living']),
                "sqft_lot": int(row['sqft_lot']),
                "floors": float(row['floors']),
                "sqft_above": int(row['sqft_above']),
                "sqft_basement": int(row['sqft_basement']),
                "zipcode": str(row['zipcode'])
            }

            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/predict/minimal",
                    json=test_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    price = data.get('predicted_price')
                    print(f"  Example {i+1}: ${price:,.2f} (zipcode: {test_data['zipcode']}) ✓")
                    successes += 1
                else:
                    print(f"  Example {i+1}: Failed (status {response.status_code}) ✗")
                    failures += 1

            except Exception as e:
                print(f"  Example {i+1}: Error - {str(e)} ✗")
                failures += 1

            # Small delay between requests
            time.sleep(0.1)

        print(f"\nResults: {successes} successful, {failures} failed")

        if failures == 0:
            print("✓ All predictions successful")
            return True
        else:
            print("✗ Some predictions failed")
            return False

    except FileNotFoundError:
        print("✗ Error: Could not find data/future_unseen_examples.csv")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_error_handling():
    """Test API error handling"""
    print_section("Test 6: Error Handling")

    # Test 1: Missing required field
    print("\n6a. Testing missing required field...")
    test_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        # Missing other required fields
        "zipcode": "98118"
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict/minimal",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 400:
            print("✓ Correctly returned 400 error for missing fields")
        else:
            print("✗ Did not handle missing fields correctly")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    # Test 2: Invalid zipcode
    print("\n6b. Testing invalid zipcode...")
    test_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "floors": 2.0,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "zipcode": "99999"  # Invalid zipcode
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict/minimal",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 400:
            print("✓ Correctly returned 400 error for invalid zipcode")
        else:
            print("✗ Did not handle invalid zipcode correctly")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    # Test 3: Invalid endpoint
    print("\n6c. Testing invalid endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/invalid", timeout=5)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 404:
            print("✓ Correctly returned 404 for invalid endpoint")
            return True
        else:
            print("✗ Did not handle invalid endpoint correctly")
            return False

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """Run all API tests"""
    print("="*80)
    print("Housing Price Prediction API Test Suite")
    print("="*80)
    print(f"Testing API at: {API_BASE_URL}")

    # Run all tests
    results = []

    results.append(("Health Check", test_health_check()))

    # Only continue if health check passed
    if not results[0][1]:
        print("\n" + "="*80)
        print("API is not responding. Please start the API server first.")
        print("="*80)
        sys.exit(1)

    results.append(("Model Info", test_model_info()))
    results.append(("Single Prediction (Full)", test_single_prediction()))
    results.append(("Single Prediction (Minimal)", test_minimal_prediction()))
    results.append(("Multiple Predictions", test_multiple_predictions()))
    results.append(("Error Handling", test_error_handling()))

    # Print summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
