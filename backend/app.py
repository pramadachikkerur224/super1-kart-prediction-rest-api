# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
super_kart_predictor_api = Flask("Super Kart Predictor API")

# Load the trained machine learning model
model = joblib.load("superKart_prediction_model_v1_0.joblib")

# Define a route for the home page (GET request)
@super_kart_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Super Kart Prediction API!"

# Define an endpoint for single property prediction (POST request)
@super_kart_predictor_api.post('/v1/predict')
def predict_rental_price():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing Super Kart details and returns
    the predicted sale as a JSON response.
    """
    # Get the JSON data from the request body
    super_kart_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': super_kart_data['Product_Weight'],
        'Product_Allocated_Area': super_kart_data['Product_Allocated_Area'],
        'Product_MRP': super_kart_data['Product_MRP'],
        'Store_Establishment_Year': super_kart_data['Store_Establishment_Year'],
        'Product_Sugar_Content': super_kart_data['Product_Sugar_Content'],
        'Product_Type': super_kart_data['Product_Type'],
        'Store_Location_City_Type': super_kart_data['Store_Location_City_Type'],
        'Store_Type': super_kart_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]


    # Convert prediction to Python float and round to 2 decimal places
    predicted_sales = round(float(predicted_sales), 2)

    # Return predicted sales
    return jsonify({'Predicted Sales': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@super_kart_predictor_api.post('/v1/batch')
def super_kart_predictor_batch():
    """
    This function handles POST requests to the '/v1/batch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make sales predictions for all SuperKart records in the DataFrame
    predicted_sales = model.predict(input_data).tolist()

    # Convert predictions to Python float and round to 2 decimal places
    predicted_sales = [round(float(sale), 2) for sale in predicted_sales]

    # Create a dictionary with row numbers as keys and predicted sales as values
    output_dict = {
      f"Product_{i+1}": sale
      for i, sale in enumerate(predicted_sales)
    }

    # Return the predicted sales as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    rental_price_predictor_api.run(debug=True)
