from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

class CustomScaler:
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()

    def fit(self, data):
        selected_columns_data = data[self.columns_to_scale]
        self.scaler.fit(selected_columns_data)

    def transform(self, data):
        scaled_data = data.copy()
        selected_columns_data = data[self.columns_to_scale]
        selected_columns_scaled = self.scaler.transform(selected_columns_data)
        scaled_columns_df = pd.DataFrame(selected_columns_scaled, columns=self.columns_to_scale, index=data.index)
        scaled_data[self.columns_to_scale] = scaled_columns_df
        return scaled_data


# Load the saved model and scaler
loaded_model = pickle.load(open('Prediction_Model.pkl', 'rb'))
loaded_scaler = pickle.load(open('Prediction_Scaler.pkl', 'rb'))


# Function to append new input data
def append_input_data(data_frame, input_data):
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, \
    median_income, ocean, inland, island, near_bay, near_ocean, bedroom_ratio, household_rooms = input_data

    new_data = {
        'longitude': [longitude], 'latitude': [latitude], 'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms], 'total_bedrooms': [total_bedrooms], 'population': [population],
        'households': [households], 'median_income': [median_income], '<1H OCEAN': [ocean], 'INLAND': [inland],
        'ISLAND': [island], 'NEAR BAY': [near_bay], 'NEAR OCEAN': [near_ocean],
        'bedroom_ratio': [bedroom_ratio], 'household_rooms': [household_rooms]
    }

    new_df = pd.DataFrame(new_data)
    updated_df = pd.concat([data_frame, new_df], ignore_index=True)
    return updated_df


# Initialize an empty DataFrame
data = {
    'longitude': [], 'latitude': [], 'housing_median_age': [], 'total_rooms': [],
    'total_bedrooms': [], 'population': [], 'households': [], 'median_income': [],
    '<1H OCEAN': [], 'INLAND': [], 'ISLAND': [], 'NEAR BAY': [], 'NEAR OCEAN': [],
    'bedroom_ratio': [], 'household_rooms': []
}
df = pd.DataFrame(data)


# API endpoint to predict house price
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json  # Parse JSON input

        # Ensure input data contains the correct keys
        required_keys = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income', '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN',
            'bedroom_ratio', 'household_rooms'
        ]
        
        # Validate input data
        if not all(key in input_data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        # Get input data and call prediction function
        data_values = tuple(input_data[key] for key in required_keys)
        prediction = House_Price_Prediction(data_values)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to predict house price
def House_Price_Prediction(input_data):
    dff = append_input_data(df, input_data)

    # Get the last row as input
    last_row = dff.tail(1)

    # Scale the input data
    scaled_data = loaded_scaler.transform(last_row)

    # Predict using the loaded model
    prediction = loaded_model.predict(scaled_data)
    prediction_value = prediction[0]

    return f"The price is estimated to be around: {prediction_value}"


if __name__ == "__main__":
    app.run(debug=True)
