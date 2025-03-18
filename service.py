# Importing Libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("./Visualization/random_forest_model.pkl")
scaler = joblib.load("./Visualization/scaler.pkl")

# Create a Flask app
app = Flask(__name__)
CORS(app)

# Define a POST endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)

    # Convert the JSON data to a NumPy array
    data_array = np.array([list(data.values())])

    # Scale the input features
    scaled_features = scaler.transform(data_array)

    # Make predictions
    prediction = model.predict(scaled_features)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=88000)