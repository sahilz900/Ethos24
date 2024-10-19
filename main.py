from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load your model
model = load_model('/Users/devilboy/Downloads/ethos_file.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile it

# Create the Flask application instance
def create_app():
    app = Flask(__name__)

    # Route for the homepage (frontend)
    @app.route('/')
    def home():
        return render_template('index.html')  # Render the HTML template

    # Route for predictions (backend)
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()  # Get data from the request
        features = np.array(data['features']).reshape(1, -1)  # Reshape as needed
        prediction = model.predict(features)  # Make prediction
        return jsonify({'prediction': prediction.tolist()})  # Return prediction as JSON

    return app

if __name__ == '__main__':
    app = create_app()  # Create the app instance
    app.run(debug=True)  # Run the application