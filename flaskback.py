from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained Keras model
model_path = 'keras_model.h5'  # Path to your exported .h5 file
model = tf.keras.models.load_model(model_path)

# Load class labels from labels.txt
labels_file_path = 'labels.txt'  # Path to labels.txt
if not os.path.exists(labels_file_path):
    raise FileNotFoundError(f"Labels file not found at {labels_file_path}")

def load_class_labels(labels_file):
    with open(labels_file, "r") as file:
        return [line.strip() for line in file.readlines()]

class_labels = load_class_labels(labels_file_path)  # Load labels into memory

# Preprocess the input image
def preprocess_image(image, target_size=(224, 224)):  # Adjust target_size based on model's expected input size
    img = Image.open(image).convert('RGB')  # Ensure 3-channel RGB
    img = img.resize(target_size)  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize pixel values (0-1)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict function
def predict(image_path):
    image_data = preprocess_image(image_path, target_size=(224, 224))  # Adjust size if needed
    predictions = model.predict(image_data)  # Use the loaded Keras model for prediction
    predicted_index = np.argmax(predictions[0])  # Index of highest probability
    confidence = predictions[0][predicted_index]  # Confidence score
    predicted_class = class_labels[predicted_index]  # Get class name from labels
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict_disease():
    # Check if an image is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Predict the class
    predicted_class, confidence = predict(file)

    # Send prediction result as JSON
    result = {
        "predicted_class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    }

    return jsonify(result)

# Specify port and host for Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
