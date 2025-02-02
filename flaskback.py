from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Define the model path, relative to the script location
model_path = os.path.join(os.path.dirname(__file__), 'models')  # Path to saved_model.pb
model = tf.saved_model.load(model_path)  # Load model

# Load class labels from labels.txt
def load_class_labels(labels_file):
    with open(labels_file, "r") as file:
        return [line.strip() for line in file.readlines()]

# Preprocess the input image
def preprocess_image(image, target_size=(224, 224)):  # Adjust target_size based on model's expected input size
    img = Image.open(image).convert('RGB')  # Ensure 3-channel RGB
    img = img.resize(target_size)  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize pixel values (0-1)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict function
def predict(image_path):
    image_data = preprocess_image(image_path, target_size=(224, 224))  # Adjust size if needed
    predictions = model(image_data)
    predicted_index = np.argmax(predictions[0])  # Index of highest probability
    confidence = predictions[0][predicted_index]  # Confidence score
    return predicted_index, confidence

@app.route('/predict', methods=['POST'])
def predict_disease():
    # Check if an image is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Predict the class
    predicted_index, confidence = predict(file)

    # Load class labels (ensure labels.txt is correctly included in your project)
    labels_file = os.path.join(os.path.dirname(__file__), 'models', 'labels.txt')  # Path to labels.txt
    class_labels = load_class_labels(labels_file)
    predicted_class = class_labels[predicted_index]

    # Send prediction result as JSON
    result = {
        "predicted_class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    }

    return jsonify(result)

# Specify port and host for Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
