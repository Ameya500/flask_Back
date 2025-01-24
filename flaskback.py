from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the saved model
model = tf.saved_model.load("saved_model.pb")  # Path to your saved model


# Load class labels from labels.txt
def load_class_labels(labels_file):
    with open(labels_file, "r") as file:
        return [line.strip() for line in file.readlines()]


# Preprocess the input image
def preprocess_image(image, target_size):
    img = Image.open(image).convert('RGB')  # Ensure 3-channel RGB
    img = img.resize(target_size)  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize pixel values (0-1)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


# Predict function
def predict(image_path):
    image_data = preprocess_image(image_path, model.input_signature[0].shape[1:3])  # Adjust to input shape
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

    # Load class labels
    class_labels = load_class_labels("labels.txt")  # Path to your labels file
    predicted_class = class_labels[predicted_index]

    # Send prediction result as JSON
    result = {
        "predicted_class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
