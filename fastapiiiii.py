from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

app = FastAPI()

# Define the model path, relative to the script location
model_path = os.path.join(os.path.dirname(__file__), 'model', 'saved_model.pb')

# Verify model existence
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = tf.saved_model.load(model_path)  # Load model

# Load class labels from labels.txt
def load_class_labels(labels_file):
    with open(labels_file, "r") as file:
        return [line.strip() for line in file.readlines()]

# Preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert('RGB')  # Ensure 3-channel RGB
    img = img.resize(target_size)  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize pixel values (0-1)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict function
def predict(image_path):
    image = preprocess_image(image_path)  # Preprocess image
    predictions = model(image)  # Get model predictions
    return tf.nn.softmax(predictions).numpy()  # Apply softmax and return probabilities

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("/tmp", file.filename)
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Perform prediction
        probabilities = predict(temp_file_path)
        os.remove(temp_file_path)  # Clean up the temporary file

        return JSONResponse(content={"predictions": probabilities.tolist()}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
