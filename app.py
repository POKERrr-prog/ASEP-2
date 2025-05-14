from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import base64
import io
from PIL import Image

# Load model
model = tf.keras.models.load_model("asl_model_final.keras")

# Load label mapping
with open("asl_labels.json", "r") as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

# App setup
app = Flask(__name__)
CORS(app)

# Preprocessing function
def preprocess_image(image_base64):
    # Decode base64 to PIL image
    image_data = image_base64.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = image.resize((64, 64))  # Match model input size
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 64, 64, 3)  # Add batch dimension
    return image_array

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        processed_image = preprocess_image(image_base64)

        prediction = model.predict(processed_image)
        predicted_class_index = int(np.argmax(prediction))
        predicted_label = label_map[predicted_class_index]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
