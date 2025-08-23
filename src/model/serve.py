from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
from src.model.model import get_gradcam_heatmap  # if you want Grad-CAM

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model('models/skin_lesion_model.keras')
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        img = Image.open(file).resize((224, 224)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        pred_prob = float(np.max(preds))
        probabilities = preds[0].tolist()

        # Optional: Grad-CAM heatmap
        heatmap = get_gradcam_heatmap(model, img_array)  # shape: (224, 224)
        heatmap = (heatmap * 255).astype(np.uint8).flatten().tolist()  # flatten for frontend canvas

        return jsonify({
            'class': pred_class,
            'probability': pred_prob,
            'probabilities': probabilities,
            'heatmap': heatmap
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
