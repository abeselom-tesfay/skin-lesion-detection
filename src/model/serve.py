from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from src.model.model import get_gradcam_heatmap

app = Flask(__name__)
model = tf.keras.models.load_model('skin_lesion_model.h5')
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = Image.open(file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    pred_prob = float(np.max(preds))
    
    heatmap = get_gradcam_heatmap(model, img_array)
    heatmap = (heatmap * 255).astype(np.uint8).tolist()
    
    return jsonify({
        'class': pred_class,
        'probability': pred_prob,
        'heatmap': heatmap
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)