## Skin Lesion Detection and Classification
This project develops a deep learning model to classify skin lesions from dermoscopic images using the ISIC 2018 dataset. The model uses EfficientNetB0 with transfer learning and includes a React + Tailwind CSS interface for doctors to upload images and view predictions with Grad-CAM heatmaps.
Features

### Multi-class classification (7 classes: melanoma, nevus, etc.).
Transfer learning with EfficientNetB0 for high accuracy.
Data augmentation to handle class imbalance and variability.
Interpretability with Grad-CAM visualizations.

### Setup

Clone the repository: git clone <repo-url>
Install dependencies: pip install -r requirements.txt
Download the ISIC 2018 dataset and place it in data/.
Run the Jupyter notebook notebooks/train_model.ipynb to train the model.
Start the Flask server: python src/model/serve.py
Run the React app: cd src/app && npm install && npm start

### Requirements
See requirements.txt for Python dependencies. For the React app, Node.js and npm are required.
Dataset

ISIC 2018 Task 3: ~10,000 dermoscopic images with 7-class labels.
Download from ISIC Archive or Kaggle (HAM10000).

### Usage

Train the model using notebooks/train_model.ipynb.
Start the Flask API to serve predictions.
Access the React UI at http://localhost:3000 to upload images and view results.

### Future Work

Fine-tune additional layers of EfficientNet.
Add support for clinical (non-dermoscopic) images.
