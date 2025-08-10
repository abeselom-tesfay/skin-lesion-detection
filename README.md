# Skin Lesion Detection and Classification

This project develops a deep learning model to classify skin lesions from dermoscopic images using the ISIC 2018 dataset. The model uses EfficientNetB0 with transfer learning and includes a React + Tailwind CSS interface for doctors to upload images and view predictions with Grad-CAM heatmaps.

## Features

- Multi-class classification (7 classes: melanoma, nevus, basal cell carcinoma, actinic keratosis/Bowen’s disease, benign keratosis-like lesions, dermatofibroma, vascular lesions).
- Transfer learning with EfficientNetB0 for high accuracy.
- Data augmentation to handle class imbalance and variability.
- Interpretability with Grad-CAM visualizations.

## Requirements

- See `requirements.txt` for Python dependencies.
- For the React app, Node.js and npm are required.

## Dataset

ISIC 2018 Task 3: 10,017 dermoscopic images with 7-class labels.  
Download from ISIC Archive or Kaggle (HAM10000).

## Usage

1. Train the model using `notebooks/train_model.ipynb`.  
2. Start the Flask API to serve predictions.  
3. Access the React UI at `http://localhost:3000` to upload images and view results.
