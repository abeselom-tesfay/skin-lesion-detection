import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_isic_data(data_dir, csv_file):
    """Load ISIC 2018 dataset from directory and CSV."""
    df = pd.read_csv(csv_file)
    image_paths = [os.path.join(data_dir, f"{img_id}.jpg") for img_id in df['image']]
    labels = df.iloc[:, 1:].values.argmax(axis=1)  
    class_names = df.columns[1:].tolist()  
    return image_paths, labels, class_names

def get_data_generators(data_dir, csv_file, img_size=(224, 224), batch_size=32):
    """Create data generators with augmentation for training and validation."""
    image_paths, labels, class_names = load_isic_data(data_dir, csv_file)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create dataframes for flow_from_dataframe
    train_df = pd.DataFrame({'filename': X_train, 'class': [str(y) for y in y_train]})
    val_df = pd.DataFrame({'filename': X_val, 'class': [str(y) for y in y_val]})
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation data (only rescale)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, class_names

def compute_class_weights(labels):
    """Compute class weights to handle imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))