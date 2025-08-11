import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_isic_data(data_dir, csv_file):
    df = pd.read_csv(csv_file)
    image_paths = [os.path.join(data_dir, f"{img_id}.jpg") for img_id in df['image']]
    labels_one_hot = df.iloc[:, 1:].values
    labels = labels_one_hot.argmax(axis=1)
    class_names = df.columns[1:].tolist()
    label_names = [class_names[i] for i in labels]
    return image_paths, labels, label_names, class_names

def get_data_generators(data_dir, csv_file, img_size=(224, 224), batch_size=32, subset_frac=None, seed=42):
    image_paths, labels, label_names, class_names = load_isic_data(data_dir, csv_file)

    if subset_frac is not None:
        sample_size = int(len(image_paths) * subset_frac)
        image_paths = image_paths[:sample_size]
        labels = labels[:sample_size]

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=seed
    )

    train_df = pd.DataFrame({'filename': X_train, 'class': [str(i) for i in y_train]})
    val_df = pd.DataFrame({'filename': X_val, 'class': [str(i) for i in y_val]})

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed
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

def compute_class_weights_from_names(label_names):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(label_names)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=label_names)
    return {int(c): w for c, w in zip(classes, weights)}
