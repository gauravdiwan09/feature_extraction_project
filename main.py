import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import download_images
from src.sanity import sanity_check

# Helper function to preprocess images
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_features(model, image_paths):
    features = []
    for img_path in image_paths:
        img_array = preprocess_image(img_path)
        img_array = np.expand_dims(img_array, axis=0)
        feature = model.predict(img_array)
        features.append(feature.flatten())
    return np.array(features)

def main():
    # Download images
    download_images('dataset/train.csv')
    download_images('dataset/test.csv')

    # Load data
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    # Process entity_value to extract numeric values
    def extract_numeric_value(value):
        try:
            return float(value.split()[0])
        except:
            return np.nan
    
    train_df['entity_value_numeric'] = train_df['entity_value'].apply(extract_numeric_value)

    # Ensure 'index' column is present in both dataframes
    if 'index' not in train_df.columns or 'index' not in test_df.columns:
        raise ValueError("'index' column is missing in one of the CSV files")

    # Encode categorical labels
    le = LabelEncoder()
    train_df['entity_name_encoded'] = le.fit_transform(train_df['entity_name'])

    # Prepare image paths and labels
    image_paths = train_df.apply(lambda row: f'dataset/images/{row["index"]}.jpg', axis=1)
    labels = train_df['entity_name_encoded']
    num_classes = len(le.classes_)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(num_classes)
    train_features = extract_features(model, X_train)
    val_features = extract_features(model, X_val)

    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_val_encoded = to_categorical(y_val, num_classes=num_classes)

    model.fit(train_features, y_train_encoded, validation_data=(val_features, y_val_encoded), epochs=5, batch_size=32)

    # Predict on test data
    test_image_paths = test_df.apply(lambda row: f'dataset/images/{row["index"]}.jpg', axis=1)
    test_features = extract_features(model, test_image_paths)

    # Generate predictions
    predictions = model.predict(test_features)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_values = le.inverse_transform(predicted_classes)

    # Prepare output
    output_df = pd.DataFrame({
        'index': test_df['index'],
        'prediction': predicted_values
    })

    output_df.to_csv('test_out.csv', index=False)

    # Perform sanity check
    sanity_check('test_out.csv')

if __name__ == "__main__":
    main()
