import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from src.utils import download_images
from src.sanity import sanity_check

# Function to load and preprocess images for ResNet-50
def load_and_preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

# Function to extract features using ResNet-50
def extract_features(img_dir, image_ids):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    features = []
    for image_id in tqdm(image_ids):
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        img_array = load_and_preprocess_image(img_path)
        if img_array is not None:
            feature = model.predict(img_array)
            features.append(feature.flatten())
        else:
            features.append(np.zeros((7 * 7 * 2048)))  # Assuming the output of ResNet-50
    return np.array(features)

def main():
    # Step 1: Download images
    download_images('dataset/train.csv')
    download_images('dataset/test.csv')

    # Step 2: Load datasets
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    # Step 3: Extract features using ResNet-50 for training and test data
    train_image_ids = train_df['index']
    test_image_ids = test_df['index']
    
    print("Extracting features for training data...")
    train_features = extract_features('dataset/images', train_image_ids)
    
    print("Extracting features for test data...")
    test_features = extract_features('dataset/images', test_image_ids)

    # Step 4: Prepare target variable (entity_value) for training
    train_df['entity_value_numeric'] = train_df['entity_value'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else 0)
    train_target = train_df['entity_value_numeric']

    # Step 5: Train a simple Ridge regression model to predict entity values
    print("Training Ridge regression model...")
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_target, test_size=0.2, random_state=42)
    model = Ridge()
    model.fit(X_train, y_train)

    # Step 6: Generate predictions for the test dataset
    print("Generating predictions...")
    test_predictions = model.predict(test_features)

    # Step 7: Format predictions with allowed units (Append units to numeric predictions)
    ALLOWED_UNITS = [
        "gram", "kilogram", "ounce", "centimetre", "metre", "inch", "watt", "volt"
    ]
    
    test_df['prediction'] = test_predictions.astype(str) + " " + ALLOWED_UNITS[0]  # Example: Assign first unit (modify as needed)
    
    # Step 8: Save the predictions to the output file
    output_df = test_df[['index', 'prediction']]
    output_df.to_csv('test_out.csv', index=False)
    
    # Step 9: Perform sanity check on the output file
    sanity_check('test_out.csv')

if __name__ == "__main__":
    main()