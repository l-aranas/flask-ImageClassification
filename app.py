from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import tensorflow as tf
import numpy as np
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'person_classifier_model.h5'
IMAGE_SIZE = (128, 128)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_trained_model():
    """Load the trained model if it exists."""
    if os.path.exists(MODEL_PATH):
        print("✅ Model loaded successfully!")
        return load_model(MODEL_PATH)
    print("⚠️ No trained model found!")
    return None

model = load_trained_model()
def load_data():
    """Load image data and labels for training."""
    image_data, labels, class_names = [], [], {}
    
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            parts = file.split('_')
            if len(parts) < 2:
                continue
            label = parts[0]
            if label not in class_names:
                class_names[label] = len(class_names)
            img_path = os.path.join(UPLOAD_FOLDER, file)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            image_data.append(img_array)
            labels.append(class_names[label])
    
    if len(class_names) < 2:
        return None, None, None
    
    return np.array(image_data), np.array(labels), list(class_names.keys())

def create_model(num_classes):
    model = Sequential([
    Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

    Conv2D(16, (3, 3), activation='relu'),  # Reduced filters
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(32, (3, 3), activation='relu'),  # Reduced filters
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),  # Reduced filters
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),  # Reduced neurons
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


@app.route('/')
def home():
    images = os.listdir(UPLOAD_FOLDER)
    return render_template('index.html', images=images)

@app.route('/check_model_status', methods=['GET'])
def check_model_status():
    """Check if a trained model exists."""
    model_exists = os.path.exists(MODEL_PATH)
    images_exist = any(file.endswith(('.jpg', '.jpeg', '.png')) for file in os.listdir(UPLOAD_FOLDER))
    return jsonify({'ready': model_exists and images_exist})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    label = request.form.get('label', 'unknown')
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    ext = file.filename.rsplit('.', 1)[-1]
    unique_filename = f"{label}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(upload_path)
    
    return jsonify({'message': 'Image uploaded successfully', 'filename': unique_filename})

@app.route('/train', methods=['POST'])
def train():
    global progress, model
    progress = 0

    # Load dataset
    image_data, labels, class_names = load_data()
    if image_data is None or len(class_names) < 2:
        return jsonify({'message': 'Not enough data to train'}), 400

    X_train, X_val, y_train, y_val = train_test_split(image_data, labels, test_size=0.2, random_state=42)

    model = create_model(len(class_names))
    
    for epoch in range(50):
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, verbose=1)
        progress = int((epoch + 1) / 50 * 100)

    model.save(MODEL_PATH)
    progress = 100

    return jsonify({'message': 'Training complete'})

@app.route('/predict')
def predict_page():
    """Render the prediction page with uploaded images."""
    images = sorted(
        [img for img in os.listdir(UPLOAD_FOLDER) if img.endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)),
        reverse=True
    )
    return render_template('predict.html', images=images)

@app.route('/predict_image', methods=['GET'])
def predict_image():
    global model
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'})
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'})
    
    if model is None:
        model = load_trained_model()
        if model is None:
            return jsonify({'error': 'Model not trained'})
    
    img = load_img(file_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    _, _, class_names = load_data()
    if not class_names:
        return jsonify({'error': 'Class names not found'})
    
    predicted_class = class_names[class_index]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
