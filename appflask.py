import matplotlib
import sys
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import os
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

model = tf.keras.models.load_model("C:/Users/tarik/Downloads/my_model.h5")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PREDICTION_FOLDER = 'predictions'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = tiff.imread(file_path)

    assert image.shape == (128, 128, 12), "Image shape must be (128, 128, 12)"

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i in range (12):
        ax = axes[i // 4, i % 4]
        ax.imshow(image[:, :, i], cmap='gray')
        ax.set_title(f'Band {i+1}')
        ax.axis('off')
    plt.tight_layout()

    band_visualization_path = os.path.join(app.config['PREDICTION_FOLDER'], 'bands_visualization.png')
    plt.savefig(band_visualization_path)
    plt.close()  

    image_rgb = image[:, :, :3]  
    image_expanded = np.expand_dims(image_rgb, axis=0)
    prediction = model.predict(image_expanded)

    prediction_squeezed = np.squeeze(prediction)

    prediction_thresholded = (prediction_squeezed > 0.5).astype(np.float32)

    prediction_image_filename = 'predicted_' + os.path.splitext(filename)[0] + '.png'
    prediction_image_path = os.path.join(app.config['PREDICTION_FOLDER'], prediction_image_filename)

    # Save the prediction as a PNG file
    Image.fromarray((prediction_thresholded * 255).astype(np.uint8)).save(prediction_image_path)

    # Debugging print statements
    print(f"Prediction image saved at: {prediction_image_path}")
    print(f"Prediction image exists: {os.path.exists(prediction_image_path)}")

    return render_template('result.html', 
                            uploaded_image=filename.encode('utf-8').decode('utf-8'), 
                            predicted_image=prediction_image_filename.encode('utf-8').decode('utf-8'), 
                            bands_visualization='bands_visualization.png')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/predictions/<filename>')
def send_predicted_file(filename):
    return send_file(os.path.join(app.config['PREDICTION_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
