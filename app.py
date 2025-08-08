from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename  # for safe filenames

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your model and labels once globally
model = load_model('model/keras_model.h5', compile=False)

# Load class names from labels.txt (assumes one class name per line)
with open('model/labels.txt', 'r') as f:
    class_names = f.read().splitlines()

# Map class names to bacteria count (keys must exactly match class_names, case-insensitive)
bacteria_counts = {
    "new": "Low",
    "used": "Medium",
    "dirty": "High",
    "frayed": "Very High"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)  # safer filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        image = Image.open(filepath).convert('RGB').resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1

        # Predict
        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence = prediction[0][index] * 100

        # Get bacteria count based on predicted class (lowercase keys)
        bacteria = bacteria_counts.get(class_name.lower(), "Unknown")

        return render_template('result.html',
                               filename=filename,
                               prediction=class_name,
                               confidence=f"{confidence:.2f}",
                               bacteria=bacteria)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
