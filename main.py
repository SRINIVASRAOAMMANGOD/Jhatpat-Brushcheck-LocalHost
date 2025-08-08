import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model & labels
model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Image size used by Teachable Machine
IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classify_image(image_path):
    data = preprocess_image(image_path)
    predictions = model.predict(data)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    return labels[class_index], confidence

if __name__ == "__main__":
    test_img = "images/new_brush.jpg"  # Change this path to test other images
    label, confidence = classify_image(test_img)
    print(f"Prediction: {label} ({confidence*100:.2f}%)")
