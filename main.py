import os
import numpy as np
from keras.models import load_model
from PIL import Image
import os

images_folder = r"C:\Users\HP\Desktop\toothbrush_classifier\images\new_brush"

# Test if folder exists
print(os.path.exists(images_folder))  # Will print True if folder exists, False if not

# Paths
model_path = r"C:\Users\HP\Desktop\toothbrush_classifier\model\keras_model.h5"
labels_path = r"C:\Users\HP\Desktop\toothbrush_classifier\model\labels.txt"
images_folder = r"C:\Users\HP\Desktop\toothbrush_classifier\images"

# Load model and labels
model = load_model(model_path, compile=False)
class_names = open(labels_path, "r").read().splitlines()
print("0 - new,1 - used,2 - Dirty,3 - Frayed\n")
# Loop through all images in folder
for img_name in os.listdir(images_folder):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(images_folder, img_name)

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1

        # Predict
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
     
        # Print result
        print(f"{img_name} --> {class_name} ({confidence_score*100:.2f}%)")
