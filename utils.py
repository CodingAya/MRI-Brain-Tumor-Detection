import numpy as np
from PIL import Image

def preprocess_image(uploaded_image):
    # Preprocessing the uploaded image
    img = Image.open(uploaded_image)
    print("Original image shape:", img.size) 
    img = img.resize((128, 128))
    print("Resized image shape:", img.size)
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.reshape(1, 128, 128, 3)
    return img

def classify_image(model, image):
    # Making predictions on the input image
    class_labels = ["No Tumor", "Tumor"]
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    prediction = class_labels[class_idx]
    return prediction, confidence
