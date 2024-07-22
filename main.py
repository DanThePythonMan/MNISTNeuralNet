import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def preprocess_image(img_path):
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = np.asarray(img)
    img_array = img_array / 255.0
    img_array = 1.0 - img_array
    img_array = np.reshape(img_array, (28, 28, 1))  # Reshape to 3D array
    print("Image converted and preprocessed successfully.")
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


def display_predictions(predictions):
    digits = range(10)
    for digit, prob in zip(digits, predictions[0]):
        print(f"Digit {digit}: {prob * 100:.4f}%")


# Load the pre-trained model
model = load_model("digit_recognition_model.keras")

# Path to the new image you want to predict
img_path = input("Enter file name: ")
img_array = preprocess_image(img_path)

# Display the preprocessed image

if input("Do you want to see the image (y/N)").lower() == "y":
    plt.imshow(img_array[0].squeeze(), cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

display_predictions(predictions)
print(f"The predicted digit is: {predicted_class}")
