import matplotlib as mpl
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image, ImageOps
import random


def preprocess_image(img_path):
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = np.asarray(img)
    img_array = img_array / 255.0
    img_array = 1.0 - img_array  # Invert colors
    img_array = np.reshape(img_array, (28, 28))  # Reshape to 2D array

    print("Image converted and preprocessed successfully.")
    return img_array


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display a sample image from the dataset
# Change this index to display a different image
while True:
    sample_index = random.randint(0, len(train_images)-1)
    sample_image = train_images[sample_index]
    if train_labels[sample_index] == 7:
        break
plt.imshow(sample_image, cmap='gray')
plt.title(f"Label: {train_labels[sample_index]}")
plt.show()

# Preprocess and display your image
img_path = "3.png"
img_array = preprocess_image(img_path)

# Display the preprocessed image
plt.imshow(img_array, cmap="gray")
plt.title(f"Label: {img_path[:-4]}")
plt.show()
