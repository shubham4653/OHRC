import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.models import load_model
from PIL import Image



# Load the trained model
model = load_model('deep_upe_model500e1.h5')

# Load a low-light image for testing
def load_image(filepath, target_size=(256, 256)):
    img = load_img(filepath, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Save the enhanced image
def save_image(image_array, save_path):
    image_array = np.squeeze(image_array, axis=0)  # Remove batch dimension
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)  # Rescale to [0, 255]
    img = Image.fromarray(image_array)
    img.save(save_path)

# Test the model on a new low-light image
def test_model_on_image(image_path, save_path):
    low_light_image = load_image(image_path)
    enhanced_image = model.predict(low_light_image)
    save_image(enhanced_image, save_path)
    print(f"Enhanced image saved to {save_path}")

# Folder paths
low_light_test_folder = 'test/low'
enhanced_output_folder = 'test/enhanced'

# Ensure the output folder exists
os.makedirs(enhanced_output_folder, exist_ok=True)

# Test the model on all images in the low-light test folder
for img_filename in os.listdir(low_light_test_folder):
    input_path = os.path.join(low_light_test_folder, img_filename)
    output_path = os.path.join(enhanced_output_folder, f"enhanced_{img_filename}")
    test_model_on_image(input_path, output_path)

print("Testing complete!")
