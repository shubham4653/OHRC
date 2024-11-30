import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from skimage.restoration import denoise_wavelet

# Load the trained model
model = load_model('deep_upe_model500e1.h5')

# Step 1: Preprocess image (Noise reduction and contrast enhancement)
def preprocess_image(image_path):
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Noise reduction using wavelet denoising
    denoised_image = denoise_wavelet(image, mode='soft', wavelet_levels=3, rescale_sigma=True)
    
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply((denoised_image * 255).astype('uint8'))

    return enhanced_image

# Save the enhanced image
def save_image(image_array, save_path):
    image_array = np.squeeze(image_array, axis=0)  # Remove batch dimension
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)  # Rescale to [0, 255]
    img = Image.fromarray(image_array)
    img.save(save_path)

# Test the model on a new low-light image with preprocessing
def test_model_on_image(image_path, save_path):
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Convert grayscale image to RGB by duplicating the single channel
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
    
    # Resize image to 256x256 (model input size) and normalize
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    
    # Prepare the image for the model
    image_input = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    
    # Enhance the image using the model
    enhanced_image = model.predict(image_input)
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
