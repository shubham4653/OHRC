import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load your trained DeepUPE model
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


# Step 2: Enhance image using your DeepUPE model
def enhance_image_with_model(image):
    # Convert grayscale image to RGB by duplicating the single channel
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize image to 256x256 (model input size) and normalize
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    
    # Prepare the image for the model
    image_input = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    
    # Use your trained model to enhance the image
    enhanced_image = model.predict(image_input)
    
    # Post-process enhanced image: remove batch dimension and rescale back to [0, 255]
    enhanced_image = np.squeeze(enhanced_image, axis=0)  # Remove batch dimension
    enhanced_image = (enhanced_image * 255).astype('uint8')  # Convert back to [0, 255]

    return enhanced_image


# Step 3: Apply super-resolution using PIL
def apply_super_resolution(image):
    # Convert to PIL Image for super-resolution (upscaling)
    pil_image = Image.fromarray(image)
    new_size = (pil_image.width * 4, pil_image.height * 4)  # 4x upscaling
    super_res_image = pil_image.resize(new_size, Image.BICUBIC)
    
    # Convert back to NumPy array
    super_res_image = np.array(super_res_image)

    return super_res_image

# Step 4: Apply edge enhancement using a bilateral filter
def enhance_edges(image):
    edge_enhanced_image = cv2.bilateralFilter(image, 9, 75, 75)
    return edge_enhanced_image

# Step 5: Apply gamma correction for contrast enhancement
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(image, table)
    return gamma_corrected_image

# Step 6: Apply sharpening filter to enhance edges
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Main function integrating all steps and plotting the results
def main(image_path):
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Step 2: Enhance the image using the DeepUPE model
    enhanced_image = enhance_image_with_model(preprocessed_image)

    # Step 3: Apply super-resolution to the enhanced image
    super_res_image = apply_super_resolution(enhanced_image)

    # Step 4: Apply gamma correction for contrast enhancement
    gamma_corrected_image = gamma_correction(super_res_image, gamma=1.5)

    # Step 5: Enhance edges for final output
    edge_enhanced_image = enhance_edges(gamma_corrected_image)

    # Step 6: Sharpen the final image
    final_image = sharpen_image(edge_enhanced_image)

    # Plot the stepwise images
    images = [preprocessed_image, enhanced_image, super_res_image, gamma_corrected_image, edge_enhanced_image, final_image]
    titles = ['Preprocessed', 'Enhanced (Model)', 'Super-Resolution', 'Gamma Corrected', 'Edge Enhanced', 'Final (Sharpened)']

    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the main function with the image path
image_path = "image.png"  # Replace with the path to your input image
main(image_path)
x