import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
import tensorflow as tf
from PIL import Image

# Step 1: Preprocess image (Noise reduction and contrast enhancement)
def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Noise reduction using wavelet denoising
    denoised_image = denoise_wavelet(image, mode='soft', wavelet_levels=3, rescale_sigma=True)
    
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply((denoised_image * 255).astype('uint8'))

    # Save and show preprocessed image
    cv2.imwrite("preprocessed_image.png", enhanced_image)
    cv2.imshow("Preprocessed Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return enhanced_image

# Step 2: Enhance image using a deep learning model
# def enhance_image_with_model(image, model_path='cidnet.h5'):
#     # Load pre-trained model
#     model = tf.keras.models.load_model(model_path)
    
#     # Prepare image for model
#     image = cv2.resize(image, (256, 256))  # Resize to model input size
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     image = image / 255.0  # Normalize the image to [0, 1]
    
#     # Use model for enhancement
#     enhanced_image = model.predict(image)
#     enhanced_image = np.squeeze(enhanced_image, axis=0)  # Remove batch dimension
#     enhanced_image = (enhanced_image * 255).astype('uint8')  # Convert back to [0, 255]

#     # Save and show enhanced image
#     cv2.imwrite("enhanced_image.png", enhanced_image)
#     cv2.imshow("Enhanced Image (Model)", enhanced_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return enhanced_image

# Step 3: Apply super-resolution using PIL (as a placeholder for missing cv2.dnn_superres)
def apply_super_resolution(image):
    # Handle single channel grayscale (shape: (256, 256, 1))
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)  # Remove the extra dimension
    
    # Convert to PIL Image for super-resolution (for upscaling)
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    new_size = (width * 4, height * 4)  # 4x upscaling
    super_res_image = pil_image.resize(new_size, Image.BICUBIC)
    
    # Convert back to NumPy array
    super_res_image = np.array(super_res_image)

    # Save and show super-resolved image
    cv2.imwrite("super_res_image.png", super_res_image)
    cv2.imshow("Super-Resolved Image", super_res_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return super_res_image

# Step 4: Enhance edges using a bilateral filter
def enhance_edges(image):
    edge_enhanced_image = cv2.bilateralFilter(image, 9, 75, 75)

    # Save and show edge-enhanced image
    cv2.imwrite("edge_enhanced_image.png", edge_enhanced_image)
    cv2.imshow("Edge Enhanced Image", edge_enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edge_enhanced_image

# Step 5: Apply gamma correction for contrast enhancement
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(image, table)

    # Save and show gamma-corrected image
    cv2.imwrite("gamma_corrected_image.png", gamma_corrected_image)
    cv2.imshow("Gamma Corrected Image", gamma_corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return gamma_corrected_image

# Step 6: Apply sharpening filter to enhance edges
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Save and show sharpened image
    cv2.imwrite("sharpened_image.png", sharpened_image)
    cv2.imshow("Sharpened Image", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sharpened_image

# Main function integrating all steps
def main(image_path):
    # Step 1: Preprocess image
    preprocessed_image = preprocess_image(image_path)
    
    # Step 2: Enhance the image using a deep learning model
    # enhanced_image = enhance_image_with_model(preprocessed_image)

    # Step 3: Apply super-resolution to the enhanced image
    super_res_image = apply_super_resolution(preprocessed_image)

    # Step 4: Apply gamma correction
    gamma_corrected_image = gamma_correction(super_res_image, gamma=1.5)

    # Step 5: Enhance edges for final output
    edge_enhanced_image = enhance_edges(gamma_corrected_image)

    # Step 6: Sharpen the final image
    final_image = sharpen_image(edge_enhanced_image)

    # Save and display the final result
    cv2.imwrite("enhanced_lunar_image_final.png", final_image)
    cv2.imshow("Enhanced Lunar Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the main function
image_path = "image.png"  # Replace with your image path
main(image_path)
