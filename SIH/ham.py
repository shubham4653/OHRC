import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage import exposure
import tensorflow as tf

# Step 1: Preprocess image (Noise reduction and contrast enhancement)
def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Noise reduction using wavelet denoising
    denoised_image = denoise_wavelet(image, mode='soft', wavelet_levels=3, rescale_sigma=True)
    
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply((denoised_image * 255).astype('uint8'))

    return enhanced_image

# Step 3: Apply super-resolution using ESPCN model
def apply_super_resolution(image):
    # Load a pre-trained ESPCN model for super-resolution
    sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
    sr_model.readModel("ESPCN_x4.pb")  # Load the pre-trained model (x4 upscaling)
    sr_model.setModel("espcn", 4)  # Set the model and scale

    # Apply super-resolution
    super_res_image = sr_model.upsample(image)

    return super_res_image

# Step 4: Enhance edges using a bilateral filter
def enhance_edges(image):
    edge_enhanced_image = cv2.bilateralFilter(image, 9, 75, 75)
    return edge_enhanced_image

# Step 5: Apply gamma correction for contrast enhancement
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Step 6: Apply sharpening filter to enhance edges
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Main function integrating all steps
def main(image_path):
    # Step 1: Preprocess image
    preprocessed_image = preprocess_image(image_path)
    
    # Step 3: Apply super-resolution to the enhanced image
    super_res_image = apply_super_resolution(preprocessed_image)

    # Step 4: Apply gamma correction
    gamma_corrected_image = gamma_correction(super_res_image, gamma=1.5)

    # Step 5: Enhance edges for final output
    edge_enhanced_image = enhance_edges(gamma_corrected_image)

    # Step 6: Sharpen the final image
    final_image = sharpen_image(edge_enhanced_image)

    # Save and display the result
    cv2.imwrite("enhanced_lunar_image_final.png", final_image)
    cv2.imshow("Enhanced Lunar Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the main function
image_path = "image.png"  # Replace with your image path
main(image_path)
