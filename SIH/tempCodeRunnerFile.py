def enhance_image_with_model(image, model_path='cidnet.h5'):
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