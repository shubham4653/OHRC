import os
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from PIL import Image
from flask import Flask, request, render_template
import torch
from torchvision import transforms
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

app = Flask(_name_)

# Paths for saving uploaded and processed images
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# U-Net model definition
class UNet(nn.Module):
    def _init_(self):
        super(UNet, self)._init_()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.pool(x1)
        x2 = self.encoder2(x2)
        x3 = self.pool(x2)
        x3 = self.middle(x3)
        x4 = self.upconv1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.decoder1(x4)
        x5 = self.upconv2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.decoder2(x5)
        x_out = self.output(x5)
        return x_out

# LLFlowNet model definition
class LLFlowNet(nn.Module):
    def _init_(self):
        super(LLFlowNet, self)._init_()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load U-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load('yolov5500.pth', map_location=device))
unet_model.eval()

# Load LLFlowNet model
llflow_model = LLFlowNet().to(device)
llflow_model.load_state_dict(torch.load('llflow_model.pth', map_location=device))
llflow_model.eval()

# Load DeepUPE model
deep_upe_model = tf.keras.models.load_model('deep_upe_model500e1.h5')

# Load Retinex-Net model
retinex_net_model = tf.keras.models.load_model('retinexnet_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Image transformations
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Step 1: Preprocess image (Noise reduction and contrast enhancement)
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    denoised_image = denoise_wavelet(image, mode='soft', wavelet_levels=3, rescale_sigma=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply((denoised_image * 255).astype('uint8'))
    return enhanced_image

# Step 3: Apply super-resolution using PIL
def apply_super_resolution(image):
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    new_size = (width * 4, height * 4)
    super_res_image = pil_image.resize(new_size, Image.BICUBIC)
    return np.array(super_res_image)

# Step 4: Apply gamma correction
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Step 5: Enhance edges using a bilateral filter
def enhance_edges(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

# Step 6: Sharpen the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Step 7: Enhance image with U-Net model
def enhance_with_unet(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = unet_model(image)
    output_image = output.squeeze().cpu().numpy().transpose((1, 2, 0))
    output_image = (output_image * 255).astype(np.uint8)
    return output_image

# Step 8: Enhance image with LLFlowNet model
def enhance_with_llflow(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = llflow_model(image)
    output_image = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype('uint8')
    return output_image

# Step 9: Enhance image with DeepUPE model
def enhance_with_deep_upe(image_path):
    preprocessed_image = preprocess_image(image_path)
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    enhanced_image = deep_upe_model.predict(image_input)
    enhanced_image = np.squeeze(enhanced_image, axis=0)
    enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)
    return enhanced_image

# Step 10: Enhance image with Retinex-Net model
def enhance_with_retinex_net(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    enhanced_img = retinex_net_model.predict(img_array)
    enhanced_img = np.squeeze(enhanced_img, axis=0)
    enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
    return enhanced_img

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Original image
            original_image = cv2.imread(file_path)

            # Process the image using the preprocessing pipeline
            preprocessed_image = preprocess_image(file_path)
            super_res_image = apply_super_resolution(preprocessed_image)
            gamma_corrected_image = gamma_correction(super_res_image, gamma=1.5)
            edge_enhanced_image = enhance_edges(gamma_corrected_image)
            final_image_pipeline = sharpen_image(edge_enhanced_image)

            # Process the image using U-Net model
            final_image_unet = enhance_with_unet(file_path)

            # Process the image using LLFlowNet model
            final_image_llflow = enhance_with_llflow(file_path)
            
            # Process the image using DeepUPE model
            final_image_deep_upe = enhance_with_deep_upe(file_path)
            
            # Process the image using Retinex-Net model
            final_image_retinex_net = enhance_with_retinex_net(file_path)
            
            # Save all images
            original_path = os.path.join(PROCESSED_FOLDER, "original_" + file.filename)
            pipeline_path = os.path.join(PROCESSED_FOLDER, "pipeline_" + file.filename)
            unet_path = os.path.join(PROCESSED_FOLDER, "unet_" + file.filename)
            llflow_path = os.path.join(PROCESSED_FOLDER, "llflow_" + file.filename)
            deep_upe_path = os.path.join(PROCESSED_FOLDER, "deep_upe_" + file.filename)
            retinex_net_path = os.path.join(PROCESSED_FOLDER, "retinex_net_" + file.filename)
            cv2.imwrite(original_path, original_image)
            cv2.imwrite(pipeline_path, final_image_pipeline)
            cv2.imwrite(unet_path, final_image_unet)
            cv2.imwrite(llflow_path, final_image_llflow)
            cv2.imwrite(deep_upe_path, final_image_deep_upe)
            cv2.imwrite(retinex_net_path, final_image_retinex_net)

            return render_template("index.html", 
                                   original_image=original_path, 
                                   pipeline_image=pipeline_path,
                                   unet_image=unet_path, 
                                   llflow_image=llflow_path,
                                   deep_upe_image=deep_upe_path,
                                   retinex_net_image=retinex_net_path)
    return render_template("index.html")

if _name_ == "_main_":
    app.run(debug=True)