import os
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from PIL import Image
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
import torch.nn as nn
import tensorflow as tf

app = Flask(__name__)

# Paths for saving uploaded and processed images
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# U-Net model definition (Placeholder)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
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

# Load U-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load('yolov5500.pth', map_location=device))
unet_model.eval()

# Load DeepUPE model
deep_upe_model = tf.keras.models.load_model('deep_upe_model500e1.h5')

# Load Retinex-Net model
retinex_net_model = tf.keras.models.load_model('retinexnet_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Image transformations
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Image processing functions (Placeholders)
def preprocess_image(image_path):
    return cv2.imread(image_path)

def enhance_with_unet(image_path):
    image = preprocess_image(image_path)
    # Implement actual U-Net processing here
    return 'path_to_unet_processed_image'

def enhance_with_llflow(image_path):
    image = preprocess_image(image_path)
    # Implement actual LLFlow processing here
    return 'path_to_llflow_processed_image'

def enhance_with_deep_upe(image_path):
    image = preprocess_image(image_path)
    # Implement actual DeepUPE processing here
    return 'path_to_deep_upe_processed_image'

def enhance_with_retinex_net(image_path):
    image = preprocess_image(image_path)
    # Implement actual Retinex-Net processing here
    return 'path_to_retinex_net_processed_image'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def upload_and_process_image():
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'success': False, 'message': 'No selected file'})
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'success': False, 'message': 'Error saving file'})
    
    # Process and generate results
    unet_image = enhance_with_unet(file_path)
    llflow_image = enhance_with_llflow(file_path)
    deep_upe_image = enhance_with_deep_upe(file_path)
    retinex_net_image = enhance_with_retinex_net(file_path)
    pipeline_image = preprocess_image(file_path)  # Add processing for pipeline image
    
    response = {
        'success': True,
        'original_image': '/static/uploads/' + file.filename,
        'unet_image': '/static/processed/' + os.path.basename(unet_image),
        'llflow_image': '/static/processed/' + os.path.basename(llflow_image),
        'deep_upe_image': '/static/processed/' + os.path.basename(deep_upe_image),
        'retinex_net_image': '/static/processed/' + os.path.basename(retinex_net_image),
        'pipeline_image': '/static/processed/' + os.path.basename(pipeline_image)
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
