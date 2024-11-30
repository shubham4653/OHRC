import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import os

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(512, 1024)
        )

        self.decoder = nn.Sequential(
            up_conv(1024, 512),
            conv_block(1024, 512),
            up_conv(512, 256),
            conv_block(512, 256),
            up_conv(256, 128),
            conv_block(256, 128),
            up_conv(128, 64),
            conv_block(128, 64)
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[2](enc1)
        enc3 = self.encoder[4](enc2)
        enc4 = self.encoder[6](enc3)
        bottleneck = self.encoder[8](enc4)

        dec4 = self.decoder[0](bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder[1](dec4)
        dec3 = self.decoder[2](dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder[3](dec3)
        dec2 = self.decoder[4](dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder[5](dec2)
        dec1 = self.decoder[6](dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder[7](dec1)

        return self.final_conv(dec1)

# Instantiate the model
model = UNet(in_channels=3, out_channels=1)

# Define Dataset Class
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

# Define Dataset Class
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

class ShadowDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Ensure that only files with valid image extensions are processed
        self.image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.mask_paths = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # Ensure that only pairs of image and mask with the same name are kept
        self.image_paths, self.mask_paths = zip(*[(img, msk) for img, msk in zip(self.image_paths, self.mask_paths) if os.path.splitext(img)[0] == os.path.splitext(msk)[0]])

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_dir, self.image_paths[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

            # Print paths for debugging
            print(f"Image path: {img_path}, Mask path: {mask_path}")

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            return image, mask
        except IndexError:
            print(f"IndexError: Index {idx} is out of range for image or mask paths.")
            raise
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            raise


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Paths to images and masks
image_dir = 'images'
mask_dir = 'labels'

# Data Transformations
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mask_transform = Compose([
    ToTensor()
])

# Load Dataset
dataset = ShadowDataset(image_dir, mask_dir, transform=transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary masks, use BCELoss for pixel-wise binary classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'shadow_removal_model.pth')

# Load the model for inference
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('shadow_removal_model.pth'))
model = model.to(device)
model.eval()
