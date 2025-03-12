# OHRC Lunar Crater Enhancement Project

## Overview
The **OHRC Lunar Crater Enhancement Project** is focused on enhancing Permanently Shadowed Regions (PSR) of lunar craters captured by the OHRC (Orbiter High-Resolution Camera) of Chandrayaan-2. The project leverages **machine learning and deep learning** to enhance low-light images of the Moon's south pole craters.

### Objectives
- Develop a **custom model** with a pipelining approach for better enhancement.
- Create a **hybrid model** combining layers from multiple models to improve shadow removal.
- Identify and enhance lunar craters while removing shadows selectively.
- Preprocess lunar south pole images using a pipeline model before training the **U-Net model**.
- Implement **edge detection** techniques and display results with pixel counts.

## Dataset
The dataset consists of **485 images** of the lunar south pole, categorized into:
- `low/` - Low-light images.
- `high/` - High-resolution reference images.

## Technologies Used
- **Machine Learning & Deep Learning**
- **TensorFlow & PyTorch**
- **OpenCV for Image Processing**
- **Flask for Backend API**
- **React.js for Frontend Visualization**

## Features
- **Shadow Removal & Image Enhancement**: Uses deep learning to enhance lunar crater images.
- **Edge Detection & Analysis**: Detects edges and calculates pixel distribution.
- **Real-time Data Processing**: Implements pipelines for real-time enhancement.
- **Frontend Visualization**: Displays results in an interactive UI.

## Installation
### Prerequisites
- Python 3.8+
- TensorFlow / PyTorch
- OpenCV
- Flask
- React.js (for frontend visualization)

## Team: Shadow Vipers

## Contributors
- **Shubham Awari**
- **Ananti**
- **Kartik Yadav**
- **Archit Mishra**
- **Parth Pareek**



