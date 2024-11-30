import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from PIL import Image

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = 'train'
HIGH_DIR = 'train/high'
LOW_DIR = 'train/low'
MODEL_SAVE_PATH = 'deepupe_model.h5'
LOG_DIR = 'logs'

# Data loading and preprocessing
def preprocess_image(image):
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

def load_data():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )
    
    train_high = datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        subset='high',
        shuffle=False
    )
    
    train_low = datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        subset='low',
        shuffle=False
    )
    
    return train_low, train_high

# Define the DeepUPE model
def build_model():
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Example layers for DeepUPE (you should replace this with the actual architecture)
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, input_img])
    x = UpSampling2D(size=(2, 2))(x)
    output_img = Conv2D(3, (3, 3), padding='same')(x)
    
    model = Model(inputs=input_img, outputs=output_img)
    
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model

# Load data
train_low, train_high = load_data()

# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Build and train the model
model = build_model()
model.summary()

model.fit(
    train_low,
    epochs=EPOCHS,
    steps_per_epoch=len(train_low),
    validation_data=train_high,
    validation_steps=len(train_high),
    callbacks=[tensorboard_callback]
)

# Save the model
model.save(UPE.h5)


