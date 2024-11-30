import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the dataset
def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        img = load_img(os.path.join(folder, filename), target_size=target_size)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images), filenames

# Define the model
def deep_upe_model(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)
    
    # DeepUPE network structure (simplified example)
    conv1 = Conv2D(32, (3, 3), padding='same')(input_img)
    bn1 = BatchNormalization()(conv1)
    act1 = ReLU()(bn1)

    conv2 = Conv2D(32, (3, 3), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = ReLU()(bn2)

    conv3 = Conv2D(3, (3, 3), padding='same')(act2)

    # Skip connection
    output_img = Add()([input_img, conv3])
    
    model = Model(inputs=input_img, outputs=output_img)
    return model

# Load images
high_images, filenames = load_images_from_folder('train/high')
low_images, _ = load_images_from_folder('train/low')

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(low_images, high_images, test_size=0.1, random_state=42)

# Compile model
model = deep_upe_model(input_shape=(256, 256, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

# TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[tensorboard_callback]
)

# Save the trained model
model.save('deep_upe_model.h5')

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
