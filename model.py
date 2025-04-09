import os
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Function to generate batches from directory
def generator(dir_path, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=32, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(
        directory=dir_path, 
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale', 
        class_mode=class_mode,
        target_size=target_size
    )

# Set batch size and image target size
BS = 32
TS = (24, 24)

# Define dataset paths
train_path = os.path.join('c:/Users/MSANI SWARAJ/Desktop/Projects/Drowsiness detection/dataset_new', 'train')
valid_path = os.path.join('c:/Users/MSANI SWARAJ/Desktop/Projects/Drowsiness detection/dataset_new', 'test')

# Generate training and validation batches
train_batch = generator(train_path, shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator(valid_path, shuffle=True, batch_size=BS, target_size=TS)

# Get the number of classes dynamically
num_classes = len(train_batch.class_indices)
print(f"Detected {num_classes} classes: {train_batch.class_indices}")

# Calculate steps per epoch
SPE = max(1, len(train_batch.classes) // BS)  # Avoid division by zero
VS = max(1, len(valid_batch.classes) // BS)

print(f"Steps per epoch: {SPE}, Validation steps: {VS}")

# Define CNN Model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),  # Changed to (2,2) for better feature reduction
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),  
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),  
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjust dynamically based on classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Ensure model directory exists
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save Model
model.save(os.path.join(model_dir, 'cnn_model.h5'), overwrite=True)
print("Model saved successfully!")
