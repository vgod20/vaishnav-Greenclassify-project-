import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Path to dataset folder
# Based on your sidebar, this is the correct relative path
data_dir = "Dataset/train"

# --- DEBUG SECTION ---
# This will print what folders Python sees before trying to load them
if os.path.exists(data_dir):
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"DEBUG: Found {len(folders)} subfolders in {data_dir}")
    print(f"DEBUG: Folders are: {folders}")
else:
    print(f"ERROR: Path '{data_dir}' not found!")
# ---------------------

# 2. Create ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Final Classes found:", train_data.class_indices)

# 3. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), 
    Dense(train_data.num_classes, activation='softmax')
])

# 4. Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Start Training
if train_data.num_classes > 1:
    print(f"Starting training for {train_data.num_classes} classes...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )
    model.save("vegetable_model.h5")
    print("Model saved successfully!")
else:
    print("Error: Only 1 class found. Check if there are loose images in the 'train' folder.")