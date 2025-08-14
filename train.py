import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
data_dir = "data"  # Folder with subfolders A, B, C...
labels = []
images = []

class_names = sorted(os.listdir(data_dir))  # e.g., ['A', 'B', 'C']

# Check image count per class
for class_name in class_names:
    folder_path = os.path.join(data_dir, class_name)
    print(f"{class_name}: {len(os.listdir(folder_path))} images")

for idx, class_name in enumerate(class_names):
    folder_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(idx)

# 2. Prepare data arrays
X = np.array(images).astype('float32') / 255.0  # Normalize pixel values
y = to_categorical(labels)                      # One-hot encode labels

# Shuffle dataset
X, y = shuffle(X, y, random_state=42)

# 3. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# 5. Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))  # Prevent overfitting
model.add(Dense(len(class_names), activation='softmax'))

# 6. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Set early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 8. Train the model
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=18,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# 9. Save the model
model.save("asl_model.keras")  # Use new format

print("✅ Training complete and model saved as asl_model.keras")
   