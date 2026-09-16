import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Load and preprocess the image
image = cv2.imread("scary_apple.png")  # Load image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = cv2.resize(image, (224, 224))  # Resize
image = image.astype("float32") / 255.0  # Normalize

# Ensure correct shape
X_train = np.expand_dims(image, axis=0)  # Add batch dimension

# Prepare labels
y_train = np.array([0])  # Example class label
y_train = to_categorical(y_train, num_classes=10)

# Build a simple model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    Flatten(),
    Dense(10, activation="softmax")
])

# Compile and train
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5)
