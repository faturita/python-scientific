from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import cv2
import numpy as np

from tensorflow.keras.utils import to_categorical

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(640, 480, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

model.summary()

img1 = cv2.imread('data/classone2.png')
img2 = cv2.imread('data/classtwo2.png')

img1 = cv2.resize(img1, (480, 640))
img1 = img1.astype('float32')/255
img2 = cv2.resize(img2, (480, 640))
img2 = img2.astype('float32')/255

print(img1.shape)

train_data = np.asarray([img1, img2])
train_label = to_categorical(np.array([1, 0]), num_classes=2)

test_data = np.asarray([img1, img2])
test_label = to_categorical(np.array([1, 0]), num_classes=2)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

hist = model.fit(train_data, train_label, batch_size=None, epochs=20,
          verbose=1, shuffle=True)

score = model.evaluate(test_data, test_label, verbose=0)
print('\n', 'Test accuracy:', score[1])