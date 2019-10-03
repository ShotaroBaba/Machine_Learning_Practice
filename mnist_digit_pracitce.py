import sys
sys.path.append('data')

import tensorflow as tf
import numpy as np
import os

from matplotlib import pyplot as plt
from mnist import MNIST
from tensorflow import keras

if not os.path.isdir("data"):
    os.mkdir("data")

if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("saved_models"):
    os.mkdir("saved_models")

mnist_data = MNIST('data', return_type = 'numpy')

model_path = "saved_models/mnist_trained_model.h5"

train_images, train_labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

# Reshape images
train_images = np.reshape(train_images, (60000, 28, 28, 1))
test_images = np.reshape(test_images, (10000, 28, 28, 1))

# Normalise the images.
train_images, test_images = train_images / 255, test_images / 255

# Construct convolutional neural network first to enable image categorisation.


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

model.save(model_path)

model = keras.models.load_model(model_path)

predicted_results = model.predict(test_images[:3000])

sampled_test_labels = test_labels[:3000]

# de-normalise the images for showing images.
sampled_test_images = np.reshape(test_images[:3000] * 255, (3000, 28, 28))

# Save the results of the image categorisation.
for i, img_data in enumerate(sampled_test_images):
    plt.imshow(img_data, interpolation='nearest')
    predicted = predicted_results[i].argmax()
    truth =  sampled_test_labels[i]
    if predicted == truth:
        plt.savefig("results/right_predicted_results_{0}_{1}_{2}.png".format(predicted, truth, i))
    else:
        plt.savefig("results/wrong_predicted_results_{0}_{1}_{2}.png".format(predicted, truth, i))

# Link to the used Python Library:
# - python-mnist: https://github.com/sorki/python-mnist & https://pypi.org/project/python-mnist/