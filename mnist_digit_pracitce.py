import sys
sys.path.append('data')

import tensorflow as tf
import numpy as np
import os
import urllib.request
import gzip
import shutil

from matplotlib import pyplot as plt
from mnist import MNIST
from tensorflow import keras

def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)

# If there is no folder, create the folder first.
if not os.path.isdir("data"):
    os.mkdir("data")

if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("saved_models"):
    os.mkdir("saved_models")

train_data_file_path_in = "data/train-images-idx3-ubyte.gz"
test_data_file_path_in = "data/t10k-images-idx3-ubyte.gz"
train_label_file_path_in = "data/train-labels-idx1-ubyte.gz"
test_label_file_path_in = "data/t10k-labels-idx1-ubyte.gz"

train_data_file_path_out = "data/train-images-idx3-ubyte"
test_data_file_path_out = "data/t10k-images-idx3-ubyte"
train_label_file_path_out = "data/train-labels-idx1-ubyte"
test_label_file_path_out = "data/t10k-labels-idx1-ubyte"

# Retrieve the mnist file from mnist file.
train_data_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
test_data_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
train_label_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
test_label_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

if not os.path.isfile(train_data_file_path_out):
    urllib.request.urlretrieve(train_data_url, train_data_file_path_in)
    gunzip_shutil(train_data_file_path_in, train_data_file_path_out)

if not os.path.isfile(test_data_file_path_out):
    urllib.request.urlretrieve(test_data_url, test_data_file_path_in)
    gunzip_shutil(test_data_file_path_in, test_data_file_path_out)

if not os.path.isfile(train_label_file_path_out):
    urllib.request.urlretrieve(train_label_url, train_label_file_path_in)
    gunzip_shutil(train_label_file_path_in, train_label_file_path_out)

if not os.path.isfile(test_label_file_path_out):
    urllib.request.urlretrieve(test_label_url, test_label_file_path_in)
    gunzip_shutil(test_label_file_path_in, test_label_file_path_out)


mnist_data = MNIST('data', return_type = 'numpy')

model_path = "saved_models/mnist_trained_model.h5"

train_images, train_labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

# Reshape images
train_images = np.reshape(train_images, (60000, 28, 28, 1))
test_images = np.reshape(test_images, (10000, 28, 28, 1))

# Normalise the images.
train_images, test_images = train_images / 255, test_images / 255

if not os.path.isfile(model_path):
    # Construct convolutional neural network first to enable image categorisation.
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()

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