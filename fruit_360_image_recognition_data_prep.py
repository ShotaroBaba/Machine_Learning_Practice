# The data used in this code: 
# Oltean, M. (2019). Fruits 360, A dataset with 82213 images of 120 fruits and 
# vegetables. Retrieved from https://www.kaggle.com/moltean/fruits

# NOTE: Download the compressed file from https://www.kaggle.com/moltean/fruits/download
# and put training and test data into "./fruit_data" folder
# respectively.

# function to get unique values 
# Create the dictionary converting label to number
# and number to label.
def create_label_dict(given_list): 
   
    uniq_list = [] 
       
    for x in given_list:  
        if x not in uniq_list: 
            uniq_list.append(x) 
    
    # return (integer to character, character to integer)
    return dict([(i,c) for i,c in enumerate(uniq_list)]),\
    dict([(c,i) for i,c in enumerate(uniq_list)])

import sys
import glob
import gc

import tensorflow as tf
import numpy as np
import os
import urllib.request
import gzip
import shutil

import json
from imageio import imread

from matplotlib import pyplot as plt
from mnist import MNIST
from tensorflow import keras

training_dir = "fruit_data/Training"
test_dir = "fruit_data/Test"
train_file_name = os.path.join(training_dir, "training_data.npy")
train_label_file_name = os.path.join(training_dir, "training_label.npy")
test_file_name = os.path.join(test_dir, "test_data.npy")
test_label_file_name = os.path.join(test_dir, "test_label.npy")

i_to_s_file_name = os.path.join(training_dir, "i_to_s.json")
s_to_i_file_name = os.path.join(training_dir, "s_to_i.json")

# Check if the data is properly loaded.
if not os.path.isdir(training_dir):
    print("No training data")
    exit()

if not os.path.isdir(test_dir):
    print("No test data")
    exit()

# Find sub directory.
list_of_training_subdir = [x[0] for x in os.walk(training_dir)]
list_of_test_subdir = [x[0] for x in os.walk(test_dir)]

list_of_test_subdir.remove(test_dir)
list_of_training_subdir.remove(training_dir)


train_label = []
train_data = []
for dir_path in list_of_training_subdir:

    label = os.path.basename(dir_path)
    for _, _, filenames in os.walk(dir_path):
        for filename in filenames:
            train_label.append(label)
            train_data.append(imread(os.path.join(dir_path, filename)))

# Create & save array data to disk
train_data = np.stack(train_data, axis = 0)
np.save(train_file_name,train_data)

del train_data
gc.collect()

i_to_s, s_to_i = create_label_dict(train_label)

# Train data creation
# Write label dictionary to disk
with open(i_to_s_file_name, "w") as f:
    f.write(json.dumps(i_to_s))

with open(s_to_i_file_name, "w") as f:
    f.write(json.dumps(s_to_i))

np.save(train_label_file_name, np.array([s_to_i[x] for x in train_label]))

# Test data creation
test_label = []
test_data = []
for dir_path in list_of_test_subdir:

    label = os.path.basename(dir_path)
    for _, _, filenames in os.walk(dir_path):
        for filename in filenames:
            test_label.append(label)
            test_data.append(imread(os.path.join(dir_path, filename)))

# Create & save array data to disk
test_data = np.stack(test_data, axis = 0)

np.save(test_file_name,test_data)
np.save(test_label_file_name, np.array([s_to_i[x] for x in test_label]))