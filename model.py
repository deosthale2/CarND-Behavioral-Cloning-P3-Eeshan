import pickle
import numpy as np
import random
import tensorflow as tf
import csv
import cv2
import glob
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from copy_data import *

def preprocess_image(images, measurements, to_flip = 0):
	## Code to preprocess the images by flipping them

    X_train, y_train = images, measurements
    if to_flip == 1:
        flip_measurements = -1.0*measurements
        flip_images = []
        for image in images:
            flip_images += [cv2.flip(image, 1)]
        X_train = np.concatenate((X_train, flip_images), axis = 0)
        y_train = np.concatenate((y_train, flip_measurements), axis = 0)
    return X_train, y_train

def generator(samples, batch_size = 32, is_validation = 0, include_side = 0, to_flip = 0, sample_size = 2000):
	## Generator definition for processing the training data in batches

    samples = random.sample(samples, k=sample_size) if is_validation == 0 else shuffle(samples)
    num_samples = sample_size
    while  True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            center_images, left_images, right_images = [], [], []
            center_measurements = []
            for batch_sample in batch_samples:
                center_images += [cv2.cvtColor(cv2.imread('./data/IMG/'+batch_sample[0].split('IMG/')[-1]), cv2.COLOR_BGR2RGB)]
                center_measurements += [float(batch_sample[3])]
                if include_side == 1:
                    left_images += [cv2.cvtColor(cv2.imread('./data/IMG/'+batch_sample[1].split('IMG/')[-1]), cv2.COLOR_BGR2RGB)]
                    right_images += [cv2.cvtColor(cv2.imread('./data/IMG/'+batch_sample[2].split('IMG/')[-1]), cv2.COLOR_BGR2RGB)]
            images = np.array(center_images)
            measurements = np.array(center_measurements)*5. # Multiplying Steering angle by 5 to better manifest the MSE loss
            if include_side == 1:
                images = np.concatenate((images, left_images, right_images), axis = 0)
                measurements = np.concatenate((measurements, measurements + 0.2*5., measurements - 0.2*5.), axis = 0) # Modifying Steering Angles for Left and Right Images
            if is_validation == 0:
                X_train, y_train = preprocess_image(images, measurements, to_flip = to_flip)
            else:
                X_train, y_train = images, measurements
            yield shuffle(X_train, y_train)

def model_nvidia(train_samples, validation_samples):
	## Definition of NVIDIA End-to-End Learning Model

    batch_size = 32
    sample_size = 3000 ## No. of images to train per epoch
    train_generator = generator(train_samples, batch_size = batch_size, is_validation = 0, include_side = 1, to_flip = 1, sample_size = sample_size)
    validation_generator = generator(validation_samples, batch_size = batch_size, is_validation = 1, include_side = 0)
    
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24, 5, strides = (2,2), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(36, 5, strides = (2,2), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, 5, strides = (2,2), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=opt)
    model.fit_generator(train_generator, 
                        steps_per_epoch = sample_size//batch_size, 
                        validation_data = validation_generator, 
                        validation_steps = len(validation_samples)//batch_size, 
                        epochs = 15, verbose = 1)
    model.save('model.h5')
    
def make_uniform(samples):
	## Code to normalize the image distribution w.r.t. Steering Angles
    no_bins = 25
    augmented_samples = []
    count_thresh = int(len(samples)/no_bins)*3
    samples_arr = np.array(samples)
    angles = np.array(list(map(float, samples_arr[:,3])))
    angle_bins = np.linspace(-1., 1.01, no_bins + 1)
    print(len(angles))
    for i in range(no_bins):
        idx = np.where((angles>=angle_bins[i]) & (angles<angle_bins[i+1]))[0]
        if len(idx) < count_thresh and len(idx) > 0:
            idx_sel = np.random.choice(idx, count_thresh - len(idx))
            samples = samples + samples_arr[idx_sel].tolist()
    samples_arr = np.array(samples)
    angles = np.array(list(map(float, samples_arr[:,3])))
    print(len(angles))
    return samples

def train_model():
    samples = []
    with open('./data/driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == 'center':
                continue
            samples += [line]
    samples = make_uniform(samples)
    csvfile.close()

    with open('data_eeshan/driving_log.csv', 'r') as csvf:
        reader = csv.reader(csvf)
        for line in reader:
            samples += [line]
            samples += [line]
    csvf.close()

    train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

    model_nvidia(train_samples, validation_samples)


if __name__ == '__main__':
    src = './data_eeshan/IMG/'
    dest = './data/IMG/'
    copy_images(src, dest)
    
    train_model()
    print('Done!')