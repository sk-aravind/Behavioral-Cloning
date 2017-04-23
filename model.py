import csv
import cv2
import numpy as np
from helper import get_file_name
from helper import augment_data
from helper import mirror
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D, Convolution2D


###=====================================================###
###================== Hyperparameters ==================###
###=====================================================###


input_shape = (160, 320, 3)
validation_split = 0.2
steering_correction = 0.17
dropout = 0.2
optimizer = 'adam'
loss_function = 'mse'
epochs = 5
batch_size = 128

'''
Parameters for controlling the amount of a certain type of data that is generated
Such as the number of left camera images or the amount of augmented images in each batch
Parameters for type of augmentations can be found in helper.py
'''

left_cam_limit = 0.3
right_cam_limit = 0.3
aug_limit = 0.3


###=====================================================###
###================== Loading Data =====================###
###=====================================================###

print ("Loading Data")

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def augmented_generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images =[]
            measurements = []

            for batch_sample in batch_samples:

                # Load Center image and label
                center_image = cv2.imread(get_file_name(batch_sample[0]))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                measurements.append(center_angle)

                # Only flip image when it's turning
                if abs(center_angle) > 0.3:
                    center_image,center_angle = mirror(center_image,center_angle)
                    images.append(center_image)
                    measurements.append(center_angle)

                # Load left camera with correction
                if np.random.normal(0.5, 0.5) < left_cam_limit :
                    left_image = cv2.imread(get_file_name(batch_sample[1]))
                    left_angle = center_angle + steering_correction
                    images.append(left_image)
                    measurements.append(left_angle)

                # Load right camera with correction
                if np.random.normal(0.5, 0.5) < right_cam_limit :
                    right_image = cv2.imread(get_file_name(batch_sample[2]))
                    right_angle = center_angle - steering_correction
                    images.append(right_image)
                    measurements.append(right_angle)

                # Load augmented camera with augmented label
                if np.random.normal(0.5, 0.5) < aug_limit :
                    center_image,center_angle = augment_data(center_image,center_angle)
                    images.append(center_image)
                    measurements.append(center_angle)

                # Yield once batch size has been fulfilled 
                if len(measurements) >=  batch_size:
                    X_train = np.array(images)
                    y_train = np.array(measurements)

                    yield shuffle(X_train, y_train)
            



def generator(samples, batch_size):
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images =[]
            measurements = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(get_file_name(batch_sample[0]))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                measurements.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)


train_generator = augmented_generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


###============================================================###
###========================== Model ===========================###
###============================================================###


print ("Building Model")
# Model Architecture

model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
# Crop
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='linear'))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(50, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))


# Mean squared error
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)


# Save the model
model.save('model.h5')
print ("Model saved")
