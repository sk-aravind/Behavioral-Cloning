import csv
import cv2
import numpy as np
import helper
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D, Convolution2D

###================== Loading Data ==================###

print ("Loading Data")

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2

# Loading all center image and some left and right images
for line in lines:
    center_image = cv2.imread(get_file_name(line[0]))
    center_angle = float(line[3])
    images.extend([center_image])
    measurements.extend([center_angle])

    if np.random.normal(0.5, 0.5) < 0.3 :
        left_image = cv2.imread(get_file_name(line[1]))
        left_angle = center_angle + correction
        images.extend([left_image])
        measurements.extend([left_angle])

    if np.random.normal(0.5, 0.5) < 0.3 :
        right_image = cv2.imread(get_file_name(line[2]))
        right_angle = center_angle - correction
        images.extend([right_image])
        measurements.extend([right_angle])

print ("Converting Data")
X_train = np.array(images)
y_train = np.array(measurements)

print ("X_train shape: ",X_train.shape)
print ("y_train shape: ",y_train.shape)

# Clear memory
lines = None
images = None
measurements = None

###================== Applying Augmentations ==================###

print ("Applying Augmentations")

aug_img = []
aug_steering = []
for i in range(len(y_train)):
    if np.random.normal(0.5, 0.5) < 0.5 :
        img,steering = augment_data(X_train[i],y_train[i])
        aug_img.append(img)
        aug_steering.append(steering)

aug_img = np.asarray(aug_img)
aug_steering = np.asarray(aug_steering)

print ("Combining Data")
X_train = np.concatenate([aug_img,X_train])
y_train = np.concatenate([aug_steering,y_train])

print ("X_train shape: ",X_train.shape)
print ("y_train shape: ",y_train.shape)

print ("Shuffling Data")
shuffle(X_train, y_train)

###======================= Model =============================###


print ("Building Model")
# Model Architecture

dropout = 0.2
input_shape =(160,320,3)

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
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, epochs=3)

# Save the model
model.save('model.h5')
print ("Model saved")
