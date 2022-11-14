from keras.optimizers import Adam
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras_preprocessing.image import ImageDataGenerator
test_path = 'Dataset/test_set'
train_path = 'Dataset/training_set'
train=ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True)
test=ImageDataGenerator(rescale=1./255)
train_batches = train.flow_from_directory(directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=300,shuffle=True,color_mode="grayscale")
test_batches = test.flow_from_directory(directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=300, shuffle=True,color_mode="grayscale")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(512, (3, 3), padding="valid"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation ="relu"))
model.add(Dense(9,activation ="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_batches, batch_size=32,validation_data=test_batches,epochs=25)

model.save('model.h5')
