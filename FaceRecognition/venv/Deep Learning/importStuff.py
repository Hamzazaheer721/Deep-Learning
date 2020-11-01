from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
# initalizing CNN
classifier = Sequential()
# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# 64 * 64 pixel with each pixel having 3 values, we used relu model

# Pooling
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second Convolutional layer
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3
# Flattening

classifier.add(Flatten())

# Step 4
# Full Convolution

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 5  Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

# Fitting the CNN to Images
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test_file_path = tf.keras.utils.get_file('data.csv', 'https://drive.google.com/drive/folders/1oGAZ4v3MEa5ZuZ7UxICaGWBEJfQ5l7QD')
# -
df = pd.read_csv("./data/data1.csv")
training_data = test_datagen.flow_from_directory(df, target_size=(64, 64), batch_size=32, class_mode='binary')