import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pylab as plt

np.random.seed(123)
numclasses = 10
# Data Preparation

imgx, imgy = 28, 28
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()    #60000
xtrain = xtrain.reshape(xtrain.shape[0], imgx, imgy, 1)     # 2 8 * 28
xtest = xtest.reshape(xtest.shape[0], imgx, imgy, 1)
inputshape = (imgx, imgy, 1)

#set data type
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 25
xtest /= 25

ytrain = to_categorical(ytrain, 10)      # [0 1 2 2 4  5]
ytest = to_categorical(ytest, 10)


model = Sequential()       # recurrent / temporal based
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=inputshape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(Dense(64, kernel_size=(5, 5), activation='relu')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dense(numclasses, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, batch_size=128, epochs=10, verbose=1)

