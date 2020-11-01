# this code will run only if we make new environment variable for machine learning
# tensorflow will download python version and other versions for itself only
# versions are different in tensorflow so using this code on Colab will cause errors in data cleaning etc
# try making new environment variable on anaconda and use jupyter notebook for getting this code working
# Google Colab will cause errors owing to different versions of LabelEncoder() and keras. Different code might be needed for it to run on Colab

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.utils.vis_utils import plot_model
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils import to_categorical
# data is being refined, Here we fist get our datta to be converted into pandas dataframae then we drop empty spaces and eradicate useless indices and getting data in float type
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)  # keeps the dataset with valid enteries only drops all missng values' rows
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# fetching dataset from path
dataset = pd.read_csv('D:/PythonProjects/FaceRecognition/venv/Deep Learning/data/data1.csv', encoding='utf-8')

# for object type data we will label it and transform into apporopriate type fo data after using fit_transform on that colomn to avoid errors
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
data = clean_dataset(dataset)
print(data)
# collecting all required colomns in data to X
X = data.iloc[0:300000, 0:86].values

# collecting our LABEL colomn in y
y = data.iloc[0:300000, -1].values
# y = data[data.columns[len(data.columns)-1]].values

print(y)
# Using Labal encoding again to get it fit
encoder = LabelEncoder()

# fitting our encoder variable to be used further
encoder.fit(y)

# transforming our LABEL colomn in our data for further results
encoder_y = encoder.transform(y)
# dummy_y = np_utils.to_categorical(encoder_y)
# Standardize a dataset along axis
X_scaled = preprocessing.scale(X)

# splitting the data on ratio of 20% test data and 80% data from our encoded X and Y and getting our test and train data parameters
X_train, X_test, y_train, y_test = train_test_split(X_scaled, encoder_y, test_size=0.20, random_state=52)

# starting our NN with sequential
model = Sequential()

# adding dropout tecnique on 0.2 percent of neurons with input of 85 to avoid overfitting
model.add(Dropout(0.2, input_shape=(86,)))

# Stacking layer to sequential model with activation function of ReLU and max weight size of 5 and neurons of input size
model.add(Dense(86, activation='relu', kernel_constraint=maxnorm(5)))
# model.add(Dense(20, input_dim=85, activation='relu'))

# adding dropout tecnique on 0.5 percent of neurons on upcoming layer to avoid overfitting
model.add(Dropout(0.5))

# Stacking layer to sequential model with activation function of ReLU and max weight size of 5 wnd 40 neurons
model.add(Dense(40, activation='relu', kernel_constraint=maxnorm(5)))

# adding dropout tecnique on 0.5 percent of neurons on upcoming layer to avoid overfitting
model.add(Dropout(0.5))

# Stacking layer to sequential model with activation function of Signmoid and 1 output neuron
# model.add(Dense(1, activation='sigmoid')) # for binary classification
model.add(Dense(9, activation='softmax'))

# plotting data for saving img to local library
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


# compiling it with optimzer as sgd and binary sigmoid function named Binary crossentropy although we could use softmax loss as well
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# y_train = to_categorical(y_train)

# finally training our data testing and validation split percentage of 0.33 and 5 iterations/epochs with batchsize 32
# and verbose to be visible
history = model.fit(X_train, y_train, validation_split=0.33, epochs=5, batch_size=32, verbose=1)

# plotting data in grid and plotting our loss, accuracy with epochs
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
model.summary()
model.evaluate(X_test, y_test)

# trying to print our training variables which are actually keys
print(history.history.keys())

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
