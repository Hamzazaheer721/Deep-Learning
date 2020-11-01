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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


# data is being refined, Here we fist get our datta to be converted into pandas dataframae then we drop empty spaces and eradicate useless indices and getting data in float type
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
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
# print(data["Sub_Cat"].value_counts())
cols = data.columns.tolist()
print(cols)
# collecting all required colomns in data to X
X = data.iloc[0:20000, 0:86].values

# collecting our LABEL colomn in y
y = data.iloc[0:20000, -1].values
# y = np.asarray(data["Label"])
# print(X)
# print(X[0:4])

# Using Labal encoding again to get it fit
encoder = LabelEncoder()

# fitting our encoder variable to be used further
encoder.fit(y)

# transforming our LABEL colomn in our data for further results
encoder_y = encoder.transform(y)

# Standardize a dataset along axis
X_scaled = preprocessing.scale(X)

print(y)
print(X)
# splitting the data on ratio of 20% test data and 80% data from our encoded X and Y and getting our test and train data parameters
X_train, X_test, y_train, y_test = train_test_split(X_scaled, encoder_y, test_size=0.20, random_state=52)

# starting our NN with sequential
# model = SVC(cache_size=8000, kernel="rbf")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
# model = GridSearchCV(SVC(), params_grid, cv=2)
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())