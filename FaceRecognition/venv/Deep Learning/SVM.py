from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

X, y = make_blobs(n_samples=125, centers=2, cluster_std=0.60, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="winter")
# plt.show()

model = SVC(kernel='linear')
history = model.fit(X_train, y_train)

# ax = plt.gca()
# xlim = ax.get_xlim()
# ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="winter", marker='s')
# w = model.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(xlim[0], xlim[1])
# yy = a * xx - (model.intercept_[0] / w[1])
# plt.plot(xx, yy)
plt.show()