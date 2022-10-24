import pandas as pd
import numpy as np
from sklearn import datasets

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


%matplotlib inline
plt.rcParams["figure.figsize"] = [8, 8]

import warnings
warnings.filterwarnings("ignore")

np.random.seed(10)

#functions for visualisation
def get_class_colour(class_label):
    return 'green' if class_label else 'blue'

def plot_points(X, y, new_points=None, new_prediction=None, nearest_points=None, file_name=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=[get_class_colour(y_i) for y_i in y], s=100, edgecolor='black', alpha=0.3)
    
    if new_points is not None:
        plt.scatter(new_points[:, 0], new_points[:, 1], c='black', s=100, edgecolor='black')
    
    if new_prediction is not None:
        plt.scatter(new_points[:, 0], new_points[:, 1], c=[get_class_colour(y_i) for y_i in new_prediction], s=100, edgecolor='black')
        
    if nearest_points is not None:
        plt.scatter(nearest_points[:, 0], nearest_points[:, 1], c='red', s=100, edgecolor='black')
    
    plt.title("Classification problem \n What is the color for the new (x1, x2) pair?")
    plt.xlabel("x1 (feature)")
    plt.ylabel("x2 (feature)")
    
    if file_name:
        plt.savefig(filename)

    X, y = datasets.make_blobs(n_samples=100, random_state=4, centers=2, cluster_std=2)
X[:5]#, y[:5]
#
plot_points(X, y)
#
y[:5]
#
knn = KNeighborsClassifier(n_neighbors=1)
knn
#
knn = KNeighborsClassifier(n_neighbors=1)
# teach
knn.fit(X, y)

# prediction for a new point
X_new = [[12, 5], [8, -3]]
y_pred = knn.predict(X_new)
y_pred
# more new points
X_new = np.c_[np.random.randint(5, 15, 10), np.random.randint(-2, 8, 10)]
X_new
#visualize
plot_points(X, y, new_points=X_new, new_prediction=knn.predict(X_new))

#increase k - number of neighbors to 10

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)
y_pred = knn.predict(X_new)
plot_points(X, y, new_points=X_new, new_prediction=y_pred)
