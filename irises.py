# author: Lina Zhu
# Used for skeleton of the code, libraries, k-nearest neighbors approach etc.: 
#   https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))

print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of Target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

# split labelled data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print shape of train samples and respective targets
print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape)) 
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(Y_test.shape))

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train) # arguments = training data and corresponding data labels, builds model on training set

# test?
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted Target Name: {}".format(iris_dataset['target_names'][prediction]))

# measuring the model
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred)) # prints which "class" of iris the model predicts it is
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == Y_test))) # prints how much we got right
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, Y_test)))
