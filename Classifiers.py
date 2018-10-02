__author__ = 'Victor'
# Demonstration of classification with sklearn and numpy
# (Will only run if these packages have been installed)
import numpy
import scipy
import sklearn

# Access a built-in dataset
from sklearn.datasets import load_iris
iris = load_iris()

# See the dimensions of the dataset
print(iris.target.shape)  # Label array
print(iris.data.shape)  # Feature array

# See parts of the label array
print(iris.target[0])  # First label
print(iris.target[:50])  # First 50 labels
print(iris.target[50:100])  # Next 50 labels
print(iris.target[100:])  # Last 50 labels
print(iris.target[[0, 50, 100]])  # Labels of three specific examples

# See parts of the feature array
print(iris.data[0])  # Features of the first example
print(iris.data[:5])  # Features of the first 5 examples
print(iris.data[[0, 4]])  # Features of two specific examples

# Train a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(iris.data, iris.target)

# See what it predicts (on the training set)
print(knn.predict(iris.data[:50]))  # Group 0
print(knn.predict(iris.data[50:100]))  # Group 1
print(knn.predict(iris.data[100:]))  # Group 2

# See its overall accuracy (on the training set)
print(knn.score(iris.data, iris.target))

# Train a decision-tree classifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(iris.data, iris.target)

# See what it predicts (on the training set)
print(tree.predict(iris.data[:50]))  # Group 0
print(tree.predict(iris.data[50:100]))  # Group 1
print(tree.predict(iris.data[100:]))  # Group 2

# See its overall accuracy (on the training set)
print(tree.score(iris.data, iris.target))

# Split the data into training and testing sets
train_data = iris.data[:35] + iris.data[50:85] + iris.data[100:135]
test_data = iris.data[35:50] + iris.data[85:100] + iris.data[135:]

train_target = iris.target[:35] + iris.target[50:85] + iris.target[100:135]
test_target = iris.target[35:50] + iris.target[85:100] + iris.target[135:]

# Now we can do a more fair evaluation
knn = KNeighborsClassifier().fit(train_data, train_target)
print(knn.score(test_data, test_target))

tree = DecisionTreeClassifier().fit(train_data, train_target)
print(tree.score(test_data, test_target))