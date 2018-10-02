__author__ = 'Victor'
# Purpose: compare classifiers on digit classification.

import numpy
import scipy
import sklearn


from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


digits = load_digits()
(length, width) = digits.data.shape

folds = 5

# Split the digits data-set into 5 folds for cross-validation.
f_length = length // folds
f_list = list()
for fold in range(folds):
    start = f_length*fold
    end = f_length + start
    f_list.append((digits.data[start:end], digits.target[start:end]))

# Evaluate the KNeighborsClassifier on each fold.
# Train a nearest-neighbor classifier
f_knn = list()
for fold in range(folds):
    (training_data, training_target) = f_list[(1+fold) % folds]
    (testing_data, testing_target) = f_list[fold]
    for i in range(folds-2):
        (data, target) = f_list[(fold + i + 2) % folds]
        numpy.append(training_data, data)
        numpy.append(training_target, target)
    knn = KNeighborsClassifier().fit(training_data, training_target)
    f_knn.append(knn.score(testing_data, testing_target))

# Evaluate the DecisionTreeClassifier on each fold.
f_decisionTree = list()
for fold in range(folds):
    (training_data, training_target) = f_list[(1+fold) % folds]
    (testing_data, testing_target) = f_list[fold]
    for fold in range(folds-2):
        (data, target) = f_list[(fold + i + 2) % folds]
        numpy.append(training_data, data)
        numpy.append(training_target, target)
    decisionTree = DecisionTreeClassifier().fit(training_data, training_target)
    f_decisionTree.append(decisionTree.score(testing_data, testing_target))


# Report the average accuracy of each classifier.
print('Accuracy Percentage of Decision Tree: ' + str((sum(f_decisionTree) / folds) * 100))
print('Accuracy of Nearest Neighbor: ' + str((sum(f_knn) / folds) * 100))

# Report whether the difference is significant.
difference = list()
for fold in range(folds):
    difference.append(f_knn[fold] - f_decisionTree[fold])
diff_average = sum(difference) / folds
diff_variable = 0
for fold in range(folds):
    diff_variable += (difference[fold] - diff_average)**2 / (folds-fold)
value = diff_average / ((diff_variable**0.5) / (folds**0.5))

print('value = '+str(value))

criticalValue = 2.776
if value >criticalValue:
    print("Significant")
else:
    print("Insignificant")


