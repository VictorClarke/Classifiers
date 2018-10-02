from builtins import print

__author__ = 'Victor'

import nltk


nltk.download()

from nltk.corpus import movie_reviews

# Inspect the corpus
print(movie_reviews.fileids())
print(len(movie_reviews.fileids()))
print(movie_reviews.words('neg/cv000_29416.txt'))

# Split into training/testing sets
train_fileids = movie_reviews.fileids()[:500] + movie_reviews.fileids()[1000:1500]
test_fileids = movie_reviews.fileids()[500:1000] + movie_reviews.fileids()[1500:2000]

# Collect all the words from the training examples
# Have vocabulary be a set rather than a list because we don't have duplicate word in words,
# and if you add a duplicate to a set it doesnt add it
vocabulary = set()
# For each training file and for each word in the file
for fileid in train_fileids:
    for word in movie_reviews.words(fileid):
        vocabulary.add(word)


def format_dataset(fileids, featureSet):
    dataset = list()
    for fileid in fileids:
        features = dict()
        for word in featureSet:
            features[word] = word in movie_reviews.words(fileid)
        pos_or_neg = fileid[:3]
        example = (features, pos_or_neg)
        dataset.append(example)
    return dataset

# Get the datasets ready
train_set = format_dataset(train_fileids, vocabulary)
test_set = format_dataset(test_fileids, vocabulary)

# Create some classifiers
from nltk.classify.decisiontree import DecisionTreeClassifier
tree = DecisionTreeClassifier.train(train_set)

from nltk.classify.naivebayes import NaiveBayesClassifier
bayes = NaiveBayesClassifier.train(train_set)

# Test the classifiers
from nltk.classify import accuracy
print("Decision Tree accuracy: ", accuracy(tree, test_set))
print("Naive Bayes accuracy: ", accuracy(bayes, test_set))