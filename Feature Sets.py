__author__ = 'Victor'
# Purpose: compare feature sets for sentiment analysis in movie reviews.

import random
import operator
from builtins import print
from nltk.corpus import movie_reviews
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify import accuracy

# Divide the corpus for training and testing
train_fileids = movie_reviews.fileids()[0:500] + movie_reviews.fileids()[1000:1500]
test_fileids = movie_reviews.fileids()[500:1000] + movie_reviews.fileids()[1500:2000]
fileid_count = 0

# Define a function for formatting classification datasets
def format_dataset(fileids, featureset):
    dataset = list()
    for fileid in fileids:
        fileid_count = fileid_count + 1
        review = set(movie_reviews.words(fileid))
        features = dict()
        for word in featureset:
            features[word] = word in review
        pos_or_neg = fileid[:3]
        dataset.append((features, pos_or_neg))
    return dataset

# Collect all the words in the training examples
vocabulary = set()
for fileid in train_fileids:
    for word in movie_reviews.words(fileid):
        vocabulary.add(word)

# Try a feature set of 500 random words
vocabulary = list(vocabulary)
random.shuffle(vocabulary)
random_featureset = vocabulary[:500]

train_set = format_dataset(train_fileids, random_featureset)
test_set = format_dataset(test_fileids, random_featureset)
bayes = NaiveBayesClassifier.train(train_set)

print("Random words: ", random_featureset)
print("Naive Bayes accuracy:", accuracy(bayes, test_set))


# Try a feature set of the 500 words that appear most often in the training examples
common_words = dict()
for fileid in train_fileids:
    for word in movie_reviews.words(fileid):
        if word not in common_words:
            common_words[word] = 1
        else:
            word = word + 1

sorted_common = sorted(common_words.items(), key=operator.itemgetter(1))[fileid_count-500:fileid_count]
train_set = format_dataset(train_fileids, sorted_common)
test_set = format_dataset(test_fileids, sorted_common)
bayes = NaiveBayesClassifier.train(train_set)

print("Most Common 500 Words: ", sorted_common)
print("Naive Bayes accuracy:", accuracy(bayes, test_set))
