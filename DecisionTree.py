import math

__author__ = 'Victor'


# Compute the entropy of a data set with p positive examples and n negative examples
def entropy(p, n):
    fp = 0 if p == 0 else p / (p+n)
    fn = 0 if n == 0 else n / (p+n)
    plog = 0 if fp == 0 else fp*math.log(fp, 2)
    nlog = 0 if fn == 0 else fn*math.log(fn, 2)
    return -plog - nlog


print(entropy(0, 10))


# Compute the information gain if we split a dataset with p positives and n negatives into two datasets
# (one with a p_true and one with a n_true, one with a p_false and n_false)
def gain(p, n, p_true, n_true, p_false, n_false):
    f_true = (p_true + n_true) / (p+n)
    f_false = (p_false + n_false) / (p+n)
    e = entropy(p, n)
    e_true = entropy(p_true, n_true)
    e_false = entropy(p_false, n_false)
    return e - f_true*e_true - f_false*e_false

p = 5
n = 5
p_true = 3
n_true = 1
p_false = 1
n_false = 4
print(gain(p, n, p_true, n_true, p_false, n_false))
