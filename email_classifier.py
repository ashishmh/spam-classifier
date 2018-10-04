import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import nearest_neighbor as nn


HAM_FOLDER = "ham/"
SPAM_FOLDER = "spam/"
MAX_HAM = 3672
TRAINING_HAM = 3300
MAX_SPAM = 1500
TRAINING_SPAM = 1350


def read_data(ham_beg, ham_end, spam_beg, spam_end):
    ham_file_list = os.listdir(HAM_FOLDER)[ham_beg:ham_end]
    spam_file_list = os.listdir(SPAM_FOLDER)[spam_beg:spam_end]
    ham_data = []
    for file in ham_file_list:
        try:
            content = open(HAM_FOLDER + file).read()
        except:
            pass
        ham_data.append(content)

    spam_data = []
    for file in spam_file_list:
        try:
            content = open(SPAM_FOLDER + file).read()
        except:
            pass
        spam_data.append(content)

    return ham_data, spam_data


def add_class_labels(total_num, ham_num, spam_num):
    # ham := 0, spam := 1
    arr_label = np.empty(shape=(total_num, 1), dtype=int)
    for i in range(0, ham_num):
        arr_label[i] = 0
    for i in range(ham_num, ham_num + spam_num):
        arr_label[i] = 1

    return arr_label


def main():
    # read data and split into training and test sets
    train_ham_data, train_spam_data = read_data(0, TRAINING_HAM, 0, TRAINING_SPAM)
    test_ham_data, test_spam_data = read_data(TRAINING_HAM, MAX_HAM, TRAINING_SPAM, MAX_SPAM)

    vectorizer = CountVectorizer()
    # create vocab and encode training document
    train_vector = vectorizer.fit_transform(train_ham_data + train_spam_data)
    print("\nvocab length: ", len(vectorizer.vocabulary_))
    # encode test document
    test_vector = vectorizer.transform(test_ham_data + test_spam_data)

    # add class labels to training data
    train_arr = train_vector.toarray()
    train_arr_label = add_class_labels(len(train_arr), len(train_ham_data), len(train_spam_data))
    # add class labels to test data
    test_arr = test_vector.toarray()
    test_arr_label = add_class_labels(len(test_arr), len(test_ham_data), len(test_spam_data))

    # run nearest neighbor L1 classifier
    nn.nearest_neighbor_l1(train_arr, train_arr_label, test_arr, test_arr_label)


if __name__ == "__main__":
    main()
