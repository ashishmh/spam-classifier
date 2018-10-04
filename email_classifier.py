import os
import sys
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer


HAM_FOLDER = "ham/"
SPAM_FOLDER = "spam/"
MAX_HAM = 3672
TRAINING_HAM = 3300
MAX_SPAM = 1500
TRAINING_SPAM = 1350


def create_vectorized_vocab(data):
    # create the transform
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(data)
    # summarize
    # print("\nvocab length: ", len(vectorizer.vocabulary_))
    # print("\nvocab: ", vectorizer.vocabulary_)
    return vectorizer


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


def nearest_neighbor_l1(train_arr, train_arr_label, test_arr, test_arr_label):
    MAX = 100
    # train_arr = train_arr[0:MAX]
    # train_arr_label = train_arr_label[0:MAX]
    test_arr = test_arr[0:MAX]
    test_arr_label = test_arr_label[0:MAX]

    # L1 manhattan distance
    predicted_label = np.empty(shape=(len(test_arr), 1), dtype=int)
    for test_row, i in zip(test_arr, range(len(test_arr))):
        row_distance = np.empty(shape=(len(train_arr), 1), dtype=int)
        for train_row, j in zip(train_arr, range(len(train_arr))):
            diff_row = np.subtract(train_row, test_row)                 # find element wise difference
            diff_row = np.absolute(diff_row)                            # take absolute value of differences
            distance = np.sum(diff_row)                                 # sum the distances
            row_distance[j] = distance                                  # array of distances for each test row
        min_distance_index = np.argmin(row_distance)
        predicted_label[i] = train_arr_label[min_distance_index]
        if i == MAX:
            break
    summer = 0
    for predicted, actual in zip(predicted_label[0:10], test_arr_label[0:10]):
        if np.asscalar(predicted) == np.asscalar(actual):
            summer += 1
    print("accuracy: ", (summer*100)/len(test_arr), "%")
    # print("predicted: ", predicted_label[0:MAX])
    # print("actual: ", test_arr_label[0:MAX])

    return


def main():
    # read date and split into train and test sets
    train_ham_data, train_spam_data = read_data(0, TRAINING_HAM, 0, TRAINING_SPAM)
    test_ham_data, test_spam_data = read_data(TRAINING_HAM, MAX_HAM, TRAINING_SPAM, MAX_SPAM)
    vectorizer = create_vectorized_vocab(train_ham_data + train_spam_data)

    # encode training document
    train_vector = vectorizer.transform(train_ham_data + train_spam_data)
    # encode test document
    test_vector = vectorizer.transform(test_ham_data + test_spam_data)

    # adding class labels to training data
    train_arr = train_vector.toarray()
    train_arr_label = add_class_labels(len(train_arr), len(train_ham_data), len(train_spam_data))
    # adding class labels to test data
    test_arr = test_vector.toarray()
    test_arr_label = add_class_labels(len(test_arr), len(test_ham_data), len(test_spam_data))

    # nearest neighbor classifier
    nearest_neighbor_l1(train_arr, train_arr_label, test_arr, test_arr_label)


if __name__ == "__main__":
    main()
