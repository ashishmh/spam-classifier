import sys
import numpy as np
import time


def nearest_neighbor_l1(train_arr, train_arr_label, test_arr, test_arr_label):
    start_time = time.time()
    # L1 manhattan distance
    predicted_label = np.empty(shape=(len(test_arr), 1), dtype=int)
    for test_row, i in zip(test_arr, range(len(test_arr))):
        row_distance = np.empty(shape=(len(train_arr), 1), dtype=int)
        for train_row, j in zip(train_arr, range(len(train_arr))):
            diff_row = np.subtract(train_row, test_row)                 # find element wise difference
            diff_row = np.absolute(diff_row)                            # take absolute value of differences
            distance = np.sum(diff_row)                                 # sum the distances
            row_distance[j] = distance                                  # array of distances for each test row
            # print("test row:", test_row, "  | label: ", test_arr_label[i])
            # print("train row:", train_row, " | label: ", train_arr_label[j])
            # print("row distances: ", diff_row)
            # print("dist sum: ", distance)
        min_distance_index = np.argmin(row_distance)                    # min distance's index in array of distances
        predicted_label[i] = train_arr_label[min_distance_index]
        # print("-----------------------")
        # print("min dist: ", np.amin(row_distance))
        # print("index of min: ", np.argmin(row_distance))
        # print("predicted label: ", predicted_label[i])
        # print("-----------------------\n")
    test_accuracy(predicted_label, test_arr_label)
    print("Time taken: {0} second !".format(round(time.time() - start_time), 2))


def test_accuracy(predicted_label, test_arr_label):
    sum = 0
    for predicted, actual in zip(predicted_label, test_arr_label):
        if np.asscalar(predicted) == np.asscalar(actual):
            sum += 1
    print("accuracy: ", round((sum * 100) / len(test_arr_label), 2), "%")


def test_classifier():
    print("Testing the classifier using self generated data...")
    # train_ham_data = ['i am ashna aggarwal', 'i am sahil aggarwal', 'ashna lol what here now']
    # train_spam_data = ['cat bat ashish shah', 'cat bat rat', 'who are you cat bat rat now']
    #
    # test_ham_data = ['i am monish godha', 'ashna where are you']
    # test_spam_data = ['i am rat bat', 'rat bat ashna aggarwal']
    # vectorizer = CountVectorizer()
    # train_vector = vectorizer.fit_transform(train_ham_data + train_spam_data)
    # test_vector = vectorizer.transform(test_ham_data + test_spam_data)
