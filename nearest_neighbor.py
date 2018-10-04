import sys
import numpy as np
import math
import time


def nearest_neighbor(train_arr, train_arr_label, test_arr):
    start_time = time.time()
    predicted_label_l1 = np.empty(shape=(len(test_arr), 1), dtype=int)
    predicted_label_l2 = np.empty(shape=(len(test_arr), 1), dtype=int)
    predicted_label_linf = np.empty(shape=(len(test_arr), 1), dtype=int)
    for test_row, i in zip(test_arr, range(len(test_arr))):
        row_distance_l1 = np.empty(shape=(len(train_arr), 1), dtype=int)
        row_distance_l2 = np.empty(shape=(len(train_arr), 1), dtype=int)
        row_distance_linf = np.empty(shape=(len(train_arr), 1), dtype=int)
        for train_row, j in zip(train_arr, range(len(train_arr))):
            distance_l1 = calculate_l1_distance(train_row, test_row)
            distance_l2 = calculate_l2_distance(train_row, test_row)
            distance_linf = calculate_linf_distance(train_row, test_row)
            row_distance_l1[j] = distance_l1                              # array of distances for each test row
            row_distance_l2[j] = distance_l2
            row_distance_linf[j] = distance_linf
            # print("test row:", test_row, "  | label: ", test_arr_label[i])
            # print("train row:", train_row, " | label: ", train_arr_label[j])
            # print("dist sum: ", distance)
        min_dist_index_l1 = np.argmin(row_distance_l1)                    # min distance's index in array of distances
        min_dist_index_l2 = np.argmin(row_distance_l2)
        min_dist_index_linf = np.argmin(row_distance_linf)

        predicted_label_l1[i] = train_arr_label[min_dist_index_l1]
        predicted_label_l2[i] = train_arr_label[min_dist_index_l2]
        predicted_label_linf[i] = train_arr_label[min_dist_index_linf]
        # print("-----------------------")
        # print("min dist: ", np.amin(row_distance))
        # print("index of min: ", np.argmin(row_distance))
        # print("predicted label: ", predicted_label[i])
        # print("-----------------------\n")
    return [predicted_label_l1, predicted_label_l2, predicted_label_linf], round(time.time() - start_time, 2)


def calculate_l1_distance(train_row, test_row):
    diff_row = np.subtract(train_row, test_row)                           # find element wise difference
    diff_row = np.absolute(diff_row)                                      # take absolute value of differences
    distance = np.sum(diff_row)                                           # sum the distances
    return distance


def calculate_l2_distance(train_row, test_row):
    diff_row = np.subtract(train_row, test_row)
    diff_row = np.square(diff_row)
    distance = np.sum(diff_row)
    return math.sqrt(distance)


def calculate_linf_distance(train_row, test_row):
    diff_row = np.subtract(train_row, test_row)
    diff_row = np.absolute(diff_row)
    return np.amax(diff_row)


def test_accuracy(predicted_label, test_arr_label):
    sum = 0
    for predicted, actual in zip(predicted_label, test_arr_label):
        if np.asscalar(predicted) == np.asscalar(actual):
            sum += 1
    print("Accuracy: ", round((sum * 100) / len(test_arr_label), 2), "%")


def main(train_arr, train_arr_label, test_arr, test_arr_label):
    print("Running nearest neighbor classifiers...")
    predicted_label, time_taken = nearest_neighbor(train_arr, train_arr_label, test_arr)

    print("\nNearest neighbor L1")
    test_accuracy(predicted_label[0], test_arr_label)
    print("\nNearest neighbor L2")
    test_accuracy(predicted_label[1], test_arr_label)
    print("\nNearest neighbor L-inf")
    test_accuracy(predicted_label[2], test_arr_label)

    print("Total time taken: {0} second !".format(time_taken))


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
