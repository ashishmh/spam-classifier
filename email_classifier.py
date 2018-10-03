import os
import sys
import numpy
from pprint import pprint
# import nltk
# nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer


HAM_FOLDER = "ham/"
SPAM_FOLDER = "spam/"
MAX_HAM = 3672
TRAINING_HAM = 3300
MAX_SPAM = 1500
TRAINING_SPAM = 1350


def create_vectorized_vocab(data):
    # sample_data = ["The quick brown fox jumped over the lazy dog.", "ashish is awesome over the lazy dog"]
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
    # print("\nham data: ", len(ham_data))

    spam_data = []
    for file in spam_file_list:
        try:
            content = open(SPAM_FOLDER + file).read()
        except:
            # print("Error opening file: ", SPAM_FOLDER + file)
            pass
        spam_data.append(content)
    # print("\nspam data: ", len(spam_data))

    return ham_data, spam_data


def nearest_neighbor():
    return


def main():
    training_ham_data, training_spam_data = read_data(0, TRAINING_HAM, 0, TRAINING_SPAM)
    test_ham_data, test_spam_data = read_data(TRAINING_HAM, MAX_HAM, TRAINING_SPAM, MAX_SPAM)
    vectorizer = create_vectorized_vocab(training_ham_data + training_spam_data)

    # encode training document
    training_vectors = vectorizer.transform(training_ham_data + training_spam_data)
    # print("vector type: ", type(training_vectors))
    # print("\nvector shape: ", training_vectors.shape)
    # print("vector to array: ", training_vectors.toarray())

    training_vec_arr = training_vectors.toarray()
    # new_arr = numpy.ndarray
    new_arr = numpy.empty((len(training_vec_arr), training_vec_arr[0:1].size + 1), int)
    for row in training_vec_arr:
        # print(type(row))
        print(row.shape)
        # print(len(row))
        print(row)
        new_row = row.reshape(-1)
        print(new_row)
        # numpy.insert(row, len(row), 999)
        # print(len(row))
        # new_row = numpy.insert(row, 0, 999, 0)
        # numpy.insert(new_arr, 0, new_row, 0)
        sys.exit(0)


    # encode test document
    test_vectors = vectorizer.transform(test_ham_data + test_spam_data)


if __name__ == "__main__":
    main()
