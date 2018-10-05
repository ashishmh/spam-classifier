#!/usr/bin/env python
# coding: utf-8
import numpy as np
import glob, sys, time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer


class Classifier:
    
    HAM = 0
    SPAM = 1

    def __init__(self, ham_list, spam_list, f_train = None):
        self.ham_list = ham_list # file list of ham emails
        self.spam_list = spam_list # file list of spam emails

        self.N_HAM = np.size(ham_list) # number of total ham emails
        self.N_SPAM = np.size(spam_list) # number of total spam emails
        self.N = np.asarray([self.N_HAM, self.N_SPAM])
        self.label = np.asarray([self.HAM]* self.N_HAM + [self.SPAM]* self.N_SPAM)
        
        # container for vocabulary list
        self.vocab = None
        self.nvocab = 0
        
        self.f_train = 0.8 if not f_train else f_train # fraction of training in total, default to be 0.8
        self.N_TRAINING = np.asarray([int(np.floor(self.N_HAM * self.f_train)), int(np.floor(self.N_SPAM * self.f_train))]) # number of training docs in ham and spam folder
        self.N_TESTING = self.N - self.N_TRAINING
        self.training_X, self.training_label, self.testing_X, self.testing_label = self.vectorize(self.f_train)
        
        self.result = None

        
#     def nwords(self, X):
#         '''return the number of distinct words in input matrix X by counting the non-empty columns in X
#             parameters:
#                 X: 2d numpy array'''
#         return np.count_nonzero(X.sum(axis = 0))
    
    def vectorize(self, f_train = None):
        start_time = time.time()
        if f_train is not None:
            self.f_train = f_train # else f_train = 0.8 by default
            # [number of ham in training, number of spam in training]
            self.N_TRAINING = np.asarray([int(np.floor(self.N_HAM * self.f_train)),
                                          int(np.floor(self.N_SPAM * self.f_train))])
            # [number of ham in testing, number of spam in testing]
            self.N_TESTING = self.N - self.N_TRAINING
        print('vectorizing the emails...')
        print('%s %% of emails are used for training...' % (self.f_train * 100))

        # word stemming
        pre = CountVectorizer(input = 'filename', decode_error = 'ignore', 
                              token_pattern = u'(?ui)\\b\\w*[a-z]+\\w{3,}\\b', max_df = 0.95, min_df = 5)
        pre_X = pre.fit_transform(self.ham_list[:self.N_TRAINING[self.HAM]] + self.spam_list[:self.N_TRAINING[self.SPAM]]).toarray()
        
        # get the vocabulary list from training data
        prevocab = pre.get_feature_names()
        stemmer = EnglishStemmer()
        stemmed = [stemmer.stem(w) for w in prevocab]
        self.vocab = list(set(stemmed))
        self.nvocab = np.size(self.vocab)
        
        # training data vectorized with our vocabulary
        training = CountVectorizer(input = 'filename', decode_error = 'ignore', vocabulary = self.vocab)
        self.training_X = training.fit_transform(self.ham_list[:self.N_TRAINING[self.HAM]] + self.spam_list[:self.N_TRAINING[self.SPAM]]).toarray()

        # testing data vectorized with our vocabulary
        testing = CountVectorizer(input = 'filename', vocabulary = self.vocab, decode_error = 'ignore')
        self.testing_X = testing.fit_transform(self.ham_list[-self.N_TESTING[self.HAM]:] + self.spam_list[-self.N_TESTING[self.SPAM]:]).toarray()
        
        # create the label arrays
        self.training_label = np.asarray([self.HAM] * self.N_TRAINING[self.HAM] + [self.SPAM] * self.N_TRAINING[self.SPAM])
        self.testing_label = np.asarray([self.HAM] * self.N_TESTING[self.HAM] + [self.SPAM] * self.N_TESTING[self.SPAM])
        
        print('vectorizing done! it took %.2f s' % (time.time() - start_time))
        return self.training_X, self.training_label, self.testing_X, self.testing_label
    
    def get_training(self):
        '''return the input matrix and label for training set
            Output:
                trainging_X, training_label'''
        return self.training_X, self.training_label
    
    def get_testing(self):
        '''return the input matrix and label for testing set
            Output:
                testing_X, testing_label'''
        return self.testing_X, self.testing_label
    
    def get_ham(self, X, label):
        return X[np.where(label == self.HAM)]
    
    def get_spam(self, X, label):
        return X[np.where(label == self.SPAM)]
                
    def accuracy(self, result = None):
        if result is None:
            if self.result is None:
                print('there is no results!')
                return np.nan
            else:
                return np.mean(self.result == self.testing_label) # number of correct predictions / total testing cases
        else:
            return np.mean(result == self.testing_label) # number of correct predictions / total testing cases
            

    def naive_bayes(self, f_train = None):
        if f_train is not None:
            # re-vectorize the data
            self.vectorize(f_train)
        # we use the multinomial naive bayes model from 
        # https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
        def get_prior():
            '''get the prior of for the Naive Bayes method which will be
            [fraction of ham emails in training set, 
            fraction of spam emails in training set]'''
            prior = self.N_TRAINING / self.N_TRAINING.sum()
            return prior

        def get_conditionals():
            '''get the conditionals of for the Naive Bayes method with some smoothing'''
            # split the traning data by label
#             training_ham = self.training_X[:self.N_TRAINING[self.HAM]]
#             training_spam = self.training_X[-self.N_TRAINING[self.SPAM]:]
            training_ham = self.get_ham(self.training_X, self.training_label)
            training_spam = self.get_spam(self.training_X, self.training_label)

            # conditionals with Laplace smoothing
            con_ham = (training_ham.sum(axis = 0) + 1) / (training_ham.sum() + self.nvocab)
            con_spam = (training_spam.sum(axis = 0) + 1) / (training_spam.sum() + self.nvocab)
            conditionals = np.asarray([con_ham, con_spam])
            return conditionals

        print('cross validating...')
        start_time = time.time()

        prior = get_prior()
        conditionals = get_conditionals()
        # start applying labels to our testing data!
        self.result = np.empty(self.N_TESTING.sum()) # the results of our classifier
        for i in np.arange(self.N_TESTING.sum()):
            # use log likelihood for easier calculation
            loglike_ham = np.dot(np.log(conditionals[self.HAM]), self.testing_X[i]) + np.log(prior[self.HAM])
            loglike_spam = np.dot(np.log(conditionals[self.SPAM]), self.testing_X[i]) + np.log(prior[self.SPAM])
            self.result[i] = self.HAM if loglike_ham > loglike_spam else self.SPAM
        print('testing took %.2f s' % (time.time() - start_time))
        return self.result

    def nearest_neighbor(self, f_train = None):
        if f_train != None:
            # re-vectorize the data
            self.vectorize(f_train)

        print('running classifier...')
        start_time = time.time()

        def calculate_l1_distance(train_row, test_row):
            diff_row = np.subtract(train_row, test_row)                           # find element wise difference
            diff_row = np.absolute(diff_row)                                      # take absolute value of differences
            distance = np.sum(diff_row)                                           # sum the distances
            return distance


        def calculate_l2_distance(train_row, test_row):
            diff_row = np.subtract(train_row, test_row)
            diff_row = np.square(diff_row)
            distance = np.sum(diff_row)
            return np.sqrt(distance)


        def calculate_linf_distance(train_row, test_row):
            diff_row = np.subtract(train_row, test_row)
            diff_row = np.absolute(diff_row)
            return np.amax(diff_row)

        predicted_label_l1 = np.empty(shape = (len(self.testing_X), 1), dtype = int)
        predicted_label_l2 = np.empty(shape = (len(self.testing_X), 1), dtype = int)
        predicted_label_linf = np.empty(shape = (len(self.testing_X), 1), dtype = int)
        for test_row, i in zip(self.testing_X, range(len(self.testing_X))):
            row_distance_l1 = np.empty(shape = (len(self.training_X), 1), dtype = int)
            row_distance_l2 = np.empty(shape = (len(self.training_X), 1), dtype = int)
            row_distance_linf = np.empty(shape = (len(self.training_X), 1), dtype = int)
            for train_row, j in zip(self.training_X, range(len(self.training_X))):
                distance_l1 = calculate_l1_distance(train_row, test_row)
                distance_l2 = calculate_l2_distance(train_row, test_row)
                distance_linf = calculate_linf_distance(train_row, test_row)
                row_distance_l1[j] = distance_l1 # array of distances for each test row
                row_distance_l2[j] = distance_l2
                row_distance_linf[j] = distance_linf
                # print("test row:", test_row, "  | label: ", self.testing_X_label[i])
                # print("train row:", train_row, " | label: ", self.training_label[j])
                # print("dist sum: ", distance)
            min_dist_index_l1 = np.argmin(row_distance_l1) # min distance's index in array of distances
            min_dist_index_l2 = np.argmin(row_distance_l2)
            min_dist_index_linf = np.argmin(row_distance_linf)

            predicted_label_l1[i] = self.training_label[min_dist_index_l1]
            predicted_label_l2[i] = self.training_label[min_dist_index_l2]
            predicted_label_linf[i] = self.training_label[min_dist_index_linf]
            # print("-----------------------")
            # print("min dist: ", np.amin(row_distance))
            # print("index of min: ", np.argmin(row_distance))
            # print("predicted label: ", predicted_label[i])
            # print("-----------------------\n")
            self.result = [predicted_label_l1.flatten(), 
                           predicted_label_l2.flatten(), 
                           predicted_label_linf.flatten()]
        print('testing took %.2f s' % (time.time() - start_time))
        return self.result
    
    def decision_tree(self, f_train = None):
        if f_train is not None:
            # re-vectorize the data
            self.vectorize(f_train)
        
        SPLIT = 30
        
        training_ham = self.get_ham(self.training_X, self.training_label)
        training_spam = self.get_spam(self.training_X, self.training_label)

        hmean = training_ham.mean(axis = 0)
        smean = training_spam.mean(axis = 0)

        freq_diff = abs(hmean - smean) # difference of each word freq in ham ans spam
        arg_fdiff = np.flip(np.argsort(freq_diff)) # arg of freq difference in descending order
        arg_list = list(arg_fdiff[np.where(freq_diff > 0)]) # list of indexes where we do the cuts

        class Node:
            def __init__(self, idx):
                self.idx = idx
#                 print('a new node at index %d' % self.idx)
                self.value = None
                self.left = None
                self.right = None

        def get_idx():
            if arg_list:
                idx = arg_list[0]
                arg_list.pop(0)
                return idx
            else:
                print('Reached maximum number of nodes')
                return None

        def build_tree(rows):
            if np.size(rows) == 1:
                node_labels = self.training_label[rows]
                return node_labels
            else:
                new_N = Node(get_idx())
                col = self.training_X[:, new_N.idx] # the column turned into array
                node_col = col[rows]
                node_labels = self.training_label[rows]

                new_N.value = Gini_min(node_col, node_labels) # node value is whatever with lowest gini index
                l_rows = rows[np.where(node_col <= new_N.value)] # cut rows by N.value
                r_rows = rows[np.where(node_col > new_N.value)]
                if np.size(l_rows) == 0:
                    if np.count_nonzero(self.training_label[r_rows] ==self.HAM) / np.size(self.training_label[r_rows]) > 0.5: # fraction ofself.HAM is higher
                        new_N.right =self.HAM
                        new_N.left =self.SPAM
                    else:
                        new_N.right =self.SPAM
                        new_N.left =self.HAM
                    return new_N    
                elif np.size(r_rows) == 0: # if one side has no other data
                    if np.count_nonzero(self.training_label[l_rows] ==self.HAM) / np.size(self.training_label[l_rows]) > 0.5: # fraction ofself.HAM is higher
                        new_N.right =self.SPAM
                        new_N.left =self.HAM
                    else:
                        new_N.right =self.HAM
                        new_N.left =self.SPAM
                    return new_N    
                else:
                    new_N.left = build_tree(l_rows)
                    new_N.right = build_tree(r_rows)
                    return new_N

        def Gini_min(col, c):
            '''return the minimum gini index given an array of word freq and associated label c'''
            # look for the lowest gini value in the column
            c_max = np.max(col) # the largest value
            c_min = np.min(col) # the smallest value/freq in column
            if c_max == c_min: # if the maximun equals the minimum -> all values are equal
                return c_max # that's the value to split
            else: # if not all elements are zero
                split_value = np.linspace(c_min, c_max, num = SPLIT) # the values of diff split
                gini_idx = Gini_index(split_value, col, c) # list of gini idx at diff split
                return gini_idx.argmin() # return the value for minimum gini index

        # Calculate the Gini index for a split dataset
        def Gini_index(cut, col, c):
            lc = c[np.where(col <= cut)] # labels for the left
            rc = c[np.where(col > cut)] # labels for the right

            len_left, len_right = len(lc), len(rc)
            len_total = len_left + len_right

            unc = 0.0
            if len_left == 0:
                unc += 0 # weighted uncertainty
            else:
                l_p_ham = (np.count_nonzero(lc ==self.HAM) / np.size(lc))
                l_p_spam = (np.count_nonzero(lc ==self.SPAM) / np.size(lc))
                unc += (1 - (l_p_ham**2 + l_p_spam**2)) * len_left / len_total

            if len_right == 0:
                unc += 0
            else:
                r_p_ham = (np.count_nonzero(rc ==self.HAM) / np.size(rc))
                r_p_spam = (np.count_nonzero(rc ==self.SPAM) / np.size(rc))
                unc += (1 - (r_p_ham**2 + r_p_spam**2)) * len_right / len_total

            return unc
        Gini_index = np.vectorize(Gini_index, excluded = [1, 2])
        
        def classify(case, N):
            if N == self.HAM:
                return self.HAM
            elif N == self.SPAM:
                return self.SPAM
            else:
                if case[N.idx] < N.value:
                    return classify(case, N.left)
                else:
                    return classify(case, N.right)
            
        root = build_tree(np.arange(self.training_X.shape[0]))
        self.result = np.empty(self.N_TESTING.sum())
        for i in np.arange(np.size(self.testing_label)):
            self.result[i] = classify(self.testing_X[i], root)
        return self.result

# How to use the classifier:
# Example: 
# test = Classifier(HAM_LIST, SPAM_LIST) # initialize
# result = test.naive_bayes()
# test.accuracy()