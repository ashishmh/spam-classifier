#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob, sys, time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer


# In[2]:


FOLDER = '../enron1/'
HAM_FOLDER = 'ham/'
SPAM_FOLDER = 'spam/'

HAM = 0
SPAM = 1

HAM_LIST = glob.glob(FOLDER + HAM_FOLDER + '*.txt')
SPAM_LIST = glob.glob(FOLDER + SPAM_FOLDER + '*.txt')


# In[44]:


class Classifier:

    def __init__(self, ham_list, spam_list):
        self.ham_list = ham_list
        self.spam_list = spam_list
        self.email_list = ham_list + spam_list
        self.N_HAM = np.size(ham_list)
        self.N_SPAM = np.size(spam_list)
        self.N = np.asarray([self.N_HAM, self.N_SPAM])
        self.label = np.asarray([HAM]* self.N_HAM + [SPAM]* self.N_SPAM)
        
        # number of training docs in ham and spam folder
        # default to be 80% of total
        self.p_training = 0.8
        self.N_TRAINING = np.asarray([int(np.floor(self.N_HAM * 0.8)), int(np.floor(self.N_SPAM * 0.8))])
        self.N_TESTING = self.N - self.N_TRAINING
        
        self.training_X = None # vectorized training data
        self.training_label = None
        self.testing_X = None # vectorized testing data
        self.testing_label = None
        self.result = []
        
        # container for vocabulary list
        self.vocab = []
        self.nvocab = 0
        
    def nwords(self, X):
        '''return the number of distinct words in input matrix X by counting the non-empty columns in X
            parameters:
                X: 2d numpy array'''
        return np.count_nonzero(X.sum(axis = 0))
    
    def vectorize(self, p_training = None):
        print('vectorizing the emails...')
        start_time = time.time()
        
        if p_training != None:
            self.p_training = p_training
        # else p_training = 0.8 by default
        
        # [number of ham in training, number of spam in training]
        self.N_TRAINING = np.asarray([int(np.floor(self.N_HAM * self.p_training)),
                                      int(np.floor(self.N_SPAM * self.p_training))])
        # [number of ham in testing, number of spam in testing]
        self.N_TESTING = self.N - self.N_TRAINING
        
        # word stemming
        stemmer = EnglishStemmer()
        analyzer = CountVectorizer(input = 'filename', decode_error = 'ignore').build_analyzer()
        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))
        
        training = CountVectorizer(input = 'filename', decode_error = 'ignore', analyzer = stemmed_words, 
                                   max_df = 0.95, min_df = 5)
        self.training_X = training.fit_transform(self.ham_list[:self.N_TRAINING[HAM]] + self.spam_list[:self.N_TRAINING[SPAM]]).toarray()
        
        # get the vocabulary list from training data
        self.vocab = training.get_feature_names()
        self.nvocab = np.size(self.vocab)

        testing = CountVectorizer(input = 'filename', vocabulary = self.vocab, decode_error = 'ignore')
        self.testing_X = testing.fit_transform(self.ham_list[-self.N_TESTING[HAM]:] + self.spam_list[-self.N_TESTING[SPAM]:]).toarray()
        
        # create the label arrays
        self.training_label = np.asarray([HAM] * self.N_TRAINING[HAM] + [SPAM] * self.N_TRAINING[SPAM])
        self.testing_label = np.asarray([HAM] * self.N_TESTING[HAM] + [SPAM] * self.N_TESTING[SPAM])
        
        print('vectorizing took %.2f s' % (time.time() - start_time))
        return self.training_X, self.training_label, self.testing_X, self.testing_label
    
    def get_ham(self, X, label):
        return X[np.where(label == HAM)]
    
    def get_spam(self, X, label):
        return X[np.where(label == SPAM)]
                
    def accuracy(self, result = None):
        if np.size(result) > 1:
            return np.mean(self.result == self.testing_label) # number of correct predictions / total testing cases
        else:
            print('there is no results!')
            return 0

    def naive_bayes(self, p_training = None):
        if p_training != None:
            # re-vectorize the data
            self.vectorize(p_training)
        # we use the multinomial naive bayes model from 
        # https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
        def get_prior():
            '''get the prior of for the Naive Bayes method which will be
            [fraction of ham emails, fraction of spam emails]'''
            prior = self.N_TRAINING / self.N_TRAINING.sum()
            return prior

        def get_conditionals():
            # split the traning data by label
            training_ham = self.training_X[:self.N_TRAINING[HAM]]
            training_spam = self.training_X[-self.N_TRAINING[SPAM]:]

            # conditionals with Laplace smoothing
            con_ham = (training_ham.sum(axis = 0) + 1) / (self.nwords(training_ham) + self.nvocab)
            con_spam = (training_spam.sum(axis = 0) + 1) / (self.nwords(training_spam) + self.nvocab)
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
            loglike_ham = np.dot(np.log(conditionals[HAM]), self.testing_X[i]) + np.log(prior[HAM])
            loglike_spam = np.dot(np.log(conditionals[SPAM]), self.testing_X[i]) + np.log(prior[SPAM])
            self.result[i] = HAM if loglike_ham > loglike_spam else SPAM
        print('testing took %.2f s' % (time.time() - start_time))
        return self.result

    def nearest_neighbor(self, p_training = None):
        if p_training != None:
            # re-vectorize the data
            self.vectorize(p_training)

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
                row_distance_l1[j] = distance_l1                              # array of distances for each test row
                row_distance_l2[j] = distance_l2
                row_distance_linf[j] = distance_linf
                # print("test row:", test_row, "  | label: ", self.testing_X_label[i])
                # print("train row:", train_row, " | label: ", self.training_label[j])
                # print("dist sum: ", distance)
            min_dist_index_l1 = np.argmin(row_distance_l1)                    # min distance's index in array of distances
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
            self.result = [predicted_label_l1, predicted_label_l2, predicted_label_linf]
        print('testing took %.2f s' % (time.time() - start_time))
        return self.result


# In[45]:


test = Classifier(HAM_LIST, SPAM_LIST)


# In[21]:


# train_arr, train_arr_label, test_arr, test_arr_label = test.vectorize(0.8)


# In[25]:


result = test.naive_bayes(0.9)


# In[46]:


result = test.nearest_neighbor(0.9)


# In[50]:


test.accuracy(result[1].flatten())


# In[52]:


np.shape(result[0])


# In[56]:


test.N_TESTING.sum()


# In[57]:


test.nvocab


# In[ ]:




