#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
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


# In[3]:


class naive_bayes:

    def __init__(self, ham_list, spam_list):
        self.ham_list = ham_list
        self.spam_list = spam_list
#         self.email_list = ham_list + spam_list
        self.N_HAM = np.size(ham_list)
        self.N_SPAM = np.size(spam_list)
        self.N = np.asarray([self.N_HAM, self.N_SPAM])
#         self.labels = np.asarray([HAM]* self.N_HAM + [SPAM]* self.N_SPAM)
        
        # number of training docs in ham and spam folder
        # default to be 80% of total
        self.p_training = 0.8
#         self.N_TRAINING = np.asarray([int(np.floor(self.N_HAM * 0.8)), int(np.floor(self.N_SPAM * 0.8))])
        self.training_X = None
        self.training_label = None
        self.testing_X = None
        self.testing_label = None
        
        # container for vocabulary list
        self.vocab = []
        self.nvocab = 0
        
    def nwords(self, X):
        '''return the number of distinct words in input matrix X
            parameters:
                X: 2d numpy array'''
        return np.count_nonzero(X.sum(axis = 0))
    
    def get_training(self, p_training = None):
        if p_training != None:
            self.p_training = p_training
        
        # [number of ham in training, number of spam in training]
        N_TRAINING = np.asarray([int(np.floor(self.N_HAM * 0.8)), int(np.floor(self.N_SPAM * 0.8))])
        
            
        return training_X, training_labels, testing_X, testing_labels
    
    def start_training(self, P_TRAINING = None):
        # update the number of training docs
        if P_TRAINING == None:
            N_TRAINING = self.N_TRAINING
        else:
            self.N_TRAINING = np.asarray([int(np.floor(self.N_HAM * P_TRAINING)), 
                                          int(np.floor(self.N_SPAM * P_TRAINING))])
            N_TRAINING = self.N_TRAINING
            
        # create the input matrix from training data
        # word stemming
        stemmer = EnglishStemmer()
        analyzer = CountVectorizer(input = 'filename', decode_error = 'ignore').build_analyzer()
        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))
        
        training = CountVectorizer(input = 'filename', decode_error = 'ignore', analyzer = stemmed_words, 
                                   max_df = 0.95, min_df = 5)
        training_X = training.fit_transform(self.ham_list[:N_TRAINING[HAM]] + self.spam_list[:N_TRAINING[SPAM]]).toarray()
        
        # get the vocabulary list from training data
        self.vocab = training.get_feature_names()
        self.nvocab = np.size(self.vocab)
        
        # split the traning data by label
        training_ham = training_X[:N_TRAINING[HAM]]
        training_spam = training_X[-N_TRAINING[SPAM]:]
        
        # we use the multinomial naive bayes model from 
        # https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
        prior = N_TRAINING / N_TRAINING.sum()
        
        # conditionals with Laplace smoothing
        con_ham = (training_ham.sum(axis = 0) + 1) / (self.nwords(training_ham) + self.nvocab)
        con_spam = (training_spam.sum(axis = 0) + 1) / (self.nwords(training_spam) + self.nvocab)
        conditionals = np.asarray([con_ham, con_spam])
        return prior, conditionals
        
    def classifier(self, prior, conditionals):
        # the number of training docs
        N_TESTING = self.N - self.N_TRAINING
        
        # create the input matrix from testing data and the vocabulary list from training data
        testing = CountVectorizer(input = 'filename', vocabulary = self.vocab, decode_error = 'ignore')
        testing_X = testing.fit_transform(self.ham_list[-N_TESTING[HAM]:] + self.spam_list[-N_TESTING[SPAM]:]).toarray()
        
        # split the traning data by their (real) label
        testing_ham = testing_X[:N_TESTING[HAM]]
        testing_spam = testing_X[-N_TESTING[SPAM]:]
        
        # start applying labels to our testing data!
        results = np.empty(N_TESTING.sum()) # the results of our classifier
        for i in np.arange(N_TESTING.sum()):
            # use log likelihood for easier calculation
            loglike_ham = np.dot(np.log(conditionals[HAM]), testing_X[i]) + np.log(prior[HAM])
            loglike_spam = np.dot(np.log(conditionals[SPAM]), testing_X[i]) + np.log(prior[SPAM])
            results[i] = HAM if loglike_ham > loglike_spam else SPAM
        return results
                
    def accuracy(self, results):
        N_TESTING = self.N - self.N_TRAINING
        
        # creating array of the real labels
        test_label = np.asarray([HAM] * N_TESTING[HAM] + [SPAM] * N_TESTING[SPAM])
        return np.mean(results == test_label) # number of correct predictions / total testing cases


# In[6]:


test = naive_bayes(HAM_LIST, SPAM_LIST)
prior, conditionals = test.start_training(0.7)
results = test.classifier(prior, conditionals)
test.accuracy(results)


# In[ ]:




