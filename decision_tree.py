import sys
import numpy as np
import math
import time
import random

def main(train_arr, train_arr_label, test_arr, test_arr_label):
    print("Main")
    data = numpyToPython(train_arr, train_arr_label)
    root = build_tree(data, 15, 0)
    test_data = numpyToPython(test_arr, test_arr_label)
    count = 0;
    for row in test_data:
        prediction = predict(root, row)
        if row[-1] == prediction:
            count += 1
    print('Total Correct=%d, Total test=%d Accuracy=%f' % (count, len(test_arr_label), count/len(test_arr_label)) )

def numpyToPython(data, label):
    result = list()
    for (obj, value) in zip(data, label):
        toadd = obj.tolist()
        toadd.append(np.asscalar(value))
        result.append(toadd)
    return result;


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def getIndex(dataset):
    counts = [0]*len(dataset[0])
    for row in dataset:
        j = 0;
        for i in row:
            if i!=0:
                counts[j]+=1
            j+=1
    return counts.index(max(counts))

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    index = getIndex(dataset)
    for row in dataset:
        groups = test_split(index, row[index], dataset)
        gini = gini_index(groups, class_values)
        if gini < b_score:
            b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    print('Splitting on value %d and index %d' % (b_value, b_index))
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    print('Splitting %d  leftsize %d rightsize %d' % (depth, len(left), len(right)))
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
