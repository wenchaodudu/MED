#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import pdb

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_name = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    def get_labels(path):
        train_labels = open(path).readlines()
        labels = {}
        for line in train_labels:
            name, l = line.split()
            labels[name] = 1 if l == event_name else 0
        return labels
    train_labels = get_labels('../all_trn.lst')
    val_labels = get_labels('../all_val.lst')
    feature = np.load('{}_feat.npy'.format(feat_name))
    ids = np.load('{}_id.npy'.format(feat_name)).tolist()
    train_ids = train_labels.keys()
    idx_select = [ids.index(i) for i in train_ids]
    train_features = feature[idx_select]
    _train_labels = [train_labels[i] for i in train_ids]
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_features, _train_labels)
    cPickle.dump(model, open(output_file, 'wb'))

    print 'SVM trained successfully for event %s!' % (event_name)
