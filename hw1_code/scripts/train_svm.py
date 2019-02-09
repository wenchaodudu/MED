#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import pdb
import scipy

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
        pos = []
        neg = []
        for line in train_labels:
            name, l = line.split()
            if l == event_name:
                pos.append(name)
                labels[name] = 1
            else:
                neg.append(name)
                labels[name] = 0
        return labels, pos, neg
    train_labels, pos_labels, neg_labels = get_labels('../all_trn.lst')
    val_labels = get_labels('../all_val.lst')
    feature = np.load('{}_feat.npy'.format(feat_name))
    ids = np.load('{}_id.npy'.format(feat_name)).tolist()
    train_ids = train_labels.keys()
    idx_select = [ids.index(i) for i in train_ids]
    train_features = feature[idx_select]
    pos_features = feature[[ids.index(i) for i in pos_labels]]
    neg_features = feature[[ids.index(i) for i in neg_labels]]
    statistic, pvalue = scipy.stats.ttest_ind(pos_features, neg_features)
    #feat_select = np.where(pvalue < .1)[0]
    feat_select = np.argsort(pvalue)[:10]
    #train_features = np.log(train_features + 1)
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    #train_features = (train_features - mean) / std
    _train_labels = [train_labels[i] for i in train_ids]
    #model = SVC(kernel='rbf', probability=True, class_weight={1: 100, 0:1})
    model = SVC(kernel='rbf', probability=True)
    #model.fit(train_features[:, feat_select], _train_labels)
    model.fit(train_features, _train_labels)
    cPickle.dump([model, feat_select], open(output_file, 'wb'))

    print 'SVM trained successfully for event %s!' % (event_name)
