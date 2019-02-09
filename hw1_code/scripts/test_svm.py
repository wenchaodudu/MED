#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import pdb

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_name = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    def get_labels(path):
        train_labels = open(path).readlines()
        labels = {}
        for line in train_labels:
            name, l = line.split()
            labels[name] = 0
        return labels
    train_labels = get_labels('../all_val.lst')
    feature = np.load('{}_feat.npy'.format(feat_name))
    ids = np.load('{}_id.npy'.format(feat_name)).tolist()
    train_ids = train_labels.keys()
    idx_select = [ids.index(i) for i in train_ids]
    train_features = feature[idx_select]
    model, feat_select = cPickle.load(open(model_file, 'rb'))
    labels = model.predict_proba(train_features)
    with open(output_file, 'w') as out:
        for l in labels:
            out.write(str(l[1]))
            out.write('\n')
        
