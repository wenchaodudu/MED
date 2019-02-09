#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import pdb
import csv

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) < 0:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    '''
    model_file = sys.argv[1]
    feat_name = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    '''

    model1 = cPickle.load(open('asr_pred/svm.P001.model', 'rb'))[0]
    #model2 = cPickle.load(open('asr_pred/svm.P002.model', 'rb'))[0]
    model2 = cPickle.load(open('mfcc_pred/svm.P002.model', 'rb'))[0]
    model3 = cPickle.load(open('asr_pred/svm.P003.model', 'rb'))[0]

    def get_labels(path):
        train_labels = open(path).readlines()
        labels = {}
        for line in train_labels:
            name = line.split()[0]
            labels[name] = 0
        return labels
    train_labels = get_labels('../all_test_fake.lst')
    asr_feature = np.load('asr_feat.npy')
    mfcc_feature = np.load('mfcc_feat.npy')
    asr_ids = np.load('asr_id.npy').tolist()
    mfcc_ids = np.load('mfcc_id.npy').tolist()
    train_ids = train_labels.keys()
    '''
    idx_select = [ids.index(i) for i in train_ids]
    train_features = feature[idx_select]
    model = cPickle.load(open(model_file, 'rb'))
    '''
    writer = csv.writer(open('results.csv', 'w'))
    writer.writerow(['VideoID', 'label'])
    for k in train_ids:
        asr_ind = asr_ids.index(k)
        mfcc_ind = mfcc_ids.index(k)
        asr_f = asr_feature[asr_ind][np.newaxis, :]
        mfcc_f = mfcc_feature[mfcc_ind][np.newaxis, :]
        p1 = model1.predict_proba(asr_f)[0, 1]
        p2 = model2.predict_proba(mfcc_f)[0, 1]
        p3 = model3.predict_proba(asr_f)[0, 1]
        pred = np.argmax([p1, p2, p3]) + 1
        writer.writerow([k, pred])
    
