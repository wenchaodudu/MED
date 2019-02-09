#!/bin/python 

import numpy
import os
from sklearn.cluster import KMeans
import cPickle
import sys
import pdb

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    #mfcc_csv_file = sys.argv[1]; 
    output_file = sys.argv[3]
    data = numpy.load('tokens.npy').reshape(-1, 390)
    cluster_num = int(sys.argv[2])
    model = KMeans(n_clusters=cluster_num, n_init=1, max_iter=150, verbose=2)
    model.fit(data)
    cPickle.dump(model, open(output_file, 'wb'))

    print "K-means trained successfully!"
    
