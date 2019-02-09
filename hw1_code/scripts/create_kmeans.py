#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import pdb
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))

    tokens = numpy.load('tokens.npy').reshape(-1, 390)
    token_ids = numpy.load('token_ids.npy')
    words = kmeans.predict(tokens)
    vectors = {}
    for line in open('list/all.video').readlines():
        vectors[line[:-1]] = numpy.zeros(cluster_num)
        
    for word, name in zip(words, token_ids):
        vectors[name][word] += 1
    names = vectors.keys()
    feature = [vectors[k] for k in names]
    numpy.save('mfcc_feat', feature)
    numpy.save('mfcc_ids', names)
    

    print "K-means features generated successfully!"
