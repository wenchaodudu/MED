#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
from collections import Counter
import sys
import pdb
import nltk

if __name__ == '__main__':
    file_list = sys.argv[1]

    fread = open(file_list,"r")
    file_list = fread.readlines()

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows

    counter = Counter()
    for line in file_list:
        file_name = 'asr/{}.txt'.format(line.strip('\n'))
        if os.path.exists(file_name):
            for trans in open(file_name).readlines():
                words = nltk.word_tokenize(trans.lower())
                counter.update(words)
    word2id = {}
    total = sum(counter[k] for k in counter)
    vocab_size = len(counter)

    word_vectors = {}
    found = 0 
    print("Loading word vecotrs.")
    word2vec_file = open('./glove.6B.300d.txt')
    next(word2vec_file)
    for line in word2vec_file:
	word, vec = line.split(' ', 1)
	if word in counter:
	    word_vectors[word] = numpy.fromstring(vec, dtype=numpy.float32, sep=' ')

    a = .001
    all_vec = []
    name = []
    for line in file_list:
        file_name = 'asr/{}.txt'.format(line.strip('\n'))
        name.append(line[:-1])
        if os.path.exists(file_name):
            words = []
            for trans in open(file_name).readlines():
                words += nltk.word_tokenize(trans.lower())
            try:
                vectors = numpy.vstack([word_vectors[w] for w in words if w in word_vectors])
                weights = [a / (a + counter[w] / float(total)) for w in words if w in word_vectors]
                all_vec.append(numpy.dot(weights, vectors) / len(weights))
            except:
                all_vec.append(numpy.zeros(300))
        else:
            all_vec.append(numpy.zeros(300))
    all_vec = numpy.vstack(all_vec).T
    u, s, v = numpy.linalg.svd(all_vec)
    uu = u[:, 0]
    all_vec -= uu[:, numpy.newaxis] * numpy.dot(uu.T, all_vec)[numpy.newaxis, :]
    numpy.save('asr_feat', all_vec.T)
    numpy.save('asr_id', name)

    print "ASR features generated successfully!"
