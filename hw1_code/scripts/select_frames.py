#!/bin/python
# Randomly select 

import numpy
import os
import sys
import pdb

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} file_list select_ratio output_file".format(sys.argv[0])
        print "file_list -- the list of video names"
        print "select_ratio -- the ratio of frames to be randomly selected from each audio file"
        print "output_file -- path to save the selected frames (feature vectors)"
        exit(1)

    file_list = sys.argv[1]; output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    numpy.random.seed(18877)

    all_vec = []
    names = []
    for line in fread.readlines():
        print(line)
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        if os.path.exists(mfcc_path) == False:
            continue
        array = numpy.genfromtxt(mfcc_path, delimiter=";")
        for x in range(array.shape[0] // 5):
            start = x * 10
            end = (x + 1) * 10
            if end < array.shape[0]:
                all_vec.append(array[start:end])
                names.append(line[:-1])
        '''
        numpy.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]

        for n in xrange(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')
        '''
    numpy.save('tokens', all_vec)
    numpy.save('token_ids', names)
    fwrite.close()

