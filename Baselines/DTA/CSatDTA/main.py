

import numpy as np
import tensorflow as tf
import random as rn

import os
# from tensorflow import keras
import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing import sequence
# from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D

from keras.layers import BatchNormalization
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.layers import Conv2D, GRU

from keras.models import Model
# from keras.utils.vis_utils import plot_model
from keras.utils import plot_model

from init_params import argparser, logging

import sys, pickle, os
import math, json, time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle

from sklearn import preprocessing
from interaction import *
from figplot import *
from atten import *
from evaluation import experiment


#below 3 lines helps to align fixed gpu
os.environ['PYTHONHASHSEED'] = '0,2,3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1" or "0,1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3";


FLAGS = argparser()
FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

logging(str(FLAGS), FLAGS)



def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select



def get_cindex(Y, P):
#     sys.exit()
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair != 0:
        return summ/pair
    else:
        return 0
    

perfmeasure = get_cindex


    
experiment(FLAGS, perfmeasure)

