import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.backend import l2_normalize, expand_dims
from netvlad_keras import NetVLADModel
from netvladlayer import NetVLAD

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets
from netvlad_keras import NetVLADModel
import os
from keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import keras.backend as K
from keras.models import load_model

# Debugging script for checking if the output of a certain layer is consistent among TF and Keras

layerToCheck = 23
inim = cv2.imread('example.jpg')
inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
inim = cv2.resize(inim,(600,600))
batch = np.expand_dims(inim, axis=0)

tf.reset_default_graph()
image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])
net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, nets.defaultCheckpoint())

model = NetVLADModel()
model.load_weights('netvlad_weights.h5')
w = model.layers[0].get_weights()
model.build()

#results tensorflow (move return statement to correct position in TF implementation)
resultTf = sess.run(net_out, feed_dict={image_batch: batch})

#results keras (here we can get the output of a certain layer with a helper function)
forward_pass_helper = K.function([model.layers[0].input],
                    [model.layers[layerToCheck+1].input])

#pass image through the network once and get output at layer "layerToCheck"
resultKeras = forward_pass_helper([batch])[0]

h = model.layers[23].get_weights()
if np.array_equal(resultKeras,resultTf):
    print("same result for layer "+str(layerToCheck))
else:
    print("failed for layer "+str(layerToCheck))