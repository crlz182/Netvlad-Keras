import cv2
import numpy as np
import tensorflow as tf

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets
from keras.models import Sequential
import netvlad_keras
import os

# This script loads the tensorflow model and transforms it into a Keras model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

checkpoint = nets.defaultCheckpoint()
# start tensorflow session
with tf.Session() as sess:

        # import graph
        saver = tf.train.import_meta_graph(checkpoint+".meta")
                
        # load weights for graph
        saver.restore(sess, checkpoint)

        # get all global variables (including model variables)
        vars_global = tf.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        for var in vars_global:
                try:
                        model_vars[var.name] = var.eval()
                except:
                        print("For var={}, an exception occurred".format(var.name))

#use cpu here to not run out of memory
with tf.device('/cpu:0'):
        model = netvlad_keras.NetVLADModel()
        model.layers[1].set_weights([model_vars['vgg16_netvlad_pca/conv1_1/kernel:0'], model_vars['vgg16_netvlad_pca/conv1_1/bias:0']])
        model.layers[2].set_weights([model_vars['vgg16_netvlad_pca/conv1_2/kernel:0'], model_vars['vgg16_netvlad_pca/conv1_2/bias:0']])
        #3: pooling
        #4: activation
        model.layers[5].set_weights([model_vars['vgg16_netvlad_pca/conv2_1/kernel:0'], model_vars['vgg16_netvlad_pca/conv2_1/bias:0']])
        model.layers[6].set_weights([model_vars['vgg16_netvlad_pca/conv2_2/kernel:0'], model_vars['vgg16_netvlad_pca/conv2_2/bias:0']])
        #7: pooling
        #8: activation
        model.layers[9].set_weights([model_vars['vgg16_netvlad_pca/conv3_1/kernel:0'], model_vars['vgg16_netvlad_pca/conv3_1/bias:0']])
        model.layers[10].set_weights([model_vars['vgg16_netvlad_pca/conv3_2/kernel:0'], model_vars['vgg16_netvlad_pca/conv3_2/bias:0']])
        model.layers[11].set_weights([model_vars['vgg16_netvlad_pca/conv3_3/kernel:0'], model_vars['vgg16_netvlad_pca/conv3_3/bias:0']])
        #12: pooling
        #13: activation
        model.layers[14].set_weights([model_vars['vgg16_netvlad_pca/conv4_1/kernel:0'], model_vars['vgg16_netvlad_pca/conv4_1/bias:0']])
        model.layers[15].set_weights([model_vars['vgg16_netvlad_pca/conv4_2/kernel:0'], model_vars['vgg16_netvlad_pca/conv4_2/bias:0']])
        model.layers[16].set_weights([model_vars['vgg16_netvlad_pca/conv4_3/kernel:0'], model_vars['vgg16_netvlad_pca/conv4_3/bias:0']])
        #17: pooling
        #18: activation
        model.layers[19].set_weights([model_vars['vgg16_netvlad_pca/conv5_1/kernel:0'], model_vars['vgg16_netvlad_pca/conv5_1/bias:0']])
        model.layers[20].set_weights([model_vars['vgg16_netvlad_pca/conv5_2/kernel:0'], model_vars['vgg16_netvlad_pca/conv5_2/bias:0']])
        model.layers[21].set_weights([model_vars['vgg16_netvlad_pca/conv5_3/kernel:0'], model_vars['vgg16_netvlad_pca/conv5_3/bias:0']])
        #22: lambda(l2 normalize)

        model.layers[23].set_weights([model_vars['vgg16_netvlad_pca/cluster_centers:0'], model_vars['vgg16_netvlad_pca/assignment/kernel:0']])
        #24 and 25: lambda (expand dims)
        model.layers[26].set_weights([model_vars['vgg16_netvlad_pca/WPCA/kernel:0'], model_vars['vgg16_netvlad_pca/WPCA/bias:0']])

        model.build()
        model.summary()
        model.save('netvlad_model.h5', include_optimizer=False)
        model.save_weights('netvlad_weights.h5')


