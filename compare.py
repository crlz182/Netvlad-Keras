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
from inspect import signature


#in case you only want to use cpu
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

""" tf.reset_default_graph()

image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, nets.defaultCheckpoint()) """

model = NetVLADModel()
model.load_weights('netvlad_weights.h5')
model.build()


referenceDir = "/home/gabercar/Documents/Datasets/Kudamm/Memory"
queryDir = "/home/gabercar/Documents/Datasets/Kudamm/Live"
databaseTf = []
databaseKeras = []

for r, d, f in os.walk(referenceDir):
    databaseTf = [0]*len(f)
    databaseKeras = [0]*len(f)
    for file in sorted(f):
        print(file)
        inim = cv2.imread(os.path.join(r,file))
        inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        inim = cv2.resize(inim,(640,360))
        batch = np.expand_dims(inim, axis=0)
        #resultTf = sess.run(net_out, feed_dict={image_batch: batch})
        resultKeras = model.predict(batch)

        number = int(int(file[5:9]) / 10)
        #databaseTf[number] = resultTf[0]
        databaseKeras[number] = resultKeras[0]



hitsKeras = []
#nbrsTf = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean').fit(databaseTf)
nbrsKeras = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean').fit(databaseKeras)
for r, d, f in os.walk(queryDir):
    hitsTf = np.zeros((len(f),len(f)))
    hitsKeras = np.zeros((len(f),len(f)))
    for file in sorted(f):
        #print(file)
        inim = cv2.imread(os.path.join(r,file))
        inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        inim = cv2.resize(inim,(640,360))
        batch = np.expand_dims(inim, axis=0)
        #resultTf = sess.run(net_out, feed_dict={image_batch: batch})
        resultKeras = model.predict(batch)

        dbNumber = int(int(file[5:9]) / 10)
        """ distances, index = nbrsTf.kneighbors(resultTf)
        hitsTf[dbNumber,index] = 1 """

        distances, index = nbrsKeras.kneighbors(resultKeras)
        hitsKeras[dbNumber,index] = 1
        print("matched reference: "+str(index)+" query: "+str(dbNumber))

""" np.save('matrixtf.npy', hitsTf)
np.save('matrixkeras.npy', hitsKeras) """

""" if np.array_equal(hitsTf,hitsKeras):
    print("Same result") """

