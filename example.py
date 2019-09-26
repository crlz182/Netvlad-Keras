import cv2
import numpy as np
import tensorflow as tf
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets
from keras.models import Sequential, load_model
from keras.backend import l2_normalize, expand_dims
from netvlad_keras import NetVLADModel
from netvladlayer import NetVLAD
from keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances


#build model and load weights
#note: loading the whole model including the weights is currentlty not supported 
model = NetVLADModel()
model.load_weights('netvlad_weights.h5')
model.build()


#loading sample images for the database and a query image
db1 = cv2.imread('db1.jpg').astype(np.float32)
db1 = cv2.cvtColor(db1, cv2.COLOR_BGR2RGB)
db2 = cv2.imread('db2.jpg').astype(np.float32)
db2 = cv2.cvtColor(db2, cv2.COLOR_BGR2RGB)
query = cv2.imread('query.jpg').astype(np.float32)
query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

#resize them
db1 = cv2.resize(db1,(640,360))
db2 = cv2.resize(db2,(640,360))
query = cv2.resize(query,(640,360))

#batch for database
db_batch = np.asarray([db1,db2])
#get output for database
database = model.predict(db_batch)

#same for query
query_batch = np.expand_dims(query, axis=0)
query_descr = model.predict(query_batch)


#nearest neighbor on database with a single candidate
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean').fit(database)

#get closest entry in database for query image
distances, index = nbrs.kneighbors(query_descr)
print(index)

