from keras import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, Activation, Input, concatenate, Lambda, ZeroPadding2D, MaxPooling2D, Layer, Flatten
from keras.optimizers import SGD
from keras.backend import l2_normalize, expand_dims, variable, constant
import cv2, numpy as np
from netvladlayer import NetVLAD
import keras
from keras import regularizers

def NetVLADModel(outputsize = 4096,input_shape=(None,None,3)):

    model = Sequential()

    model.add(SubstractAverage(input_shape = input_shape))  #0
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))    #1
    model.add(Conv2D(64, (3, 3), padding="same"))   #2
    model.add(MaxPooling2D(strides=(2,2)))  #3
    model.add(Activation('relu'))   #4

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))   #5
    model.add(Conv2D(128, (3, 3), padding="same"))  #6
    model.add(MaxPooling2D(strides=(2,2)))  #7
    model.add(Activation('relu'))   #8

    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))   #9
    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))   #10
    model.add(Conv2D(256, (3, 3), padding="same"))     #11
    model.add(MaxPooling2D(strides=(2,2)))  #12
    model.add(Activation('relu'))   #13

    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #14
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #15
    model.add(Conv2D(512, (3, 3), padding="same"))  #16
    model.add(MaxPooling2D(strides=(2,2)))  #17
    model.add(Activation('relu'))   #18

    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #19
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #20
    model.add(Conv2D(512, (3, 3), padding="same"))  #21


    model.add(Lambda(lambda a: l2_normalize(a,axis=-1)))    #22

    model.add(NetVLAD(num_clusters=64)) #23


    #PCA
    model.add(Lambda(lambda a: expand_dims(a,axis=1)))  #24
    model.add(Lambda(lambda a: expand_dims(a,axis=1)))  #25
    model.add(Conv2D(4096,(1,1)))   #26
    model.add(Flatten())    #27
    model.add(Lambda(lambda a: l2_normalize(a,axis=-1)))    #28
    
    sgd = SGD(lr=0.001, decay=0.001, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

"""Initializes pre-trained average RGB
"""
def average_rgb_init(shape, dtype=None):
    return np.array([123.68, 116.779, 103.939])

class SubstractAverage(Layer):
    """Custom layer for subtracting a tensor from another
    """
    def __init__(self, **kwargs):
        super(SubstractAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.average_rgb = self.add_weight(name='average_rgb',
                                    initializer=average_rgb_init,
                                    shape=(3,),
                                    dtype='float32',
                                    trainable=False)
        
        super(SubstractAverage, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        #subtract
        v = inputs - self.average_rgb
        return v

if __name__ == "__main__":
    m = NetVLADModel()
    m.summary()
