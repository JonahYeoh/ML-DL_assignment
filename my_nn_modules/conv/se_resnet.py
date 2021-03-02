import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import add, Input, Concatenate, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, ReLU, multiply, BatchNormalization

from my_nn_modules.fcl.dense import _fully_connected_layer
from my_nn_modules.conv.fundamental import _conv2d

def _se_block(in_tensor, filters, d=8):
    bn = BatchNormalization()(in_tensor)
    squeeze = GlobalAveragePooling2D()(bn)
    fully_connected = ReLU()(Dense(d)(squeeze))
    fully_connected = ReLU()(Dense(filters)(fully_connected))
    sig_activated = tf.keras.activations.sigmoid(fully_connected)
    return multiply([bn, sig_activated]) # excited

# can be simplified breaking into with and without bottle_neck
def _se_res_block(in_tensor, filters, d=8, downsample=True, bottle_neck=True):
    res = in_tensor
    strides = 1
    if downsample:
        strides = 2

    new_filters = filters
    if bottle_neck:
        new_filters = filters * 4
    # adjust res's dimension
    if in_tensor.shape[3] != new_filters:
        res = _conv2d(in_tensor, new_filters, 1, strides)
    elif downsample:
        res = _conv2d(in_tensor, new_filters, 1, strides)
    
    conv = _conv2d(in_tensor, filters, 1, strides)
    conv = _conv2d(conv, filters, 3, 1)
    if bottle_neck:
        conv = _conv2d(conv, new_filters, 1, 1)
    conv = _se_block(conv, new_filters, d) # more variant available
    summation = add([conv, res])
    return ReLU()(summation)

def _pre_block(in_tensor, filters=64):
    conv = _conv2d(in_tensor, filters, 7, 2)
    return MaxPooling2D(pool_size=2, strides=2, padding='same')(conv)
    
def _post_block(in_tensor, n_classes):
    pool = GlobalAveragePooling2D()(in_tensor)
    return _fully_connected_layer(pool, n_classes)

def _convx(in_tensor, filters, n_times, d=8, downsample=True, bottle_neck=True):
    res = _se_res_block(in_tensor, filters, d, downsample, bottle_neck)
    for i in range(n_times-1):
        res = _se_res_block(res, filters, bottle_neck=bottle_neck)
    return res

def _se_resnet(in_shape=(224, 224, 3), n_classes=100, opt='adam', units_by_block=[64, 128, 512, 256], n_convx=[3, 4, 4, 4], bottle_neck=False, d_by_block=[]):
    X_in = Input(in_shape)
    X = _pre_block(X_in)
    resblock_1 = _convx(X, units_by_block[0], n_convx[0], d_by_block[0], False, bottle_neck)
    resblock_2 = _convx(resblock_1, units_by_block[1], n_convx[1], d_by_block[1], bottle_neck=bottle_neck)
    resblock_3 = _convx(resblock_2, units_by_block[2], n_convx[2], d_by_block[2], bottle_neck=bottle_neck)
    resblock_4 = _convx(resblock_3, units_by_block[3], n_convx[3], d_by_block[3], bottle_neck=bottle_neck)
    Y_out = _post_block(resblock_4, n_classes)
    model = Model(X_in, Y_out)
    return model

def se_resnet(in_shape=(224, 224, 3), n_classes=100, opt='adam', units_by_block=[64, 128, 512, 256], n_convx=[3, 4, 4, 4], bottle_neck=False, r=8):
    return _se_resnet(in_shape, n_classes, opt, units_by_block, n_convx, bottle_neck, [int(kernels/r) for kernels in units_by_block])

if __name__ == '__main__':
    model = se_resnet()
    print(model.summary())
