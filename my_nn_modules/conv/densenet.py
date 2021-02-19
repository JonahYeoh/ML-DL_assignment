import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D

from my_nn_modules.fcl.dense import _fully_connected_layer
from my_nn_modules.conv.fundamental import _conv2d, _transition_layer2d

def _dense_block(in_tensor, filters, bottle_neck=True):
    conv = _conv2d(in_tensor, filters, 1, 1)
    conv = _conv2d(conv, filters, 3, 1)
    if bottle_neck:
        return _conv2d(conv, filters, 1, 1)
    return conv

def _dense_block_output(in_tensor, filters, pooling_method='avg', bottle_neck=True, last_block=False):
    if not bottle_neck:
        return _transition_layer2d(in_tensor, filters, not last_block, 'avg')
    return in_tensor # last block is followed by GlobalPooling

def _pre_block(in_tensor, filters=64):
    conv = _conv2d(in_tensor, filters, 7, 2)
    return MaxPooling2D(pool_size=2, strides=2, padding='same')(conv)
    
def _post_block(in_tensor, n_classes):
    pool = GlobalAveragePooling2D()(in_tensor)
    return _fully_connected_layer(pool, n_classes)

def _dense_block6(in_tensor, filters, bottle_neck=True, last_block=False):
    dense1 = _dense_block(in_tensor, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5])
    last = _dense_block(dense, filters, bottle_neck)
    return _dense_block_output(last, filters, 'avg', bottle_neck, last_block)

def _dense_block12(in_tensor, filters, bottle_neck=True, last_block=False):
    dense1 = _dense_block(in_tensor, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6])
    dense7 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7])
    dense8 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8])
    dense9 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9])
    dense10 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10])
    dense11 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11])
    last = _dense_block(dense, filters, bottle_neck)
    return _dense_block_output(last, filters, 'avg', bottle_neck, last_block)

def _dense_block16(in_tensor, filters, bottle_neck=True, last_block=False):
    dense1 = _dense_block(in_tensor, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6])
    dense7 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7])
    dense8 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8])
    dense9 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9])
    dense10 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10])
    dense11 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11])
    dense12 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12])
    dense13 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13])
    dense14 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14])
    dense15 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15])
    last = _dense_block(dense, filters, bottle_neck)
    return _dense_block_output(last, filters, 'avg', bottle_neck, last_block)

def _dense_block24(in_tensor, filters, bottle_neck=True, last_block=False):
    dense1 = _dense_block(in_tensor, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6])
    dense7 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7])
    dense8 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8])
    dense9 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9])
    dense10 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10])
    dense11 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11])
    dense12 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12])
    dense13 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13])
    dense14 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14])
    dense15 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15])
    dense16 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16])
    dense17 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17])
    dense18 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18])
    dense19 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19])
    dense20 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20])
    dense21 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20, dense21])
    dense22 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20, dense21, dense22])
    dense23 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20, dense21, dense22, dense23])
    last = _dense_block(dense, filters, bottle_neck)    
    return _dense_block_output(last, filters, 'avg', bottle_neck, last_block)

block_fn_dict = { 'db6': ['dense block 6', _dense_block6], 'db12': ['dense block 12', _dense_block12], 'db16': ['dense block 16', _dense_block16], 'db24': ['dense block 24', _dense_block24] }

def get_dict():
    return block_fn_dict

def _densenet(in_shape=(224, 224, 3), n_classes=100, opt='adam', kernels=64, bottle_neck=False, fn_dict=block_fn_dict, block_fn=['db6', 'db12', 'db24', 'db16']):
    X_in = Input(in_shape)
    X = _pre_block(X_in)
    convx1 = fn_dict[block_fn[0]][1](X, kernels, bottle_neck)
    convx2 = fn_dict[block_fn[1]][1](convx1, kernels, bottle_neck)
    convx3 = fn_dict[block_fn[2]][1](convx2, kernels, bottle_neck)
    convx4 = fn_dict[block_fn[3]][1](convx3, kernels, bottle_neck, True)
    Y_out = _post_block(convx4, n_classes)
    model = Model(X_in, Y_out)
    return model

def densenet(in_shape=(224, 224, 3), n_classes=100, opt='adam', kernels=64,  bottle_neck=False, fn_dict=block_fn_dict, block_fn_keys=['db6', 'db12', 'db24', 'db16']):
    return _densenet(in_shape, n_classes, opt, kernels, bottle_neck, block_fn_dict, block_fn_keys)

if __name__ == '__main__':
    model = densenet()
    print(model.summary())