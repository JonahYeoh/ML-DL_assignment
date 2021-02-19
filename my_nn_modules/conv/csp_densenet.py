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

def _csp_dense_block6(in_tensor, filters, split_ratio=0.5, bottle_neck=True, last_block=False):
    split_index = int(in_tensor.shape[3] * split_ratio)
    part_a = in_tensor[:, :, :, :split_index]
    part_b = in_tensor[:, :, :, split_index:]
    dense1 = _dense_block(part_a, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    last = Concatenate()([part_b, dense6])
    return _dense_block_output(last, filters, 'avg', False, last_block) # arg 4 hardcode to False to activate transition layer

def _csp_dense_block12(in_tensor, filters, split_ratio=0.5, bottle_neck=True, last_block=False):
    split_index = int(in_tensor.shape[3] * split_ratio)
    part_a = in_tensor[:, :, :, :split_index]
    part_b = in_tensor[:, :, :, split_index:]
    dense1 = _dense_block(part_a, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6])
    dense7 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7])
    dense8 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8])
    dense9 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9])
    dense10 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10])
    dense11 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11])
    dense12 = _dense_block(dense, filters, bottle_neck)
    last = Concatenate()([part_b, dense12])
    return _dense_block_output(last, filters, 'avg', False, last_block) # arg 4 hardcode to False to activate transition layer

def _csp_dense_block16(in_tensor, filters, split_ratio=0.5, bottle_neck=True, last_block=False):
    split_index = int(in_tensor.shape[3] * split_ratio)
    part_a = in_tensor[:, :, :, :split_index]
    part_b = in_tensor[:, :, :, split_index:]
    dense1 = _dense_block(part_a, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6])
    dense7 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7])
    dense8 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8])
    dense9 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9])
    dense10 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10])
    dense11 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11])
    dense12 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12])
    dense13 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13])
    dense14 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14])
    dense15 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15])
    dense16 = _dense_block(dense, filters, bottle_neck)
    last = Concatenate()([part_b, dense16])
    return _dense_block_output(last, filters, 'avg', False, last_block) # arg 4 hardcode to False to activate transition layer


def _csp_dense_block24(in_tensor, filters, split_ratio=0.5, bottle_neck=True, last_block=False):
    split_index = int(in_tensor.shape[3] * split_ratio)
    part_a = in_tensor[:, :, :, :split_index]
    part_b = in_tensor[:, :, :, split_index:]
    dense1 = _dense_block(part_a, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1])
    dense2 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2])
    dense3 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3])
    dense4 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4])
    dense5 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5])
    dense6 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6])
    dense7 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7])
    dense8 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8])
    dense9 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9])
    dense10 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10])
    dense11 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11])
    dense12 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12])
    dense13 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13])
    dense14 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14])
    dense15 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15])
    dense16 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16])
    dense17 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17])
    dense18 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18])
    dense19 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19])
    dense20 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20])
    dense21 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20, dense21])
    dense22 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20, dense21, dense22])
    dense23 = _dense_block(dense, filters, bottle_neck)
    dense = Concatenate()([part_a, dense1, dense2, dense3, dense4, dense5, dense6, dense7, dense8, dense9, dense10, dense11, dense12, dense13, dense14, dense15, dense16, dense17, dense18, dense19, dense20, dense21, dense22, dense23])
    dense24 = _dense_block(dense, filters, bottle_neck)    
    last = Concatenate()([part_b, dense24])
    return _dense_block_output(last, filters, 'avg', False, last_block) # arg 4 hardcode to False to activate transition layer

block_fn_dict = { 'csp_db6': ['csp dense block 6', _csp_dense_block6], 'csp_db12': ['csp dense block 12', _csp_dense_block12], 'csp_db16': ['csp dense block 16', _csp_dense_block16], 'csp_db24': ['csp dense block 24', _csp_dense_block24] }

def get_dict():
    return block_fn_dict

def _csp_densenet(in_shape=(224, 224, 3), n_classes=100, opt='adam', kernels=64, bottle_neck=False, fn_dict=block_fn_dict, block_fn=['csp_db6', 'csp_db12', 'csp_db24', 'csp_db16'], split_ratio=0.5):
    X_in = Input(in_shape)
    X = _pre_block(X_in)
    convx1 = fn_dict[block_fn[0]][1](X, kernels, split_ratio, bottle_neck)
    convx2 = fn_dict[block_fn[1]][1](convx1, kernels, split_ratio, bottle_neck)
    convx3 = fn_dict[block_fn[2]][1](convx2, kernels, split_ratio, bottle_neck)
    convx4 = fn_dict[block_fn[3]][1](convx3, kernels, split_ratio, bottle_neck, True)
    Y_out = _post_block(convx4, n_classes)
    model = Model(X_in, Y_out)
    return model

def csp_densenet(in_shape=(224, 224, 3), n_classes=100, opt='adam', kernels=64, bottle_neck=False, fn_dict=block_fn_dict, block_fn_keys=['csp_db6', 'csp_db12', 'csp_db24', 'csp_db16'], split_ratio=0.5):
    return _csp_densenet(in_shape, n_classes, opt, kernels, bottle_neck, block_fn_dict, block_fn_keys, split_ratio)

if __name__ == '__main__':
    model = csp_densenet()
    print(model.summary())