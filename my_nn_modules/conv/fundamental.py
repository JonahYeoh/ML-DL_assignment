import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D

def _after_conv(in_tensor):
    bn = BatchNormalization()(in_tensor)
    return ReLU()(bn)

def _conv2d(in_tensor, filters, k_size=3, strides=1):
    conv = Conv2D(filters, kernel_size=k_size, strides=strides, padding='same')(in_tensor)
    return _after_conv(conv)

def _transition_layer2d(in_tensor, filters, pooling=True, pooling_method='avg'):
    conv = _conv2d(in_tensor, filters, 1, 1)
    if pooling:
        if pooling_method == 'avg':
            return AveragePooling2D(pool_size=2,strides=2, padding='same')(conv)
        return MaxPooling2D(pool_size=2, strides=2, padding='same')(conv)
    return conv
