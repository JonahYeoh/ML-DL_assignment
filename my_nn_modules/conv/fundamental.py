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

def _channel_attention(in_tensor, reduction=8):
    w0 = Dense(in_tensor.shape[-1]//reduction, activation='relu')
    w1 = Dense(in_tensor.shape[-1], activation='relu')
    p_avg = GlobalAveragePooling2D()(in_tensor)
    p_max = GlobalMaxPool2D()(in_tensor)
    dense_avg = w1(w0(p_avg))
    dense_max = w1(w0(p_max))
    summation = Add()([dense_avg, dense_max])
    return Activation('sigmoid')(summation)

def _spatial_attention(in_tensor, kernels):
    p_avg = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='same')(in_tensor)
    p_max = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(in_tensor)
    stack = Concatenate()([p_avg, p_max])
    conv = Conv2D(1, kernel_size=(7,7), strides=(1,1), padding='same')(stack)
    return Activation('sigmoid')(conv)

def _spatial_attention_module(in_tensor, kernels):
    f1 = Multiply()([_channel_attention(in_tensor), in_tensor])
    f2 = Multiply()([_spatial_attention(f1), f1])
    return Add()([in_tensor, f2])
