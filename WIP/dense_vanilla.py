"""
    Hand-Crafted-Model
    DenseNet
"""
import tensorflow as tf
from tensorflow.keras import models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import add, Concatenate, Dense, Dropout, Flatten, Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.metrics import Precision, Recall

def _after_conv(in_tensor):
    norm = BatchNormalization()(in_tensor)
    return ReLU()(norm)

def _conv2(in_tensor, k_units, k_size, s=1):
    conv = Conv2D(k_units, kernel_size=k_size, strides=s, padding='same')(in_tensor)
    return _after_conv(conv)

def _block(in_tensor, filters=64, bottleneck=False):
    conv = _conv2(in_tensor, filters, k_size=1, s=1)
    conv = _conv2(conv, filters, k_size=3, s=1)
    if bottleneck:
        return _conv2(conv, filters, k_size=1, s=1)
    return conv

def _transition_block(in_tensor, filters=64, k_size=1, s=1):
    return _conv2(in_tensor, filters, k_size)

def _dense_block4(in_tensor, filters, bottleneck):
    dense1 = _block(in_tensor, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _block(dense, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _block(dense, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3])
    dense4 = _block(dense, filters, bottleneck)
    return dense4

def _dense_block6(in_tensor, filters, bottleneck, pooling=True):
    dense1 = _block(in_tensor, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _block(dense, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _block(dense, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3])
    dense4 = _block(dense, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4])
    dense5 = _block(dense, filters, bottleneck)
    dense = Concatenate()([in_tensor, dense1, dense2, dense3, dense4, dense5])
    dense6 = _block(dense, filters, bottleneck)
    transition = _transition_block(dense6, filters)
    if pooling:
        return AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(transition)
    return transition

def _dense_block12(in_tensor, filters, bottleneck, pooling=True):
    conv = _dense_block4(in_tensor, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    transition = _transition_block(conv, filters)
    if pooling:
        return AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(transition)
    return transition

def _dense_block16(in_tensor, filters, bottleneck, pooling=True):
    conv = _dense_block4(in_tensor, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    transition = _transition_block(conv, filters)
    if pooling:
        return AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(transition)
    return transition

def _dense_block24(in_tensor, filters, bottleneck, pooling=True):
    conv = _dense_block4(in_tensor, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    conv = _dense_block4(conv, filters, bottleneck)
    transition = _transition_block(conv, filters)
    if pooling:
        return AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(transition)
    return transition

def _pre_block(in_tensor, filters, k_size=7, s=2):
    conv = _conv2(in_tensor, filters, k_size, s=s)
    return MaxPooling2D(3, 2, padding='same')(conv)

def _post_block(in_tensor, n_classes):
    pool = GlobalAveragePooling2D()(in_tensor)
    return Dense(n_classes, activation='softmax')(pool)

def _densenet(in_shape, n_classes, opt, kernels, n_layers, useBottleNeck):
    in_layer = Input(in_shape)

    conv1x = _pre_block(in_layer, kernels)

    conv2x = n_layers[0](conv1x, kernels, useBottleNeck)
    conv3x = n_layers[1](conv2x, kernels, useBottleNeck)
    conv4x = n_layers[2](conv3x, kernels, useBottleNeck)
    conv5x = n_layers[3](conv4x, kernels, useBottleNeck, False)
    preds = _post_block(conv5x, n_classes)

    model = Model(in_layer, preds)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])
    return model

def densenet_wo_bottleneck(in_shape=(224, 224, 3), n_classes=100, opt='adam', kernels=64, n_layers=[3, 3, 4, 4]):
    return _densenet(in_shape, n_classes, opt, kernels, [3, 3, 4, 4], False)    

def densenet_w_bottleneck(in_shape=(224, 224, 3), n_classes=100, opt='adam', kernels=64, n_layers=[_dense_block6, _dense_block12, _dense_block24, _dense_block16]):
    return _densenet(in_shape, n_classes, opt, kernels, n_layers, True)  

if __name__ == '__main__':
    model = densenet_wo_bottleneck()
    print(model.summary())