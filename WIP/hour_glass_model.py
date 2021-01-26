"""
    Hour-Glass-Model
"""
import tensorflow as tf
from tensorflow.keras import models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import add, Concatenate, Dense, Dropout, Flatten, Input, Conv2D, Conv2DTranspose, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras.metrics import Precision, Recall

def _after_conv(in_tensor):
    norm = BatchNormalization()(in_tensor)
    return ReLU()(norm)

def _conv2(in_tensor, k_units, k_size, s=1):
    conv = Conv2D(k_units, kernel_size=k_size, strides=s, padding='same')(in_tensor)
    return _after_conv(conv)

def _upsampling2d(in_tensor, k_units):
    return UpSampling2D(size=(2, 2), interpolation='bilinear')(in_tensor)

def _block(in_tensor, filters):
    conv = _conv2(in_tensor, filters, k_size=1, s=1)
    conv = _conv2(conv, filters, k_size=3, s=1)
    return _conv2(conv, filters, k_size=1, s=1)

def _transition_block(in_tensor, filters, k_size=1, s=1):
    return _conv2(in_tensor, filters, k_size)

def _dense_block3(in_tensor, filters, pooling=True):
    dense1 = _block(in_tensor, filters)
    # in_tensor = _transition_block(in_tensor, filters)
    dense = Concatenate()([in_tensor, dense1])
    dense2 = _block(dense, filters)
    dense = Concatenate()([in_tensor, dense1, dense2])
    dense3 = _block(dense, filters)
    transition = _transition_block(dense3, filters)
    if pooling:
        return AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(transition)
    return transition

def _pre_block(in_tensor, filters=128, k_size=1, s=2):
    conv = _conv2(in_tensor, filters, k_size, s=s)
    return MaxPooling2D(3, 2, padding='same')(conv)

def _post_block(in_tensor, n_classes):
    pool = GlobalAveragePooling2D()(in_tensor)
    return Dense(n_classes, activation='softmax')(pool)

def _glassnet(in_shape, n_classes, opt, kernels=128, n_layers=[]):
    in_layer = Input(in_shape)

    conv1x = _pre_block(in_layer, kernels)

    conv2x = _dense_block3(conv1x, kernels//2)
    conv3x = _dense_block3(conv2x, kernels//4)
    conv4x = _dense_block3(conv3x, kernels, False)
    conv4x_up = _upsampling2d(conv4x, kernels)
    conv5x = _dense_block3(conv4x_up, kernels//4, False)
    conv5x_up = _upsampling2d(conv5x, kernels)
    conv6x = _dense_block3(conv5x_up, kernels//2, False)
    conv6x_up = _upsampling2d(conv6x, kernels)
    conv7x = _dense_block3(conv6x_up, kernels, False)

    preds = _post_block(conv7x, n_classes)

    model = Model(in_layer, preds)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])
    return model

def glassnet(in_shape=(224, 224, 3), n_classes=10, opt='adam', kernels=128, n_layers=[]):
    return _glassnet(in_shape, n_classes, opt, kernels, n_layers)

if __name__ == '__main__':
    model = glassnet()
    print(model.summary())