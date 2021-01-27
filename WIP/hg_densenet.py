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

def _pre_block(in_tensor, filters=128, k_size=2, s=2):
    conv = _conv2(in_tensor, filters, k_size, s=s)
    return MaxPooling2D(2, 2, padding='same')(conv)

def _post_block(in_tensor, n_classes):
    pool = GlobalAveragePooling2D()(in_tensor)
    return Dense(n_classes, activation='softmax')(pool)

def _glassnet_ConvTranspose(in_shape, n_classes, opt, kernels=128, n_layers=[]):
    in_layer = Input(in_shape) # 224 * 224 * 3

    conv1x = _pre_block(in_layer, kernels) # 56 * 56 * 128

    conv2x = _dense_block6(conv1x, kernels//2) # 28 * 28 * 64
    conv3x = _dense_block12(conv2x, kernels//4) # 14 * 14 * 32
    conv4x = _dense_block24(conv3x, kernels//8, False) # 14 * 14 * 16
    conv5x = _dense_block16(conv4x, kernels//4, False) # 14 * 14 * 32
    conv5x_up = Conv2DTranspose(kernels//2, kernel_size=(2,2), strides=2, padding='same')(conv5x) # 28 * 28 * 64
    conv5x_up = Concatenate()([conv2x, conv5x_up]) # 28 * 28 * 128
    conv6x = _dense_block6(conv5x_up, kernels//2, False) # 28 * 28 * 64
    conv6x_up = Conv2DTranspose(kernels//2, kernel_size=(2,2), strides=2, padding='same')(conv6x) # 56 * 56 * 64
    conv6x_up = Concatenate()([conv1x, conv6x_up]) # 56 * 56 * 192
    conv7x = _dense_block6(conv6x_up, kernels, False) # 56 * 56 * 128
    preds = _post_block(conv7x, n_classes) # 1 * 1 * n_classes

    model = Model(in_layer, preds)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])
    return model

def glassnet_ConvTranspose(in_shape=(224, 224, 3), n_classes=10, opt='adam', kernels=128, n_layers=[]):
    return _glassnet_ConvTranspose(in_shape, n_classes, opt, kernels, n_layers)

if __name__ == '__main__':
    model = glassnet_ConvTranspose()
    print(model.summary())