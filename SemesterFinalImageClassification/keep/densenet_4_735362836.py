"""

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class densenet3x7(tf.keras.Model):
    def __init__(self, k = 32, blocks=3, per_block=[7, 7, 7], label_size=10, output_activation='softmax'):
        super(densenet3x7, self).__init__()
        self.input_conv = Conv2D(k, kernel_size=(7,7), strides=1, padding='same')
        self.input_batchnorm = BatchNormalization()
        # block 1
        self.block1_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm1 = BatchNormalization()
        self.block1_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm2 = BatchNormalization()
        self.block1_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm3 = BatchNormalization()
        self.block1_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm4 = BatchNormalization()
        self.block1_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm5 = BatchNormalization()
        self.block1_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm6 = BatchNormalization()
        self.block1_conv7 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm7 = BatchNormalization()
        self.block1_conv8 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm8 = BatchNormalization()
        self.block1_conv9 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm9 = BatchNormalization()
        self.block1_conv10 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm10 = BatchNormalization()
        self.block1_conv11 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm11 = BatchNormalization()
        self.block1_conv12 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm12 = BatchNormalization()
        self.block1_conv13 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm13 = BatchNormalization()
        self.block1_conv14 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm14 = BatchNormalization()
        self.tl_1 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 2
        self.block2_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm1 = BatchNormalization()
        self.block2_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm2 = BatchNormalization()
        self.block2_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm3 = BatchNormalization()
        self.block2_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm4 = BatchNormalization()
        self.block2_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm5 = BatchNormalization()
        self.block2_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm6 = BatchNormalization()
        self.block2_conv7 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm7 = BatchNormalization()
        self.block2_conv8 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm8 = BatchNormalization()
        self.block2_conv9 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm9 = BatchNormalization()
        self.block2_conv10 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm10 = BatchNormalization()
        self.block2_conv11 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm11 = BatchNormalization()
        self.block2_conv12 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm12 = BatchNormalization()
        self.block2_conv13 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm13 = BatchNormalization()
        self.block2_conv14 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm14 = BatchNormalization()
        self.tl_2 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 3
        self.block3_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm1 = BatchNormalization()
        self.block3_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm2 = BatchNormalization()
        self.block3_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm3 = BatchNormalization()
        self.block3_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm4 = BatchNormalization()
        self.block3_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm5 = BatchNormalization()
        self.block3_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm6 = BatchNormalization()
        self.block3_conv7 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm7 = BatchNormalization()
        self.block3_conv8 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm8 = BatchNormalization()
        self.block3_conv9 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm9 = BatchNormalization()
        self.block3_conv10 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm10 = BatchNormalization()
        self.block3_conv11 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm11 = BatchNormalization()
        self.block3_conv12 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm12 = BatchNormalization()
        self.block3_conv13 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm13 = BatchNormalization()
        self.block3_conv14 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm14 = BatchNormalization()
        self.tl_1 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        self.classifier = Dense(label_size, activation=output_activation)
    
    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.input_batchnorm(x)
        x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)
        # block 1
        layer_1 = self.block1_conv1(x)
        layer_1 = ReLU()(self.block1_batchnorm1(layer_1))
        layer_1 = self.block1_conv2(layer_1)
        layer_1 = ReLU()(self.block1_batchnorm2(layer_1))

        layer_2 = Concatenate()([x, layer_1])
        layer_2 = self.block1_conv3(layer_2)
        layer_2 = ReLU()(self.block1_batchnorm3(layer_2))
        layer_2 = self.block1_conv4(layer_2)
        layer_2 = ReLU()(self.block1_batchnorm4(layer_2))

        layer_3 = Concatenate()([x, layer_1, layer_2])
        layer_3 = self.block1_conv5(layer_3)
        layer_3 = ReLU()(self.block1_batchnorm5(layer_3))
        layer_3 = self.block1_conv6(layer_3)
        layer_3 = ReLU()(self.block1_batchnorm6(layer_3))

        layer_4 = Concatenate()([x, layer_1, layer_2, layer_3])
        layer_4 = self.block1_conv7(layer_4)
        layer_4 = ReLU()(self.block1_batchnorm7(layer_4))
        layer_4 = self.block1_conv8(layer_4)
        layer_4 = ReLU()(self.block1_batchnorm8(layer_4))

        layer_5 = Concatenate()([x, layer_1, layer_2, layer_3, layer_4])
        layer_5 = self.block1_conv9(layer_5)
        layer_5 = ReLU()(self.block1_batchnorm9(layer_5))
        layer_5 = self.block1_conv10(layer_5)
        layer_5 = ReLU()(self.block1_batchnorm10(layer_5))

        layer_6 = Concatenate()([x, layer_1, layer_2, layer_3, layer_4, layer_5])
        layer_6 = self.block1_conv11(layer_6)
        layer_6 = ReLU()(self.block1_batchnorm11(layer_6))
        layer_6 = self.block1_conv12(layer_6)
        layer_6 = ReLU()(self.block1_batchnorm12(layer_6))

        layer_7 = Concatenate()([x, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6])
        layer_7 = self.block1_conv13(layer_7)
        layer_7 = ReLU()(self.block1_batchnorm13(layer_7))
        layer_7 = self.block1_conv14(layer_7)
        layer_7 = ReLU()(self.block1_batchnorm14(layer_7))
        
        layer_7 = Concatenate()([x, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7])
        block1 = self.tl_1(layer_7)
        block1 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block1)
        # block 2
        layer_8 = self.block2_conv1(block1)
        layer_8 = ReLU()(self.block2_batchnorm1(layer_8))
        layer_8 = self.block2_conv2(layer_8)
        layer_8 = ReLU()(self.block2_batchnorm2(layer_8))

        layer_9 = Concatenate()([block1, layer_8])
        layer_9 = self.block2_conv3(layer_9)
        layer_9 = ReLU()(self.block2_batchnorm3(layer_9))
        layer_9 = self.block2_conv4(layer_9)
        layer_9 = ReLU()(self.block2_batchnorm4(layer_9))

        layer_10 = Concatenate()([block1, layer_8, layer_9])
        layer_10 = self.block2_conv5(layer_10)
        layer_10 = ReLU()(self.block2_batchnorm5(layer_10))
        layer_10 = self.block2_conv6(layer_10)
        layer_10 = ReLU()(self.block2_batchnorm6(layer_10))

        layer_11 = Concatenate()([block1, layer_8, layer_9, layer_10])
        layer_11 = self.block2_conv7(layer_11)
        layer_11 = ReLU()(self.block2_batchnorm7(layer_11))
        layer_11 = self.block2_conv8(layer_11)
        layer_11 = ReLU()(self.block2_batchnorm8(layer_11))

        layer_12 = Concatenate()([block1, layer_8, layer_9, layer_10, layer_11])
        layer_12 = self.block2_conv9(layer_12)
        layer_12 = ReLU()(self.block2_batchnorm9(layer_12))
        layer_12 = self.block2_conv10(layer_12)
        layer_12 = ReLU()(self.block2_batchnorm10(layer_12))

        layer_13 = Concatenate()([block1, layer_8, layer_9, layer_10, layer_11, layer_12])
        layer_13 = self.block2_conv11(layer_13)
        layer_13 = ReLU()(self.block2_batchnorm11(layer_13))
        layer_13 = self.block2_conv12(layer_13)
        layer_13 = ReLU()(self.block2_batchnorm12(layer_13))

        layer_14 = Concatenate()([block1, layer_8, layer_9, layer_10, layer_11, layer_12, layer_13])
        layer_14 = self.block2_conv13(layer_14)
        layer_14 = ReLU()(self.block2_batchnorm13(layer_14))
        layer_14 = self.block2_conv14(layer_14)
        layer_14 = ReLU()(self.block2_batchnorm14(layer_14))
        
        layer_14 = Concatenate()([block1, layer_8, layer_9, layer_10, layer_11, layer_12, layer_13, layer_14])
        block2 = self.tl_2(layer_14)
        block2 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block2)
        # block 3
        layer_15 = self.block3_conv1(block2)
        layer_15 = ReLU()(self.block3_batchnorm1(layer_15))
        layer_15 = self.block3_conv2(layer_15)
        layer_15 = ReLU()(self.block3_batchnorm2(layer_15))

        layer_16 = Concatenate()([block2, layer_15])
        layer_16 = self.block3_conv3(layer_16)
        layer_16 = ReLU()(self.block3_batchnorm3(layer_16))
        layer_16 = self.block3_conv4(layer_16)
        layer_16 = ReLU()(self.block3_batchnorm4(layer_16))

        layer_17 = Concatenate()([block2, layer_15, layer_16])
        layer_17 = self.block3_conv5(layer_17)
        layer_17 = ReLU()(self.block3_batchnorm5(layer_17))
        layer_17 = self.block3_conv6(layer_17)
        layer_17 = ReLU()(self.block3_batchnorm6(layer_17))

        layer_18 = Concatenate()([block2, layer_15, layer_16, layer_17])
        layer_18 = self.block3_conv7(layer_18)
        layer_18 = ReLU()(self.block3_batchnorm7(layer_18))
        layer_18 = self.block3_conv8(layer_18)
        layer_18 = ReLU()(self.block3_batchnorm8(layer_18))

        layer_19 = Concatenate()([block2, layer_15, layer_16, layer_17, layer_18])
        layer_19 = self.block3_conv9(layer_19)
        layer_19 = ReLU()(self.block3_batchnorm9(layer_19))
        layer_19 = self.block3_conv10(layer_19)
        layer_19 = ReLU()(self.block3_batchnorm10(layer_19))

        layer_20 = Concatenate()([block2, layer_15, layer_16, layer_17, layer_18, layer_19])
        layer_20 = self.block3_conv11(layer_20)
        layer_20 = ReLU()(self.block3_batchnorm11(layer_20))
        layer_20 = self.block3_conv12(layer_20)
        layer_20 = ReLU()(self.block3_batchnorm12(layer_20))

        layer_21 = Concatenate()([block2, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20])
        layer_21 = self.block3_conv13(layer_21)
        layer_21 = ReLU()(self.block3_batchnorm13(layer_21))
        layer_21 = self.block3_conv14(layer_21)
        layer_21 = ReLU()(self.block3_batchnorm14(layer_21))
        
        layer_21 = Concatenate()([block2, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21])

        x = GlobalAveragePooling2D()(layer_21)
        x = Flatten()(x)
        return self.classifier(x)


class densenet3x3(tf.keras.Model):
    def __init__(self, k = 32, label_size=10, output_activation='softmax'):
        super(densenet3x3, self).__init__()
        self.input_conv = Conv2D(k, kernel_size=(7,7), strides=1, padding='same')
        self.input_batchnorm = BatchNormalization()
        # block 1
        self.block1_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm1 = BatchNormalization()
        self.block1_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm2 = BatchNormalization()
        self.block1_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm3 = BatchNormalization()
        self.block1_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm4 = BatchNormalization()
        self.block1_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm5 = BatchNormalization()
        self.block1_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm6 = BatchNormalization()
        self.tl_1 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 2
        self.block2_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm1 = BatchNormalization()
        self.block2_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm2 = BatchNormalization()
        self.block2_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm3 = BatchNormalization()
        self.block2_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm4 = BatchNormalization()
        self.block2_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm5 = BatchNormalization()
        self.block2_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm6 = BatchNormalization()
        self.tl_2 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 3
        self.block3_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm1 = BatchNormalization()
        self.block3_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm2 = BatchNormalization()
        self.block3_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm3 = BatchNormalization()
        self.block3_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm4 = BatchNormalization()
        self.block3_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm5 = BatchNormalization()
        self.block3_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm6 = BatchNormalization()
        self.classifier = Dense(label_size, activation=output_activation)
    
    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.input_batchnorm(x)
        x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)
        # block 1
        block1_layer1 = self.block1_conv1(x)
        block1_layer1 = ReLU()(self.block1_batchnorm1(block1_layer1))
        block1_layer1 = self.block1_conv2(block1_layer1)
        block1_layer1 = ReLU()(self.block1_batchnorm2(block1_layer1))

        block1_layer2 = Concatenate()([x, block1_layer1])
        block1_layer2 = self.block1_conv3(block1_layer2)
        block1_layer2 = ReLU()(self.block1_batchnorm3(block1_layer2))
        block1_layer2 = self.block1_conv4(block1_layer2)
        block1_layer2 = ReLU()(self.block1_batchnorm4(block1_layer2))

        block1_layer3 = Concatenate()([x, block1_layer1, block1_layer2])
        block1_layer3 = self.block1_conv5(block1_layer3)
        block1_layer3 = ReLU()(self.block1_batchnorm5(block1_layer3))
        block1_layer3 = self.block1_conv6(block1_layer3)
        block1_layer3 = ReLU()(self.block1_batchnorm6(block1_layer3))

        block1_layer3 = Concatenate()([x, block1_layer1, block1_layer2, block1_layer3])
        block1_layer3 = self.tl_1(block1_layer3)
        block1 = ReLU()(block1_layer3)
        block1 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block1)
        # block 2
        block2_layer1 = self.block2_conv1(block1)
        block2_layer1 = ReLU()(self.block2_batchnorm1(block2_layer1))
        block2_layer1 = self.block2_conv2(block2_layer1)
        block2_layer1 = ReLU()(self.block2_batchnorm2(block2_layer1))

        block2_layer2 = Concatenate()([block1, block2_layer1])
        block2_layer2 = self.block2_conv3(block2_layer2)
        block2_layer2 = ReLU()(self.block2_batchnorm3(block2_layer2))
        block2_layer2 = self.block2_conv4(block2_layer2)
        block2_layer2 = ReLU()(self.block2_batchnorm4(block2_layer2))

        block2_layer3 = Concatenate()([block1, block2_layer1, block2_layer2])
        block2_layer3 = self.block2_conv5(block2_layer3)
        block2_layer3 = ReLU()(self.block2_batchnorm5(block2_layer3))
        block2_layer3 = self.block2_conv6(block2_layer3)
        block2_layer3 = ReLU()(self.block2_batchnorm6(block2_layer3))

        block2_layer3 = Concatenate()([block1, block2_layer1, block2_layer2, block2_layer3])
        block2_layer3 = self.tl_2(block2_layer3)
        block2 = ReLU()(block2_layer3)
        block2 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block2)
        # block 3
        block3_layer1 = self.block3_conv1(block2)
        block3_layer1 = ReLU()(self.block3_batchnorm1(block3_layer1))
        block3_layer1 = self.block3_conv2(block3_layer1)
        block3_layer1 = ReLU()(self.block3_batchnorm2(block3_layer1))

        block3_layer2 = Concatenate()([block2, block3_layer1])
        block3_layer2 = self.block3_conv3(block3_layer2)
        block3_layer2 = ReLU()(self.block3_batchnorm3(block3_layer2))
        block3_layer2 = self.block3_conv4(block3_layer2)
        block3_layer2 = ReLU()(self.block3_batchnorm4(block3_layer2))

        block3_layer3 = Concatenate()([block2, block3_layer1, block3_layer2])
        block3_layer3 = self.block3_conv5(block3_layer3)
        block3_layer3 = ReLU()(self.block3_batchnorm5(block3_layer3))
        block3_layer3 = self.block3_conv6(block3_layer3)
        block3_layer3 = ReLU()(self.block3_batchnorm6(block3_layer3))

        x = GlobalAveragePooling2D()(block3_layer3)
        x = Flatten()(x)
        return self.classifier(x)


class densenet_4(tf.keras.Model):
    def __init__(self, k = 32, label_size=10, output_activation='softmax'):
        super(densenet_4, self).__init__()
        self.input_conv = Conv2D(k, kernel_size=(7,7), strides=1, padding='same')
        self.input_batchnorm = BatchNormalization()
        # block 1
        self.block1_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm1 = BatchNormalization()
        self.block1_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm2 = BatchNormalization()
        self.block1_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm3 = BatchNormalization()
        self.block1_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm4 = BatchNormalization()
        self.tl_1 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 2
        self.block2_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm1 = BatchNormalization()
        self.block2_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm2 = BatchNormalization()
        self.block2_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm3 = BatchNormalization()
        self.block2_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm4 = BatchNormalization()
        self.tl_2 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 3
        self.block3_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm1 = BatchNormalization()
        self.block3_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm2 = BatchNormalization()
        self.block3_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm3 = BatchNormalization()
        self.block3_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm4 = BatchNormalization()
        self.block3_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm5 = BatchNormalization()
        self.block3_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm6 = BatchNormalization()
        self.tl_3 = Conv2D(k, kernel_size=(1,1), strides=(1,1), padding='same')
        # block 4
        self.block4_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm1 = BatchNormalization()
        self.block4_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm2 = BatchNormalization()
        self.block4_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm3 = BatchNormalization()
        self.block4_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm4 = BatchNormalization()
        self.block4_conv5 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm5 = BatchNormalization()
        self.block4_conv6 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm6 = BatchNormalization()
        self.classifier = Dense(label_size, activation=output_activation)
    
    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.input_batchnorm(x)
        x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)
        # block 1
        block1_layer1 = self.block1_conv1(x)
        block1_layer1 = ReLU()(self.block1_batchnorm1(block1_layer1))
        block1_layer1 = self.block1_conv2(block1_layer1)
        block1_layer1 = ReLU()(self.block1_batchnorm2(block1_layer1))

        block1_layer2 = Concatenate()([x, block1_layer1])
        block1_layer2 = self.block1_conv3(block1_layer2)
        block1_layer2 = ReLU()(self.block1_batchnorm3(block1_layer2))
        block1_layer2 = self.block1_conv4(block1_layer2)
        block1_layer2 = ReLU()(self.block1_batchnorm4(block1_layer2))

        block1_layer3 = Concatenate()([x, block1_layer1, block1_layer2])
        block1_layer3 = self.tl_1(block1_layer3)
        block1 = ReLU()(block1_layer3)
        block1 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block1)
        # block 2
        block2_layer1 = self.block2_conv1(block1)
        block2_layer1 = ReLU()(self.block2_batchnorm1(block2_layer1))
        block2_layer1 = self.block2_conv2(block2_layer1)
        block2_layer1 = ReLU()(self.block2_batchnorm2(block2_layer1))

        block2_layer2 = Concatenate()([block1, block2_layer1])
        block2_layer2 = self.block2_conv3(block2_layer2)
        block2_layer2 = ReLU()(self.block2_batchnorm3(block2_layer2))
        block2_layer2 = self.block2_conv4(block2_layer2)
        block2_layer2 = ReLU()(self.block2_batchnorm4(block2_layer2))

        block2_layer3 = Concatenate()([block1, block2_layer1, block2_layer2])
        block2_layer3 = self.tl_2(block2_layer3)
        block2 = ReLU()(block2_layer3)
        block2 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block2)
        # block 3
        block3_layer1 = self.block3_conv1(block2)
        block3_layer1 = ReLU()(self.block3_batchnorm1(block3_layer1))
        block3_layer1 = self.block3_conv2(block3_layer1)
        block3_layer1 = ReLU()(self.block3_batchnorm2(block3_layer1))

        block3_layer2 = Concatenate()([block2, block3_layer1])
        block3_layer2 = self.block3_conv3(block3_layer2)
        block3_layer2 = ReLU()(self.block3_batchnorm3(block3_layer2))
        block3_layer2 = self.block3_conv4(block3_layer2)
        block3_layer2 = ReLU()(self.block3_batchnorm4(block3_layer2))

        block3_layer3 = Concatenate()([block2, block3_layer1, block3_layer2])
        block3_layer3 = self.block3_conv5(block3_layer3)
        block3_layer3 = ReLU()(self.block3_batchnorm5(block3_layer3))
        block3_layer3 = self.block3_conv6(block3_layer3)
        block3_layer3 = ReLU()(self.block3_batchnorm6(block3_layer3))
        block3_layer3 = Concatenate()([block2, block3_layer1, block3_layer2, block3_layer3])
        block3_layer3 = self.tl_3(block3_layer3)
        block3 = ReLU()(block3_layer3)
        block3 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block3)
        # block 4
        block4_layer1 = self.block4_conv1(block3)
        block4_layer1 = ReLU()(self.block4_batchnorm1(block4_layer1))
        block4_layer1 = self.block4_conv2(block4_layer1)
        block4_layer1 = ReLU()(self.block4_batchnorm2(block4_layer1))

        block4_layer2 = Concatenate()([block3, block4_layer1])
        block4_layer2 = self.block4_conv3(block4_layer2)
        block4_layer2 = ReLU()(self.block4_batchnorm3(block4_layer2))
        block4_layer2 = self.block4_conv4(block4_layer2)
        block4_layer2 = ReLU()(self.block4_batchnorm4(block4_layer2))

        block4_layer3 = Concatenate()([block3, block4_layer1, block4_layer2])
        block4_layer3 = self.block4_conv5(block4_layer3)
        block4_layer3 = ReLU()(self.block4_batchnorm5(block4_layer3))
        block4_layer3 = self.block4_conv6(block4_layer3)
        block4_layer3 = ReLU()(self.block4_batchnorm6(block4_layer3))
        #
        x = Concatenate()([block3, block4_layer1, block4_layer2, block4_layer3])
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        return self.classifier(x)



"""
name = densenet_4
epoch = 100
patience = epoch//4
monitor = val_loss
optimizer = adam
output_activation = sigmoid
========================================================================
k=32	loss	accuracy	precision	recall	f1-score
0	0.1427	0.94975	0.9785	0.50361	0.664974104
1	0.15301	0.9465	0.9864	0.46804	0.634848679
2	0.17211	0.94869	0.98211	0.48076	0.645524488
3	0.17528	0.94744	0.99003	0.44093	0.610127366
4	0.16294	0.94934	0.98652	0.4852	0.650476319
mean	0.161208	0.948344	0.984712	0.475708	0.641190191
min	0.1427	0.9465	0.9785	0.44093	0.610127366
max	0.17528	0.94975	0.99003	0.50361	0.664974104
stdev	0.013505812	0.001351492	0.004465386	0.023251075	0.020468309
batch 50					
					
k=48	loss	accuracy	precision	recall	f1-score
0	0.16523	0.94419	0.98121	0.59345	0.739587053
1					#DIV/0!
2					#DIV/0!
3					#DIV/0!
4					#DIV/0!
mean	0.033046	0.188838	0.196242	0.11869	#DIV/0!
min	0.16523	0.94419	0.98121	0.59345	#DIV/0!
max	0.16523	0.94419	0.98121	0.59345	#DIV/0!
stdev	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!
batch 50					

""""
