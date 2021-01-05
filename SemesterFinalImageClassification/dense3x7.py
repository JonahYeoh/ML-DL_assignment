"""

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

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