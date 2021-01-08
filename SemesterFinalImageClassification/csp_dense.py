"""

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class csp_densenet_4(tf.keras.Model):
    def __init__(self, k = 32, label_size=10, output_activation='softmax'):
        super(csp_densenet_4, self).__init__()
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
        self.block1_tl = Conv2D(k, kernel_size=(1,1), strides=2, padding='same')
        # block 2
        self.block2_conv1 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm1 = BatchNormalization()
        self.block2_conv2 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm2 = BatchNormalization()
        self.block2_conv3 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm3 = BatchNormalization()
        self.block2_conv4 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm4 = BatchNormalization()
        self.block1_t2 = Conv2D(k, kernel_size=(1,1), strides=2, padding='same')
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
        self.block1_t3 = Conv2D(k, kernel_size=(1,1), strides=2, padding='same')
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
        self.block1_t4 = Conv2D(k, kernel_size=(1,1), strides=2, padding='same')
        self.classifier = Dense(label_size, activation=output_activation)
        self.part1_chnls = k * 0.5
        self.part2_chnls = k * 0.5

    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.input_batchnorm(x)
        x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)
        # block 1
        part_a = x[:, :self.part1_chnls, :, :]
        part_b = x[:, self.part1_chnls:, :, :]
        block1_layer1 = self.block1_conv1(part_a)
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
        block4_layer3 = Concatenate()([block3, block4_layer1, block4_layer2, block4_layer3])
        x = GlobalAveragePooling2D()(block4_layer3)
        x = Flatten()(x)
        return self.classifier(x)