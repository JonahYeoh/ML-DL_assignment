"""
    name: csp_densenet30
    layers: 30 + 5 pooling
    blocks: 4
    distribution: [6, 6, 6, 8]
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class csp_densenet30(tf.keras.Model):
    def __init__(self, k = 32, label_size=10, ratio=0.5, output_activation='softmax'):
        super(csp_densenet30, self).__init__()
        self.split_ratio = int(k * 0.5) # has to be integer
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
        self.block1_tl = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
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
        self.block2_tl = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
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
        self.block3_tl = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
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
        self.block4_conv7 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm7 = BatchNormalization()
        self.block4_conv8 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm8 = BatchNormalization()
        self.classifier = Dense(label_size, activation=output_activation)


    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.input_batchnorm(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
        # block 1
        part_a = x[:, :, :, :self.split_ratio]
        part_b = x[:, :, :, self.split_ratio:]
        block1_layer1 = self.block1_conv1(part_a)
        block1_layer1 = ReLU()(self.block1_batchnorm1(block1_layer1))
        block1_layer1 = self.block1_conv2(block1_layer1)
        block1_layer1 = ReLU()(self.block1_batchnorm2(block1_layer1))

        block1_layer2 = Concatenate()([part_b, block1_layer1])
        block1_layer2 = self.block1_conv3(block1_layer2)
        block1_layer2 = ReLU()(self.block1_batchnorm3(block1_layer2))
        block1_layer2 = self.block1_conv4(block1_layer2)
        block1_layer2 = ReLU()(self.block1_batchnorm4(block1_layer2))
        
        block1_layer3 = Concatenate()([part_b, block1_layer1, block1_layer2])
        block1_layer3 = self.block1_conv5(block1_layer3)
        block1_layer3 = ReLU()(self.block1_batchnorm5(block1_layer3))
        block1_layer3 = self.block1_conv6(block1_layer3)
        block1_layer3 = ReLU()(self.block1_batchnorm6(block1_layer3))

        block1_layer3 = Concatenate()([part_a, block1_layer3])
        block1_layer3 = self.block1_tl(block1_layer3)
        block1 = ReLU()(block1_layer3)
        block1 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block1)
        # block 2
        part_c = block1[:, :, :, :self.split_ratio]
        part_d = block1[:, :, :, self.split_ratio:]
        block2_layer1 = self.block2_conv1(part_d)
        block2_layer1 = ReLU()(self.block2_batchnorm1(block2_layer1))
        block2_layer1 = self.block2_conv2(block2_layer1)
        block2_layer1 = ReLU()(self.block2_batchnorm2(block2_layer1))

        block2_layer2 = Concatenate()([part_d, block2_layer1])
        block2_layer2 = self.block2_conv3(block2_layer2)
        block2_layer2 = ReLU()(self.block2_batchnorm3(block2_layer2))
        block2_layer2 = self.block2_conv4(block2_layer2)
        block2_layer2 = ReLU()(self.block2_batchnorm4(block2_layer2))

        block2_layer3 = Concatenate()([part_d, block2_layer1, block2_layer2])
        block2_layer3 = self.block2_conv5(block2_layer3)
        block2_layer3 = ReLU()(self.block2_batchnorm5(block2_layer3))
        block2_layer3 = self.block2_conv6(block2_layer3)
        block2_layer3 = ReLU()(self.block2_batchnorm6(block2_layer3))

        block2_layer3 = Concatenate()([part_c, block2_layer3])
        block2_layer3 = self.block2_tl(block2_layer3)
        block2 = ReLU()(block2_layer3)
        block2 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block2)
        # block 3
        part_e = block2[:, :, :, :self.split_ratio]
        part_f = block2[:, :, :, self.split_ratio:]
        block3_layer1 = self.block3_conv1(part_f)
        block3_layer1 = ReLU()(self.block3_batchnorm1(block3_layer1))
        block3_layer1 = self.block3_conv2(block3_layer1)
        block3_layer1 = ReLU()(self.block3_batchnorm2(block3_layer1))

        block3_layer2 = Concatenate()([part_f, block3_layer1])
        block3_layer2 = self.block3_conv3(block3_layer2)
        block3_layer2 = ReLU()(self.block3_batchnorm3(block3_layer2))
        block3_layer2 = self.block3_conv4(block3_layer2)
        block3_layer2 = ReLU()(self.block3_batchnorm4(block3_layer2))

        block3_layer3 = Concatenate()([part_f, block3_layer1, block3_layer2])
        block3_layer3 = self.block3_conv5(block3_layer3)
        block3_layer3 = ReLU()(self.block3_batchnorm5(block3_layer3))
        block3_layer3 = self.block3_conv6(block3_layer3)
        block3_layer3 = ReLU()(self.block3_batchnorm6(block3_layer3))

        block3_layer3 = Concatenate()([part_e, block3_layer3])
        block3_layer3 = self.block3_tl(block3_layer3)
        block3 = ReLU()(block3_layer3)
        block3 = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(block3)
        # block 4
        part_g = block3[:, :, :, :self.split_ratio]
        part_h = block3[:, :, :, self.split_ratio:]
        block4_layer1 = self.block4_conv1(part_h)
        block4_layer1 = ReLU()(self.block4_batchnorm1(block4_layer1))
        block4_layer1 = self.block4_conv2(block4_layer1)
        block4_layer1 = ReLU()(self.block4_batchnorm2(block4_layer1))

        block4_layer2 = Concatenate()([part_h, block4_layer1])
        block4_layer2 = self.block4_conv3(block4_layer2)
        block4_layer2 = ReLU()(self.block4_batchnorm3(block4_layer2))
        block4_layer2 = self.block4_conv4(block4_layer2)
        block4_layer2 = ReLU()(self.block4_batchnorm4(block4_layer2))

        block4_layer3 = Concatenate()([part_h, block4_layer1, block4_layer2])
        block4_layer3 = self.block4_conv5(block4_layer3)
        block4_layer3 = ReLU()(self.block4_batchnorm5(block4_layer3))
        block4_layer3 = self.block4_conv6(block4_layer3)
        block4_layer3 = ReLU()(self.block4_batchnorm6(block4_layer3))

        block4_layer4 = Concatenate()([part_h, block4_layer1, block4_layer2, block4_layer3])
        block4_layer4 = self.block4_conv7(block4_layer4)
        block4_layer4 = ReLU()(self.block4_batchnorm7(block4_layer4))
        block4_layer4 = self.block4_conv8(block4_layer4)
        block4_layer4 = ReLU()(self.block4_batchnorm8(block4_layer4))

        block4 = Concatenate()([part_g, block4_layer4])

        x = GlobalAveragePooling2D()(block4)
        x = Flatten()(x)
        return self.classifier(x)
