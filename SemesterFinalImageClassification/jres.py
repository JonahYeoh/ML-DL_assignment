"""
"""
"""

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add, Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class j_resnet(tf.keras.Model):
    def __init__(self, k = 64, label_size=10, output_activation='softmax'):
        super(j_resnet, self).__init__()
        self.input_conv = Conv2D(k, kernel_size=(7,7), strides=1, padding='same')
        self.input_batchnorm = BatchNormalization()
        # block 1
        self.block1_conv0 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm_1 = BatchNormalization()
        self.block1_conv1 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm_2 = BatchNormalization()
        self.block1_conv2 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm_3 = BatchNormalization()
        self.block1_conv3 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm_4 = BatchNormalization()
        self.block1_conv4 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm_5 = BatchNormalization()
        self.block1_conv5 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm_6 = BatchNormalization()
        self.transition_1 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        # block 2
        self.block2_conv0 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_1 = BatchNormalization()
        self.block2_conv1 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_2 = BatchNormalization()
        self.block2_conv2 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_3 = BatchNormalization()
        self.block2_conv3 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_4 = BatchNormalization()
        self.block2_conv4 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_5 = BatchNormalization()
        self.block2_conv5 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_6 = BatchNormalization()
        self.transition_2 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        # block 3
        self.block3_conv0 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_1 = BatchNormalization()
        self.block3_conv1 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_2 = BatchNormalization()
        self.block3_conv2 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_3 = BatchNormalization()
        self.block3_conv3 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_4 = BatchNormalization()
        self.block3_conv4 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_5 = BatchNormalization()
        self.block3_conv5 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_6 = BatchNormalization()
        self.classifier = Dense(label_size, activation=output_activation)
        
    def call(self, inputs):
        x = self.input_conv(inputs)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(7,7), strides=2, padding='same')(x) # to test a smaller pool_size
        # block 1
        ## layer 1
        block1_layer1 = self.block1_conv0(x)
        block1_layer1 = self.block1_batchnorm_1(block1_layer1)
        block1_layer1 = ReLU()(block1_layer1)
        block1_layer1 = self.block1_conv1(block1_layer1)
        block1_layer1 = self.block1_batchnorm_2(block1_layer1)
        block1_layer1 = add([x, block1_layer1])
        block1_layer1 = ReLU()(block1_layer1)
        ## layer 2
        block1_layer2 = self.block1_conv2(block1_layer1)
        block1_layer2 = self.block1_batchnorm_3(block1_layer2)
        block1_layer2 = ReLU()(block1_layer2)
        block1_layer2 = self.block1_conv3(block1_layer2)
        block1_layer2 = self.block1_batchnorm_4(block1_layer2)
        block1_layer2 = add([block1_layer1, block1_layer2])
        block1_layer2 = ReLU()(block1_layer2)
        ## layer 3
        block1_layer3 = self.block1_conv4(block1_layer2)
        block1_layer3 = self.block1_batchnorm_5(block1_layer3)
        block1_layer3 = ReLU()(block1_layer3)
        block1_layer3 = self.block1_conv5(block1_layer3)
        block1_layer3 = self.block1_batchnorm_6(block1_layer3)
        block1_layer3 = add([block1_layer2, block1_layer3])
        block1_layer3 = ReLU()(block1_layer3)

        block1_layer6 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(block1_layer3)
        # block 2
        ## layer 1
        block2_layer1 = self.block2_conv0(block1_layer6)
        block2_layer1 = self.block2_batchnorm_1(block2_layer1)
        block2_layer1 = ReLU()(block2_layer1)
        block2_layer1 = self.block2_conv1(block2_layer1)
        block2_layer1 = self.block2_batchnorm_2(block2_layer1)
        x = self.transition_1(block1_layer6)
        block2_layer1 = add([x, block2_layer1])
        block2_layer1 = ReLU()(block2_layer1)
        ## layer 2
        block2_layer2 = self.block2_conv2(block2_layer1)
        block2_layer2 = self.block2_batchnorm_3(block2_layer2)
        block2_layer2 = ReLU()(block2_layer2)
        block2_layer2 = self.block2_conv3(block2_layer2)
        block2_layer2 = self.block2_batchnorm_4(block2_layer2)
        block2_layer2 = add([block2_layer1, block2_layer2])
        block2_layer2 = ReLU()(block2_layer2)
        ## layer 3
        block2_layer3 = self.block2_conv4(block2_layer2)
        block2_layer3 = self.block2_batchnorm_5(block2_layer3)
        block2_layer3 = ReLU()(block2_layer3)
        block2_layer3 = self.block2_conv5(block2_layer3)
        block2_layer3 = self.block2_batchnorm_6(block2_layer3)
        block2_layer3 = add([block2_layer2, block2_layer3])
        block2_layer3 = ReLU()(block2_layer3)
        
        block2_layer6 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(block2_layer3)
        # block 3
        ## layer 1
        block3_layer1 = self.block3_conv0(block2_layer6)
        block3_layer1 = self.block3_batchnorm_1(block3_layer1)
        block3_layer1 = ReLU()(block3_layer1)
        block3_layer1 = self.block3_conv1(block3_layer1)
        block3_layer1 = self.block3_batchnorm_2(block3_layer1)
        x = self.transition_2(block2_layer6)
        block3_layer1 = add([x, block3_layer1])
        block3_layer1 = ReLU()(block3_layer1)
        ## layer 2
        block3_layer2 = self.block3_conv2(block3_layer1)
        block3_layer2 = self.block3_batchnorm_3(block3_layer2)
        block3_layer2 = ReLU()(block3_layer2)
        block3_layer2 = self.block3_conv3(block3_layer2)
        block3_layer2 = self.block3_batchnorm_4(block3_layer2)
        block3_layer2 = add([block3_layer1, block3_layer2])
        block3_layer2 = ReLU()(block3_layer2)
        ## layer 3
        block3_layer3 = self.block3_conv4(block3_layer2)
        block3_layer3 = self.block3_batchnorm_5(block3_layer3)
        block3_layer3 = ReLU()(block3_layer3)
        block3_layer3 = self.block3_conv5(block3_layer3)
        block3_layer3 = self.block3_batchnorm_6(block3_layer3)
        block3_layer3 = add([block3_layer2, block3_layer3])
        block3_layer3 = ReLU()(block3_layer3)
        
        x = GlobalAveragePooling2D()(block3_layer3)
        x = Flatten()(x)
        return self.classifier(x)