"""
    name: se_resnet29
    layers: 29 + 2 pooling
    blocks: 4
    distribution: [6, 8, 8, 6]
    r: 8
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add, multiply, Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class se_resnet29(tf.keras.Model):
    def __init__(self, k = 64, r=8, label_size=10, output_activation='softmax'):
        super(se_resnet29, self).__init__()
        d = k // r
        self.input_conv = Conv2D(k, kernel_size=(7,7), strides=1, padding='same')
        self.input_batchnorm = BatchNormalization()
        # block 1
        self.block1_conv0 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm_1 = BatchNormalization()
        self.block1_conv1 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm_2 = BatchNormalization()
        self.block1_dense1 = Dense(d)
        self.block1_dense2 = Dense(d*r)
        self.block1_conv2 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm_3 = BatchNormalization()
        self.block1_conv3 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm_4 = BatchNormalization()
        self.block1_dense3 = Dense(d)
        self.block1_dense4 = Dense(d*r)
        self.block1_conv4 = Conv2D(k, kernel_size=(1,1), strides=1, padding='same')
        self.block1_batchnorm_5 = BatchNormalization()
        self.block1_conv5 = Conv2D(k, kernel_size=(3,3), strides=1, padding='same')
        self.block1_batchnorm_6 = BatchNormalization()
        self.block1_dense5 = Dense(d)
        self.block1_dense6 = Dense(d*r)
        self.transition_1 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        # block 2
        self.block2_conv0 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_1 = BatchNormalization()
        self.block2_conv1 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_2 = BatchNormalization()
        self.block2_dense1 = Dense(d*2)
        self.block2_dense2 = Dense(d*2*r)
        self.block2_conv2 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_3 = BatchNormalization()
        self.block2_conv3 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_4 = BatchNormalization()
        self.block2_dense3 = Dense(d*2)
        self.block2_dense4 = Dense(d*2*r)
        self.block2_conv4 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_5 = BatchNormalization()
        self.block2_conv5 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_6 = BatchNormalization()
        self.block2_dense5 = Dense(d*2)
        self.block2_dense6 = Dense(d*2*r)
        self.block2_conv6 = Conv2D(2*k, kernel_size=(1,1), strides=1, padding='same')
        self.block2_batchnorm_7 = BatchNormalization()
        self.block2_conv7 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.block2_batchnorm_8 = BatchNormalization()
        self.block2_dense7 = Dense(d*2)
        self.block2_dense8 = Dense(d*2*r)
        self.transition_2 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        # block 3
        self.block3_conv0 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_1 = BatchNormalization()
        self.block3_conv1 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_2 = BatchNormalization()
        self.block3_dense1 = Dense(d*4)
        self.block3_dense2 = Dense(d*4*r)
        self.block3_conv2 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_3 = BatchNormalization()
        self.block3_conv3 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_4 = BatchNormalization()
        self.block3_dense3 = Dense(d*4)
        self.block3_dense4 = Dense(d*4*r)
        self.block3_conv4 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_5 = BatchNormalization()
        self.block3_conv5 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_6 = BatchNormalization()
        self.block3_dense5 = Dense(d*4)
        self.block3_dense6 = Dense(d*4*r)
        self.block3_conv6 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block3_batchnorm_7 = BatchNormalization()
        self.block3_conv7 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block3_batchnorm_8 = BatchNormalization()
        self.block3_dense7 = Dense(d*4)
        self.block3_dense8 = Dense(d*4*r)
        self.transition_3 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        # block 4
        self.block4_conv0 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm_1 = BatchNormalization()
        self.block4_conv1 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm_2 = BatchNormalization()
        self.block4_dense1 = Dense(d*4)
        self.block4_dense2 = Dense(d*4*r)
        self.block4_conv2 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm_3 = BatchNormalization()
        self.block4_conv3 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm_4 = BatchNormalization()
        self.block4_dense3 = Dense(d*4)
        self.block4_dense4 = Dense(d*4*r)
        self.block4_conv4 = Conv2D(4*k, kernel_size=(1,1), strides=1, padding='same')
        self.block4_batchnorm_5 = BatchNormalization()
        self.block4_conv5 = Conv2D(4*k, kernel_size=(3,3), strides=1, padding='same')
        self.block4_batchnorm_6 = BatchNormalization()
        self.block4_dense5 = Dense(d*4)
        self.block4_dense6 = Dense(d*4*r)
        
        self.classifier = Dense(label_size, activation=output_activation)
        
    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.input_batchnorm(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x) # to test a smaller pool_size
        # block 1
        ## layer 1
        block1_layer1 = self.block1_conv0(x)
        block1_layer1 = self.block1_batchnorm_1(block1_layer1)
        block1_layer1 = ReLU()(block1_layer1)
        block1_layer1 = self.block1_conv1(block1_layer1)
        block1_layer1 = self.block1_batchnorm_2(block1_layer1)
        sq_1 = GlobalAveragePooling2D()(block1_layer1)
        sq_1 = self.block1_dense1(sq_1)
        sq_1 = ReLU()(sq_1)
        sq_1 = self.block1_dense2(sq_1)
        sq_1 = tf.keras.activations.sigmoid(sq_1)
        exci_1 = multiply([block1_layer1, sq_1])
        block1_layer1 = add([x, exci_1])
        block1_layer1 = ReLU()(block1_layer1)
        ## layer 2
        block1_layer2 = self.block1_conv2(block1_layer1)
        block1_layer2 = self.block1_batchnorm_3(block1_layer2)
        block1_layer2 = ReLU()(block1_layer2)
        block1_layer2 = self.block1_conv3(block1_layer2)
        block1_layer2 = self.block1_batchnorm_4(block1_layer2)
        sq_2 = GlobalAveragePooling2D()(block1_layer2)
        sq_2 = self.block1_dense3(sq_2)
        sq_2 = ReLU()(sq_2)
        sq_2 = self.block1_dense4(sq_2)
        sq_2 = tf.keras.activations.sigmoid(sq_2)
        exci_2 = multiply([block1_layer2, sq_2])
        block1_layer2 = add([block1_layer1, exci_2])
        block1_layer2 = ReLU()(block1_layer2)
        ## layer 3
        block1_layer3 = self.block1_conv4(block1_layer2)
        block1_layer3 = self.block1_batchnorm_5(block1_layer3)
        block1_layer3 = ReLU()(block1_layer3)
        block1_layer3 = self.block1_conv5(block1_layer3)
        block1_layer3 = self.block1_batchnorm_6(block1_layer3)
        sq_3 = GlobalAveragePooling2D()(block1_layer3)
        sq_3 = self.block1_dense5(sq_3)
        sq_3 = ReLU()(sq_3)
        sq_3 = self.block1_dense6(sq_3)
        sq_3 = tf.keras.activations.sigmoid(sq_3)
        exci_3 = multiply([block1_layer3, sq_3])
        block1_layer3 = add([block1_layer2, exci_3])
        block1_layer3 = ReLU()(block1_layer3)
        block1_layer3 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(block1_layer3)
        # block 2
        ## layer 1
        block2_layer1 = self.block2_conv0(block1_layer3)
        block2_layer1 = self.block2_batchnorm_1(block2_layer1)
        block2_layer1 = ReLU()(block2_layer1)
        block2_layer1 = self.block2_conv1(block2_layer1)
        block2_layer1 = self.block2_batchnorm_2(block2_layer1)
        ## transition
        block1_layer3 = self.transition_1(block1_layer3)
        sq_4 = GlobalAveragePooling2D()(block2_layer1)
        sq_4 = self.block2_dense1(sq_4)
        sq_4 = ReLU()(sq_4)
        sq_4 = self.block2_dense2(sq_4)
        sq_4 = tf.keras.activations.sigmoid(sq_4)
        exci_4 = multiply([block2_layer1, sq_4])
        block2_layer1 = add([block1_layer3, exci_4])
        block2_layer1 = ReLU()(block2_layer1)
        ## layer 2
        block2_layer2 = self.block2_conv2(block2_layer1)
        block2_layer2 = self.block2_batchnorm_3(block2_layer2)
        block2_layer2 = ReLU()(block2_layer2)
        block2_layer2 = self.block2_conv3(block2_layer2)
        block2_layer2 = self.block2_batchnorm_4(block2_layer2)
        sq_5 = GlobalAveragePooling2D()(block2_layer2)
        sq_5 = self.block2_dense3(sq_5)
        sq_5 = ReLU()(sq_5)
        sq_5 = self.block2_dense4(sq_5)
        sq_5 = tf.keras.activations.sigmoid(sq_5)
        exci_5 = multiply([block2_layer2, sq_5])
        block2_layer2 = add([block2_layer1, exci_5])
        block2_layer2 = ReLU()(block2_layer2)
        ## layer 3
        block2_layer3 = self.block2_conv4(block2_layer2)
        block2_layer3 = self.block2_batchnorm_5(block2_layer3)
        block2_layer3 = ReLU()(block2_layer3)
        block2_layer3 = self.block2_conv5(block2_layer3)
        block2_layer3 = self.block2_batchnorm_6(block2_layer3)
        sq_6 = GlobalAveragePooling2D()(block2_layer3)
        sq_6 = self.block2_dense5(sq_6)
        sq_6 = ReLU()(sq_6)
        sq_6 = self.block2_dense6(sq_6)
        sq_6 = tf.keras.activations.sigmoid(sq_6)
        exci_6 = multiply([block2_layer3, sq_6])
        block2_layer3 = add([block2_layer2, exci_6])
        block2_layer3 = ReLU()(block2_layer3)
        block2_layer3 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(block2_layer3)
        # block 3
        ## layer 1
        block3_layer1 = self.block3_conv0(block2_layer3)
        block3_layer1 = self.block3_batchnorm_1(block3_layer1)
        block3_layer1 = ReLU()(block3_layer1)
        block3_layer1 = self.block3_conv1(block3_layer1)
        block3_layer1 = self.block3_batchnorm_2(block3_layer1)
        ## transition
        block2_layer3 = self.transition_2(block2_layer3)
        sq_7 = GlobalAveragePooling2D()(block3_layer1)
        sq_7 = self.block3_dense1(sq_7)
        sq_7 = ReLU()(sq_7)
        sq_7 = self.block3_dense2(sq_7)
        sq_7 = tf.keras.activations.sigmoid(sq_7)
        exci_7 = multiply([block3_layer1, sq_7])
        block3_layer1 = add([block2_layer3, exci_7])
        block3_layer1 = ReLU()(block3_layer1)
        ## layer 2
        block3_layer2 = self.block3_conv2(block3_layer1)
        block3_layer2 = self.block3_batchnorm_3(block3_layer2)
        block3_layer2 = ReLU()(block3_layer2)
        block3_layer2 = self.block3_conv3(block3_layer2)
        block3_layer2 = self.block3_batchnorm_4(block3_layer2)
        sq_8 = GlobalAveragePooling2D()(block3_layer2)
        sq_8 = self.block3_dense3(sq_8)
        sq_8 = ReLU()(sq_8)
        sq_8 = self.block3_dense4(sq_8)
        sq_8 = tf.keras.activations.sigmoid(sq_8)
        exci_8 = multiply([block3_layer2, sq_8])
        block3_layer2 = add([block3_layer1, exci_8])
        block3_layer2 = ReLU()(block3_layer2)
        ## layer 3
        block3_layer3 = self.block3_conv4(block3_layer2)
        block3_layer3 = self.block3_batchnorm_5(block3_layer3)
        block3_layer3 = ReLU()(block3_layer3)
        block3_layer3 = self.block3_conv5(block3_layer3)
        block3_layer3 = self.block3_batchnorm_6(block3_layer3)
        sq_9 = GlobalAveragePooling2D()(block3_layer3)
        sq_9 = self.block3_dense5(sq_9)
        sq_9 = ReLU()(sq_9)
        sq_9 = self.block3_dense6(sq_9)
        sq_9 = tf.keras.activations.sigmoid(sq_9)
        exci_9 = multiply([block3_layer3, sq_9])
        block3_layer3 = add([block3_layer2, exci_9])
        block3_layer3 = ReLU()(block3_layer3)
        ## layer 4
        block3_layer4 = self.block3_conv6(block3_layer3)
        block3_layer4 = self.block3_batchnorm_7(block3_layer4)
        block3_layer4 = ReLU()(block3_layer4)
        block3_layer4 = self.block3_conv7(block3_layer4)
        block3_layer4 = self.block3_batchnorm_8(block3_layer4)
        sq_10 = GlobalAveragePooling2D()(block3_layer4)
        sq_10 = self.block3_dense7(sq_10)
        sq_10 = ReLU()(sq_10)
        sq_10 = self.block3_dense8(sq_10)
        sq_10 = tf.keras.activations.sigmoid(sq_10)
        exci_10 = multiply([block3_layer4, sq_10])
        block3_layer4 = add([block3_layer3, exci_10])
        block3_layer4 = ReLU()(block3_layer4)
        # block 4
        ## layer 1
        block4_layer1 = self.block4_conv0(block2_layer3)
        block4_layer1 = self.block4_batchnorm_1(block4_layer1)
        block4_layer1 = ReLU()(block4_layer1)
        block4_layer1 = self.block4_conv1(block4_layer1)
        block4_layer1 = self.block4_batchnorm_2(block4_layer1)
        ## transition
        block3_layer4 = self.transition_3(block3_layer4)
        sq_11 = GlobalAveragePooling2D()(block4_layer1)
        sq_11 = self.block4_dense1(sq_11)
        sq_11 = ReLU()(sq_11)
        sq_11 = self.block4_dense2(sq_11)
        sq_11 = tf.keras.activations.sigmoid(sq_11)
        exci_11 = multiply([block4_layer1, sq_11])
        block4_layer1 = add([block3_layer4, exci_11])
        block4_layer1 = ReLU()(block4_layer1)
        ## layer 2
        block4_layer2 = self.block4_conv2(block4_layer1)
        block4_layer2 = self.block4_batchnorm_3(block4_layer2)
        block4_layer2 = ReLU()(block4_layer2)
        block4_layer2 = self.block4_conv3(block4_layer2)
        block4_layer2 = self.block4_batchnorm_4(block4_layer2)
        sq_12 = GlobalAveragePooling2D()(block4_layer2)
        sq_12 = self.block4_dense3(sq_12)
        sq_12 = ReLU()(sq_12)
        sq_12 = self.block4_dense4(sq_12)
        sq_12 = tf.keras.activations.sigmoid(sq_12)
        exci_12 = multiply([block4_layer2, sq_12])
        block4_layer2 = add([block4_layer1, exci_12])
        block4_layer2 = ReLU()(block4_layer2)
        ## layer 3
        block4_layer3 = self.block4_conv4(block4_layer2)
        block4_layer3 = self.block4_batchnorm_5(block4_layer3)
        block4_layer3 = ReLU()(block4_layer3)
        block4_layer3 = self.block4_conv5(block4_layer3)
        block4_layer3 = self.block4_batchnorm_6(block4_layer3)
        sq_13 = GlobalAveragePooling2D()(block4_layer3)
        sq_13 = self.block4_dense5(sq_13)
        sq_13 = ReLU()(sq_13)
        sq_13 = self.block4_dense6(sq_13)
        sq_13 = tf.keras.activations.sigmoid(sq_13)
        exci_13 = multiply([block4_layer3, sq_13])
        block4_layer3 = add([block4_layer2, exci_13])
        block4_layer3 = ReLU()(block4_layer3)

        x = GlobalAveragePooling2D()(block4_layer3)
        x = Flatten()(x)
        return self.classifier(x)
