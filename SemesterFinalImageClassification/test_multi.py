import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add, multiply, Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class test_se(tf.keras.Model):
    def __init__(self, k = 64, d=8, label_size=10, output_activation='softmax'):
        super(test_se, self).__init__()
        self.conv1 = Conv2D(k, kernel_size=(7,7),strides=1, padding='same')
        self.conv2 = Conv2D(2*k, kernel_size=(3,3), strides=1, padding='same')
        self.dense1 = Dense(10, activation='relu')
        self.classifier = Dense(label_size, activation=output_activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        y = x
        y = GlobalAveragePooling2D()(y)
        x = tf.keras.layers.multiply([x,y])
        x = ReLU()(x)
        x = Flatten()(x)
        x = self.classifier(x)
        return x