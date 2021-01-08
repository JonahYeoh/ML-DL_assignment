# SE-RESNET
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add, multiply, Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU

class j_resnet_se(tf.keras.Model):
    def __init__(self, k = 64, r=8, label_size=10, output_activation='softmax'):
        super(j_resnet_se, self).__init__()
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
        x = GlobalAveragePooling2D()(block3_layer3)
        x = Flatten()(x)
        return self.classifier(x)


"""
epoch = 100
patience = epoch//4
monitor = val_loss
optimizer = adam
output_activation = sigmoid
========================================================================
k=32	loss	accuracy	precision	recall	f1-score
0	0.18853	0.94661	0.99421	0.53859	0.698684191
1	0.22606	0.9449	0.99083	0.7165	0.831625632
2	0.21023	0.94738	0.97707	0.79693	0.877853884
3	0.17737	0.9359	0.98994	0.63482	0.773571125
4	0.15419	0.94934	0.99642	0.39548	0.566224846
mean	0.191276	0.944826	0.989694	0.616464	0.749591936
min	0.15419	0.9359	0.97707	0.39548	0.566224846
max	0.22606	0.94934	0.99642	0.79693	0.877853884
stdev	0.028042243	0.005238338	0.007522462	0.156373279	0.122444624
batch 100					
					
k=32	loss	accuracy	precision	recall	f1-score
0	0.1567	0.95176	0.98591	0.74521	0.848826183
1	0.14233	0.94744	0.98604	0.70247	0.82044349
2	0.13782	0.9494	0.98968	0.70964	0.826585358
3	0.16108	0.94904	0.99318	0.63761	0.776631571
4	0.16829	0.94596	0.98941	0.66915	0.798359663
mean	0.153244	0.94872	0.988844	0.692816	0.814169253
min	0.13782	0.94596	0.98591	0.63761	0.776631571
max	0.16829	0.95176	0.99318	0.74521	0.848826183
stdev	0.01281351	0.002183025	0.003012097	0.041015728	0.027629917
batch 50/1/50					
					
k=48	loss	accuracy	precision	recall	f1-score
0	0.16951	0.95082	0.98709	0.75574	0.856059853
1	0.16547	0.9462	0.97863	0.78876	0.873496171
2	0.21247	0.94851	0.98766	0.77219	0.866734296
3	0.17046	0.95265	0.98848	0.69573	0.816662044
4	0.15702	0.94667	0.97883	0.8428	0.905735988
mean	0.174986	0.94897	0.984138	0.771044	0.86373767
min	0.15702	0.9462	0.97863	0.69573	0.816662044
max	0.21247	0.95265	0.98848	0.8428	0.905735988
stdev	0.021615116	0.002744968	0.004961972	0.053316284	0.032183576
batch 50/1/50					
					
k=64	loss	accuracy	precision	recall	f1-score
0	0.16479	0.94164	0.98544	0.78504	0.873898398
1	0.1773	0.95076	0.98187	0.81096	0.888268598
2	0.15336	0.9459	0.97835	0.78646	0.87197278
3	0.17588	0.94798	0.9799	0.81084	0.8873897
4	0.14244	0.95259	0.98189	0.73165	0.838497868
mean	0.162754	0.947774	0.98149	0.78499	0.872005469
min	0.14244	0.94164	0.97835	0.73165	0.838497868
max	0.1773	0.95259	0.98544	0.81096	0.888268598
stdev	0.014907457	0.004280488	0.002659633	0.032365037	0.020171189
batch 50/1/50					

"""
