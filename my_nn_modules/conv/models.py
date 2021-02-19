'''

'''
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.metrics import Accuracy, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

from my_nn_modules.conv.densenet import densenet
from my_nn_modules.conv.csp_densenet import csp_densenet

# if __name__ == '__main__':
model = csp_densenet()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
print(model.summary())

