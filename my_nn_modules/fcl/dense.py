import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

def _fully_connected_layer(in_tensor, n_classes, layers=0, units_by_layer=[], activ_fn='relu', a_regularizer=None, k_regularizer=None, use_dropout=False, dropout_factor=None):
    dense = in_tensor
    for layer in range(layers):
        dense = Dense(units=units_by_layer[layer], activation=activ_fn, activity_regularizer=a_regularizer, kernel_regularizer=k_regularizer)(dense)
        if use_dropout:
            dense = Dropout(dropout_factor)(dense)
    return Dense(n_classes, activation='softmax')(dense)
