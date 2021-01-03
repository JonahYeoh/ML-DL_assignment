#utility
import matplotlib.pyplot as plt
import numpy as np

def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def show_plot(history, train, validation):
    fig1, ax1 = plt.subplots(figsize=(12,5))  # Create a figure containing a single axes.
    ax1.plot(history[train], "m--")  # Plot some data on the axes.
    ax1.plot(history[validation], "b--")
    
    x = [i for i in range(len(history[train]))]
    train_trend = np.polyfit(x, history[train], 2)
    val_trend = np.polyfit(x, history[validation], 2)
    p_train = np.poly1d(train_trend)
    p_val = np.poly1d(val_trend)
    ax1.plot(x, p_train(x), "m-.")
    ax1.plot(x, p_val(x), "b-.")
    
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('{}'.format(train))
    ax1.legend(['train', 'validation'], loc='upper left')
    ax1.set_title('epoch vs {}'.format(train))