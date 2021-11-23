import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
import tensorflow as tf
import keras_tuner as kt

def open_pickled_file(filename):
    with open(filename, 'rb') as x_train_pickle:
        return np.array(pickle.load(x_train_pickle))

def load_dataset():
    # Open x_train and x_test
    x_train = open_pickled_file('data/x_train.pkl') / 255.0
    x_test = open_pickled_file('data/x_test.pkl') / 255.0
    n_examples = len(x_train)

    # Open y_train and convert to numbers
    y_dictionary = {'big_cats':0, 'butterfly':1, 'cat':2, 'chicken':3, 'cow':4, 'dog':5, 
        'elephant':6, 'goat':7, 'horse':8, 'spider':9, 'squirrel':10}
    y_train_names = open_pickled_file('data/y_train.pkl')
    y_train = np.zeros(y_train_names.shape, dtype=int)
    for index, name in enumerate(y_train_names):
        y_train[index] = y_dictionary[name]

    return x_train, y_train, x_test
    
def print_f1_micro(y_pred, y_true, title):
    f1 = f1_score(list(y_pred), list(y_true), average='micro')
    print(title + f', F1-micro: {f1:.3f}')

def flatten(x):
    return np.reshape(x, (x.shape[0], -1))

def is_tf_using_gpus():
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 0:
        print(f'TensorFlow is using a GPU. Number of GPUs available: {gpus}')
    else:
        print(f'TensorFlow is not using any GPUs.')
