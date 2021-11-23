import pickle
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

def open_pickled_file(filename):
    """
    Return np array from pickle file
    """
    with open(filename, 'rb') as file:
        return np.array(pickle.load(file))

def load_numpy_dataset():
    """
    Open Kaggle competition image dataset
    Scale images to floats and replace labels with class numbers
    """
    # Open x_train and x_test
    x_train = open_pickled_file('data/x_train.pkl') / 255.0
    x_test = open_pickled_file('data/x_test.pkl') / 255.0

    # Open y_train and convert to numbers
    y_dictionary = {'big_cats':0, 'butterfly':1, 'cat':2, 'chicken':3, 'cow':4, 'dog':5, 
        'elephant':6, 'goat':7, 'horse':8, 'spider':9, 'squirrel':10}
    y_train_names = open_pickled_file('data/y_train.pkl')
    y_train = np.zeros(y_train_names.shape, dtype=int)
    for index, name in enumerate(y_train_names):
        y_train[index] = y_dictionary[name]

    return x_train, y_train, x_test
    
def load_dataset(batch_size=None):
    """
    Convert numpy or other dataset to TensorFlow Dataset
    Batch using batch_size
    """
    x_train, y_train, x_test = load_numpy_dataset()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, ))

    if batch_size is not None:
        return train_dataset.batch(batch_size), test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset

def print_f1_micro(y_pred, y_true, title):
    """
    Wrapper function to print f1-micro score quickly
    """
    f1 = f1_score(list(y_pred), list(y_true), average='micro')
    print(title + f', F1-micro: {f1:.3f}')

def flatten(x):
    """
    Wrapper function to flatten image array
    """
    return np.reshape(x, (x.shape[0], -1))

def is_tf_using_gpus():
    """
    Check that TensorFlow is using a GPU
    """
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 0:
        print(f'TensorFlow is using a GPU. Number of GPUs available: {gpus}')
    else:
        print(f'TensorFlow is not using any GPUs.')
