import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

def open_pickled_file(filename):
    """
    Return np array from pickle file
    """
    with open(filename, 'rb') as file:
        return np.array(pickle.load(file))

def load_test_set():
    """
    Open Kaggle competition image dataset
    Scale images to floats and replace labels with class numbers
    """
    return open_pickled_file('data/x_test.pkl') / 255.0


def load_train_set():
    """
    Open Kaggle competition image dataset
    Scale images to floats and replace labels with class numbers
    Split out a testing set with labels
    """
    # Open x_train and x_test
    x_train = open_pickled_file('data/x_train.pkl') / 255.0

    # Add extra dimension to images for "channels"
    x_train = np.reshape(x_train, (-1, 96, 96, 1))

    # Open y_train and convert to numbers
    y_train_raw = open_pickled_file('data/y_train.pkl')


    # Convert word labels to numbers
    y_dictionary = {'big_cats':0, 'butterfly':1, 'cat':2, 'chicken':3, 'cow':4, 'dog':5, 
        'elephant':6, 'goat':7, 'horse':8, 'spider':9, 'squirrel':10}
    y_train = np.zeros(y_train_raw.shape, dtype=int)
    for index, name in enumerate(y_train_raw):
        y_train[index] = y_dictionary[name]

    # Add extra dimension to images for conversion to dataset
    y_train = np.reshape(y_train, (-1, 1))

    x_train_partial, x_test_fake, y_train_partial, y_test_fake = train_test_split(
        x_train, y_train, test_size=0.2, random_state=1)

    return x_train, y_train, x_train_partial, y_train_partial, x_test_fake, y_test_fake
    
def load_train_as_dataset(batch_size=None):
    """
    Convert numpy or other dataset to TensorFlow Dataset
    Batch using batch_size
    """
    x_train, y_train, x_train_partial, y_train_partial, x_test_fake, y_test_fake = load_train_set()

    complete_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    partial_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_partial, y_train_partial))
    fake_test_dataset = tf.data.Dataset.from_tensor_slices((x_test_fake, y_test_fake))

    if batch_size is not None:
        return (complete_train_dataset.batch(batch_size),
            partial_train_dataset.batch(batch_size),
            fake_test_dataset.batch(batch_size))
    
    return complete_train_dataset, partial_train_dataset, fake_test_dataset

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
