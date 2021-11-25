import pickle
import numpy as np
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

def get_label_dictionary():
    return {'big_cats':0, 'butterfly':1, 'cat':2, 'chicken':3, 'cow':4, 'dog':5, 
        'elephant':6, 'goat':7, 'horse':8, 'spider':9, 'squirrel':10}

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
    y_dictionary = get_label_dictionary()
    y_train = np.zeros(y_train_raw.shape, dtype=int)
    for index, name in enumerate(y_train_raw):
        y_train[index] = y_dictionary[name]

    # Add extra dimension to images for conversion to dataset
    y_train = np.reshape(y_train, (-1, 1))

    x_train_partial, x_test_fake, y_train_partial, y_test_fake = train_test_split(
        x_train, y_train, test_size=0.2, random_state=1)

    return x_train, y_train, x_train_partial, y_train_partial, x_test_fake, y_test_fake
    
def load_train_as_dataset(return_complete_set=False):
    """
    Convert numpy or other dataset to TensorFlow Dataset
    Batch using batch_size
    """
    x_train, y_train, x_train_partial, y_train_partial, x_test_fake, y_test_fake = load_train_set()

    
    partial_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_partial, y_train_partial))
    fake_test_dataset = tf.data.Dataset.from_tensor_slices((x_test_fake, y_test_fake))

    if return_complete_set:
        complete_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        return complete_train_dataset, partial_train_dataset, fake_test_dataset
    
    return partial_train_dataset, fake_test_dataset

def print_accuracy(y_true, y_pred):
    """
    Wrapper function to print accuracy quickly
    """
    accuracy = accuracy_score(list(y_true), list(y_pred))
    print(f'Accuracy: {accuracy:.4f}')

def plot_model_history(history, labels_to_plot=[]):
    label_count = len(labels_to_plot)
    if label_count > 0:
        fig, axes = plt.subplots(1, label_count, figsize=(4*label_count, 4))
        for axis, label in zip(axes, labels_to_plot):
            axis.plot(history.history[label], label=label)
            axis.set_xlabel('Epoch')
            axis.set_ylabel(label)
            if 'accuracy' in label:
                axis.set_ylim([0, 1])
            axis.grid(True)

def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        normalize='true',
        cmap=plt.cm.BuPu,
        colorbar=False,
        display_labels=get_label_dictionary().keys()
    )
    confusion_matrix_display.figure_.set_size_inches(10, 10)
    plt.show()

def save_test_pred(filename, array):
    # Outputs a np array in .csv format
    array_with_ids = np.c_[np.arange(0, len(array)), array]
    np.savetxt(
        filename, array_with_ids, header='Id,class', comments='',
        delimiter = ',', fmt='%d', newline='\n')

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
