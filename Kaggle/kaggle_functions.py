import pickle
import numpy as np
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.ops.gen_logging_ops import image_summary

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
        return complete_train_dataset, partial_train_dataset, fake_test_dataset, y_test_fake
    
    return partial_train_dataset, fake_test_dataset, y_test_fake

def augment_dataset(dataset, batch_size):
    """
    Augment images from training set with standard augmentations
    Also repeat and shuffle data for good training
    """
    dataset = dataset.shuffle(len(dataset))

    dataset = dataset.batch(batch_size)

    dataset = dataset.repeat()

    augmentation = tf.keras.Sequential()
    augmentation.add(layers.RandomFlip())
    # Adding rotation and translation simultaneously creates horrible images
    if tf.random.uniform([1]) > 0.5:
        augmentation.add(layers.RandomRotation(0.25))
    else:
        augmentation.add(layers.RandomTranslation((-0.3, 0.3), (-0.3, 0.3)))
    augmentation.add(layers.RandomContrast(0.4))

    dataset = dataset.map(
        lambda image, y: (augmentation(image, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=AUTOTUNE)

def show_images(dataset):
    plt.figure(figsize=(10, 10))
    for image, label in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image[i].numpy(), cmap=plt.cm.gray)
            plt.title(label[i].numpy()[0])
            plt.axis("off")

def print_accuracy(y_true, y_pred):
    """
    Wrapper function to print accuracy quickly
    """
    accuracy = accuracy_score(list(y_true), list(y_pred))
    print(f'Accuracy: {accuracy:.4f}')

def plot_model_history(history, labels_to_plot=[]):
    """
    Plots the history of specified labels during model training
    """
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
    """
    Pretty-plots a confusion matrix with corresponding labels
    """
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
    """
    Saves tests for kaggle submission
    """
    array_with_ids = np.c_[np.arange(0, len(array)), array]
    np.savetxt(
        filename, array_with_ids, header='Id,class', comments='',
        delimiter = ',', fmt='%d', newline='\n')

def is_tf_using_gpus():
    """
    Check that TensorFlow is using a GPU
    """
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 0:
        print(f'TensorFlow is using a GPU. Number of GPUs available: {gpus}')
    else:
        print(f'TensorFlow is not using any GPUs.')
