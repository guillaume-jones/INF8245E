import pickle
import numpy as np
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as layers
import keras_tuner as kt
import matplotlib.pyplot as plt
import math


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

    # Split out a validation set
    x_train_partial, x_valid, y_train_partial, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=1)

    return x_train, y_train, x_train_partial, y_train_partial, x_valid, y_valid
    
def load_train_as_dataset(return_complete_set=False):
    """
    Convert numpy or other dataset to TensorFlow Dataset
    Batch using batch_size
    """
    x_complete, y_complete, x_train, y_train, x_valid, y_valid = load_train_set()
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    if return_complete_set:
        complete_train_dataset = tf.data.Dataset.from_tensor_slices((x_complete, y_complete))
        return complete_train_dataset, train_dataset, valid_dataset, y_valid
    
    return train_dataset, valid_dataset, y_valid

def augment_dataset(dataset, batch_size):
    """
    Augment images from training set with standard augmentations
    Also repeat and shuffle data for good training
    """
    epoch_length = math.ceil(len(dataset) / batch_size)
    dataset = dataset.shuffle(len(dataset))

    dataset = dataset.batch(batch_size)

    dataset = dataset.repeat()

    augmentation = tf.keras.Sequential()
    augmentation.add(layers.RandomFlip(mode='horizontal'))
    augmentation.add(layers.RandomRotation(0.1))
    augmentation.add(layers.RandomTranslation((-0.25, 0.25), (-0.25, 0.25)))
    # augmentation.add(layers.RandomZoom(.05, .05)) # Breaks certain models completely (0 learning)
    augmentation.add(layers.RandomContrast(0.7))

    dataset = dataset.map(
        lambda image, y: (augmentation(image, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE), epoch_length

def show_images(dataset, count):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(count):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy(), cmap=plt.cm.gray)
            plt.title(labels[i].numpy()[0])
            plt.axis("off")
        plt.show()

def print_accuracy(y_true, y_pred):
    """
    Wrapper function to print accuracy quickly
    """
    accuracy = f1_score(list(y_true), list(y_pred), average='micro')
    print(f'Accuracy: {accuracy:.4f}')

def plot_model_history(history, labels_to_plot=[]):
    """
    Plots the history of specified labels during model training
    """
    label_count = len(labels_to_plot)
    if label_count > 0:
        fig, axes = plt.subplots(1, label_count, figsize=(5*label_count, 5))
        for axis, label in zip(axes, labels_to_plot):
            if(isinstance(label, list)):
                for sub_label in label:
                    axis.plot(history.history[sub_label], label=sub_label)
                axis.legend(loc="lower right")
            else:
                axis.plot(history.history[label], label=label)
            if 'accuracy' in label:
                axis.set_ylim([0, 1])
            axis.set_xlabel('Epoch')
            axis.set_ylabel(label)
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


def train_model(model, dataset, valid_dataset, epochs, valid_patience, epoch_length=None):
    """
    Trains models from scratch
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=valid_patience),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=round(valid_patience*0.5), 
            min_lr=5E-6, verbose=1)
    ]
    try:
        history = model.fit(
            dataset, validation_data=valid_dataset.batch(128).cache(),
            epochs=epochs, steps_per_epoch=epoch_length, 
            callbacks=callbacks, verbose=1)
    except KeyboardInterrupt:
        print('Training interrupted')
        return model, None

    return model, history

def fine_tune_model(
    model, dataset, valid_dataset, epochs, 
    learning_rate=None, valid_patience=None, epoch_length=None):
    """
    Fine-tune existing models. Uses epoch_length if using an infinite dataset (like augmented), otherwise set to None
    """
    if learning_rate is not None:
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    if valid_patience is not None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=valid_patience),  
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=round(valid_patience*0.5), 
                min_lr=5E-6, verbose=1)
        ]
    else:
        callbacks=[]
    try:
        history = model.fit(
            dataset, validation_data=valid_dataset.batch(128).cache(),
            epochs=epochs, steps_per_epoch=epoch_length, 
            callbacks=callbacks, verbose=1)
    except KeyboardInterrupt:
        print('Fine-tuning interrupted.')
        return model, None

    return model, history

def fine_tune_model_filepath(
    model_filepath, dataset, valid_dataset, epochs, 
    learning_rate=None, valid_patience=None, epoch_length=None):
    """
    Load model from file before fine-tuning
    """
    model = tf.keras.models.load_model(model_filepath)
    model, history = fine_tune_model(
        model, dataset, valid_dataset, epochs, 
        learning_rate=learning_rate, valid_patience=valid_patience, 
        epoch_length=epoch_length)

    return model, history

def load_hypertuner(model, model_number, tuner_filepath, tuner_type='random'):
    if tuner_type == 'bayesian':
        tuner = kt.BayesianOptimization(model,
            objective='val_accuracy',
            directory=f'models/{model_number}',
            project_name=tuner_filepath)
    elif tuner_type == 'random':
        tuner = kt.RandomSearch(model,
            objective='val_accuracy',
            directory=f'models/{model_number}',
            project_name=tuner_filepath)

    tuner.results_summary()
    return tuner

def hypertune_model(
    model, dataset, valid_dataset, model_number, tuner_filepath, epochs, trials,  
    tuner_type='random', valid_patience=None, epoch_length=None):

    tuner_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=valid_patience),  
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=round(valid_patience*0.5), 
            min_lr=5E-6, verbose=1)
    ]
    if tuner_type == 'bayesian':
        tuner = kt.BayesianOptimization(model,
            objective='val_accuracy',
            max_trials=trials,
            seed=2,
            directory=f'models/{model_number}',
            project_name=tuner_filepath,
            overwrite=True)
    elif tuner_type == 'random':
        tuner = kt.RandomSearch(model,
            objective='val_accuracy',
            max_trials=trials,
            seed=2,
            directory=f'models/{model_number}',
            project_name=tuner_filepath,
            overwrite=True)

    tuner.search_space_summary()
    try:
        tuner.search(
            dataset, 
            validation_data=valid_dataset.batch(128).cache(),
            epochs=epochs, steps_per_epoch=epoch_length,
            callbacks=tuner_callbacks, verbose=1)
    except KeyboardInterrupt:
        print('Tuner interrupted.')

    tuner.results_summary()

    return tuner

    

def save_test_pred(filename, array):
    """
    Saves tests for kaggle submission
    """
    array_with_ids = np.c_[np.arange(0, len(array)), array]
    np.savetxt(
        filename, array_with_ids, header='Id,class', comments='',
        delimiter = ',', fmt='%d', newline='\n')

def generate_test_pred(model, pred_filepath):
    x_test_real = load_test_set()
    true_test_pred = np.argmax(model.predict(x_test_real), axis=1)

    save_test_pred(pred_filepath, true_test_pred)

def generate_test_pred_filepath(model_filepath):
    try:
        model = tf.keras.models.load_model(model_filepath)
        print('Model found, generating predictions...')
    except:
        print('No model at filepath.')
        return

    generate_test_pred(model, f'{model_filepath}_test_pred.csv')

def is_tf_using_gpus():
    """
    Check that TensorFlow is using a GPU
    """
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 0:
        print(f'TensorFlow is using a GPU. Number of GPUs available: {gpus}')
    else:
        print(f'TensorFlow is not using any GPUs.')
