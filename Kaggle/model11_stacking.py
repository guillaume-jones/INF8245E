from numpy.lib.arraysetops import isin
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt
import numpy as np

# Epochs for training : 10-20
# Good validation patience : 2-4
# Good fine-tuning lr : 

model_number = 'model11'

def load_base_model(inputs, model_filepath):
    base_model = models.load_model(model_filepath)
    base_model._name = model_filepath
    base_model.trainable = False

    return layers.Softmax()(base_model(inputs))

def generate_predictions_for_datasets(datasets, test_index=2):
    # Same input for all models (accepts augmented data)
    input_layer = layers.Input(shape=(96, 96, 1))

    # Loads all models we want
    print('Loading base models')
    base_model_filepaths = [
        'model6/VGG_6_79',
        'model7/DeeperVGG_4_79',
        'model7/DeeperVGG_6_84',
        'model8/VGGRes_3_79',
        'model9/DeeperVGG2_2_84',
        'model10/WideResNet_3_80',
        'model10/WideResNet_5_84',
        'model10/WideResNet_8_83',
        'model14/SimpleNet_5_87',
        'model14/SimpleNet_6_87',
        'model14/SimpleNet_8_89'
    ]
    base_models = []
    for filepath in base_model_filepaths:
        base_models.append(load_base_model(input_layer, f'models/{filepath}'))

    # Concatenates models before final trained layers
    output = layers.concatenate(base_models)

    model = models.Model(
        inputs=input_layer, outputs=output, name='stacked_model')

    predicted_datasets = []
    for index, dataset in enumerate(datasets):
        print(f'Predicting for set {index + 1}')
        predictions = model.predict(dataset)
        if index == test_index:
            predicted_datasets.append(tf.data.Dataset.from_tensor_slices(
                (predictions)).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE))
        else:
            labels = np.reshape(np.concatenate([y for _, y in dataset], axis=0), (-1, 1))
            predicted_datasets.append(tf.data.Dataset.from_tensor_slices(
                (predictions, labels)).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE))

    return predicted_datasets

class Model(kt.HyperModel):
    def dense_layer(self, inputs, size, l2_reg=0, dropout=0, activation='relu'):
        bn = layers.BatchNormalization()(inputs)
        dropout = layers.Dropout(dropout)(bn)
        return layers.Dense(
            size, activation=activation,
            kernel_regularizer=reg.l2(l2_reg), 
            bias_regularizer=reg.l2(l2_reg))(dropout)

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None:
            l2_reg = hyperparameters.Float('l2_reg', 1E-6, 1E-1, sampling='log')
            dropout = hyperparameters.Float('dropout', 0.1, 0.6)
        else:
            l2_reg = 5E-1
            dropout = 0

        # Fixed hyperparameters
        learning_rate = 1E-4

        # Use already predicted data as input for stacking
        # Input should be num_models*11
        input_layer = layers.Input(shape=(12*11))

        # Applies dense layers on top of concatenated models
        output = self.dense_layer(input_layer, 512, l2_reg, dropout)
        output = self.dense_layer(output, 256, l2_reg, dropout)
        output = self.dense_layer(output, 128, l2_reg, dropout)
        output = self.dense_layer(output, 11, l2_reg, dropout, activation=None)

        model = models.Model(inputs=input_layer, outputs=output, name='stacked_model')

        # Create model
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
