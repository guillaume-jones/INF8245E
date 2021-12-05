import tensorflow.keras as keras
from tensorflow.keras import models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 10-20
# Good validation patience : 2-4
# Good fine-tuning lr : 

class Model(kt.HyperModel):
    def load_base_model(self, inputs, model_filepath):
        base_model = models.load_model(model_filepath)
        base_model._name = model_filepath
        base_model.trainable = False

        return base_model(inputs, training=False)

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None:
            dropout = hyperparameters.Float('dropout', 0.3, 0.6, step=0.3)
            l2_reg = hyperparameters.Choice('l2_reg', [0.0001, 0.001])

        else:
            dropout = 0.3
            l2_reg = 0.0001

        # Fixed hyperparameters
        dense_activation = 'elu'
        learning_rate = 0.00005
        base_model_filepaths = [
            'model6/VGG_6_79',
            'model7/DeeperVGG_4_79',
            'model7/DeeperVGG_6_84',
            'model8/VGGRes_3_79',
            'model9/DeeperVGG2_2_84',
            'model10/WideResNet_4_81',
            'model10/WideResNet_5_84'
        ]

        # Same input for all models (accepts augmented data)
        input_layer = layers.Input(shape=(96, 96, 1))

        # Loads all models we want
        base_models = []
        for filepath in base_model_filepaths:
            base_models.append(self.load_base_model(input_layer, f'models/{filepath}'))

        # Concatenates models before 1 final learnt layer
        output = layers.concatenate(base_models)
        output = layers.Dropout(dropout)(output)
        output = layers.Dense(
            512, activation=dense_activation,
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg))(output)
        output = layers.Dropout(dropout)(output)
        output = layers.Dense(
            11, kernel_regularizer=reg.l2(l2_reg), 
            bias_regularizer=reg.l2(l2_reg))(output)

        model = keras.models.Model(inputs=input_layer, outputs=output, name='stacked_model')

        # Create model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
