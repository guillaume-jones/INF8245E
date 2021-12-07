import tensorflow.keras as keras
from tensorflow.keras import models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 10-20
# Good validation patience : 2-4
# Good fine-tuning lr : 

model_number = 'model12'

class Model(kt.HyperModel):
    def load_base_model(self, inputs, model_filepath):
        base_model = models.load_model(model_filepath)
        base_model._name = model_filepath
        base_model.trainable = False

        return base_model(inputs, training=False)

    def load_autoencoders(self, inputs, model_name):
        autoencoders = []
        for i in range(11):
            base_autoencoder = self.load_base_model(
                inputs, f'models/{model_number}/{model_name}/{i}')
            autoencoders.append(self.difference_layer(inputs, base_autoencoder))
        return autoencoders

    def dense_layer(self, inputs, size, l2_reg=0, dropout=0, activation='elu'):
        output = layers.Dropout(dropout)(inputs)
        return layers.Dense(
            size, activation=activation,
            kernel_regularizer=reg.l2(l2_reg), 
            bias_regularizer=reg.l2(l2_reg))(output)

    def sum_abs_diff(tensors):
        image1, image2 = tensors
        return keras.backend.sum(keras.backend.abs(image1 - image2))
    
    def difference_layer(self, layer1, layer2):
        return layers.Lambda(Model.sum_abs_diff)([layer1, layer2])

    def build(self, hyperparameters):
        # Tunable hyperparameters
        # if hyperparameters is not None:
        #     dropout = hyperparameters.Float('dropout', 0.3, 0.6, step=0.3)
        #     l2_reg = hyperparameters.Choice('l2_reg', [0.0001, 0.001])

        # else:
        #     dropout = 0.3
        #     l2_reg = 0.0001

        # Fixed hyperparameters
        learning_rate = 0.00001
        model_name = 'Autoencoder_1'

        # Same input for all models
        input_layer = layers.Input(shape=(96, 96, 1))

        # Loads 11 models, 1 for each class
        output = self.load_autoencoders(input_layer, model_name)
        # Output should be 11 x 1, with higher = more likely
        output = layers.Softmax(output)

        model = keras.models.Model(inputs=input_layer, outputs=output, name='stacked_model')

        # Create model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        return model
