import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

class Model(kt.HyperModel):
    def conv_layer(self, input, filters, stride=1, kernel=3, l2_reg=0.0001, padding='same'):
        return layers.Conv2D(
            filters, kernel_size=(kernel,kernel), strides=(stride,stride), 
            padding=padding, activation='relu',
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_uniform')(input)

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None:
            dense_l2_reg = hyperparameters.Float('dense_l2_reg', 0.0001, 0.01, sampling='log')
            dense_dropout = hyperparameters.Float('dense_dropout', 0.2, 0.6, step=0.2)
        else:
            dense_l2_reg = 0.0008
            dense_dropout = 0.4

        # Fixed hyperparameters
        learning_rate = 0.0002

        input_layer = layers.Input(shape=(96, 96, 1))

        output = self.conv_layer(input_layer, 32, stride=2)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 32)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 64, stride=2)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 64)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 128, stride=2)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 128)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 256, stride=2)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 256)
        output = layers.Dropout(dense_dropout / 8)(output)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 512, stride=2)
        output = layers.Dropout(dense_dropout / 4)(output)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 1024, padding='valid')
        output = layers.BatchNormalization()(output)

        # Final output
        output = layers.Flatten()(output)
        output = layers.Dropout(dense_dropout / 2)(output)
        output = layers.Dense(
            256, activation='relu', kernel_initializer='he_uniform',
            kernel_regularizer=keras.regularizers.l2(dense_l2_reg),
            bias_regularizer=keras.regularizers.l2(dense_l2_reg))(output)
        output = layers.Dropout(dense_dropout)(output) 
        output = layers.Dense(
            11, kernel_regularizer=keras.regularizers.l2(dense_l2_reg),
            bias_regularizer=keras.regularizers.l2(dense_l2_reg))(output)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        # Create model
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
