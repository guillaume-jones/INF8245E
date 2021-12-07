import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 200+
# Good validation patience : 30
# Good fine-tuning lr : 1E-5

model_number = 'model8'

class Model(kt.HyperModel):
    def residual_module(self, input, filters, stride=1, bottleneck=0, l2_reg=0, batch_norm=0.99):
        # Applies bottleneck if necessary, to reduce dimensions
        if bottleneck > 0:
            conv_0 = layers.Conv2D(
                bottleneck, kernel_size=(1,1),
                padding='same', activation='relu',
                kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
                kernel_initializer='he_normal')(input)
        else:
            bottleneck = filters
            conv_0 = input

        # Applies relu convolution, then linear convolution before shortcut
        conv_1 = layers.Conv2D(
            bottleneck, kernel_size=(3,3), strides=(stride, stride),
            padding='same', activation='relu',
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_normal')(conv_0)
        conv_2 = layers.Conv2D(
            filters, kernel_size=(3,3), 
            padding='same', activation='linear',
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_normal')(conv_1)
        
        # Ensures shortcut is correct depth by adding a 1x1 convolution
        if input.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, kernel_size=(1,1), strides=(stride,stride),
                padding='same', activation='relu',
                kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
                kernel_initializer='he_normal')(input)
        else:
            shortcut = input

        # Adds shortcut
        addition = layers.add([conv_2, shortcut])

        # Batch Norm is performed in the original paper
        addition = layers.BatchNormalization(momentum=batch_norm)(addition)

        activation = layers.Activation('relu')(addition)
        return activation

    def conv_layer(self, input, filters, stride=1, kernel=3, l2_reg=0, padding='same'):
        return layers.Conv2D(
            filters, kernel_size=(kernel,kernel), strides=(stride,stride), 
            padding=padding, activation='relu',
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_uniform')(input)

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None: 
            dense_l2_reg = hyperparameters.Float('dense_l2_reg', 0.00001, 0.001, sampling='log')
            dense_dropout = hyperparameters.Float('dense_dropout', 0.3, 0.5, step=0.1)
        else:
            dense_l2_reg = 0.0001
            dense_dropout = 0.6

        # Fixed hyperparameters
        learning_rate = 0.0005
        conv_12_reg = 0.00001

        input_layer = layers.Input(shape=(96, 96, 1))

        output = self.conv_layer(input_layer, 32, stride=2, l2_reg=conv_12_reg)
        output = self.conv_layer(output, 32, l2_reg=conv_12_reg)
        output = layers.BatchNormalization()(output)
        
        output = self.conv_layer(output, 64, stride=2, l2_reg=conv_12_reg)
        output = self.conv_layer(output, 64, l2_reg=conv_12_reg)
        output = layers.BatchNormalization()(output)

        output = self.residual_module(output, 128, stride=2, l2_reg=conv_12_reg)

        output = self.residual_module(output, 256, stride=2, l2_reg=conv_12_reg)

        output = self.residual_module(
            output, 512, stride=2, bottleneck=256, l2_reg=conv_12_reg)

        # Final output
        output = layers.Flatten()(output)
        output = layers.Dropout(dense_dropout / 2)(output)
        output = layers.Dense(
            128, activation='relu', kernel_initializer='he_uniform',
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
