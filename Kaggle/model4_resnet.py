import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 100
# Good validation patience : 10
# Good fine-tuning lr : ?

class Model(kt.HyperModel):
    def residual_module(self, input, filters, stride=1, bottleneck=0, l2_reg=0.0001, batch_norm=0.99):
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
        if batch_norm >= 0:
            addition = layers.BatchNormalization(momentum=batch_norm)(addition)

        activation = layers.Activation('relu')(addition)
        return activation

    def conv_layer(self, input, filters, kernel, stride, l2_reg=0.0001):
        return layers.Conv2D(
            filters, kernel_size=(kernel,kernel), strides=(stride,stride), 
            padding='same', activation='relu',
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_normal')(input)

    def dense_layer(self, input, size, l2_reg, dropout):
        dense = layers.Dense(
            size, activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg))(input)
        return layers.Dropout(dropout)(dense)

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None:
            spatial_dropout = hyperparameters.Float('spatial_dropout', 0.0, 0.6, step=0.1)
            dense_l2_reg = hyperparameters.Float('dense_l2_reg', 0.001, 0.1, step=0.001, sampling='log')
        else:
            spatial_dropout = 0.3 # Not in original paper
            dense_l2_reg = 0.01

        # Fixed hyperparameters
        dense_dropout = 0.7 # Not in original paper
        conv_dropout = 0.3 # Not in original paper
        learning_rate = 0.0001

        input_layer = layers.Input(shape=(96, 96, 1))

        # ResNet, conv1 (input)
        output = self.conv_layer(input_layer, 64, 7, 2)

        # ResNet, conv2
        output = self.residual_module(output, 64)
        output = layers.SpatialDropout2D(spatial_dropout)(output)
        output = self.residual_module(output, 64)

        # ResNet, conv3
        output = self.residual_module(output, 128, 2)
        output = layers.SpatialDropout2D(spatial_dropout)(output)
        output = self.residual_module(output, 128)

        # ResNet, conv4
        output = self.residual_module(output, 256, 2)
        output = layers.Dropout(conv_dropout)(output)
        output = self.residual_module(output, 256)
        
        # ResNet, conv5
        output = self.residual_module(output, 512, 2)
        output = layers.Dropout(conv_dropout)(output)
        output = self.residual_module(output, 512)

        # Final output
        output = layers.GlobalAveragePooling2D()(output)
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
