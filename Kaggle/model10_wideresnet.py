import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 
# Good validation patience : 
# Good fine-tuning lr : 

class Model(kt.HyperModel):
    def residual_module(self, input, filters, stride=1, l2_reg=0, dropout=0.3):
        # Applies relu convolution, then linear convolution before shortcut
        bn_1 = layers.BatchNormalization()(input)
        conv_1 = layers.Conv2D(
            filters, kernel_size=(3,3), strides=(stride, stride),
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=reg.l2(l2_reg), use_bias=False)(bn_1)
        dropout = layers.Dropout(dropout)(conv_1)
        bn_2 = layers.BatchNormalization()(dropout)
        conv_2 = layers.Conv2D(
            filters, kernel_size=(3,3), strides=(1,1),
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=reg.l2(l2_reg), use_bias=False)(bn_2)
        
        # Ensures shortcut is correct depth by adding a 1x1 convolution
        if input.shape[-1] != filters:
            shortcut_bn = layers.BatchNormalization()(input)
            shortcut_relu = layers.ReLU(negative_slope=0)(shortcut_bn)
            shortcut = layers.Conv2D(
                filters, kernel_size=(3,3), strides=(stride,stride),
                padding='same', kernel_initializer='he_normal',
                kernel_regularizer=reg.l2(l2_reg), use_bias=False)(shortcut_relu)
        else:
            shortcut = input

        # Adds shortcut to convolutions
        return layers.add([conv_2, shortcut])

    def conv_layer(self, input, filters, stride=1, l2_reg=0):
        return layers.Conv2D(
            filters, kernel_size=(3,3), strides=(stride,stride), 
            padding='same', activation='relu',
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_normal')(input)

    def prediction_layer(self, input, size, l2_reg, dropout):
        dropout_layer = layers.Dropout(dropout)(input)
        dense = layers.Dense(
            size, activation='linear',
            kernel_regularizer=reg.l2(l2_reg), 
            bias_regularizer=reg.l2(l2_reg))(dropout_layer)
        return dense

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None:
            conv_dropout = hyperparameters.Float('spatial_dropout', 0.0, 0.6, step=0.2)
            l2_reg = hyperparameters.Float('dense_l2_reg', 0.0001, 0.01, sampling='log')
            k = hyperparameters.Int('k', 2, 10, step=2)
            n = hyperparameters.Int('n', 1, 4)
        else:
            conv_dropout = 0.3
            l2_reg = 0.00001
            k=8
            n=3

        # Fixed hyperparameters
        dense_dropout = 0.3
        learning_rate = 0.001

        input_layer = layers.Input(shape=(96, 96, 1))

        # WideResNet, conv group 1 (input)
        output = self.conv_layer(input_layer, 16, l2_reg=l2_reg)

        # WideResNet, conv group 2
        output = self.residual_module(output, 16 * k, 2, l2_reg=l2_reg, dropout=conv_dropout)
        for i in range(n - 1):
            output = self.residual_module(output, 16 * k, l2_reg=l2_reg, dropout=conv_dropout)

        # WideResNet, conv group 3
        output = self.residual_module(output, 32 * k, 2, l2_reg=l2_reg, dropout=conv_dropout)
        for i in range(n - 1):
            output = self.residual_module(output, 32 * k, l2_reg=l2_reg, dropout=conv_dropout)
        
        # WideResNet, conv group 4
        output = self.residual_module(output, 64 * k, 2, l2_reg=l2_reg, dropout=conv_dropout)
        for i in range(n - 1):
            output = self.residual_module(output, 64 * k, l2_reg=l2_reg, dropout=conv_dropout)
        
        # Final average pool and prediction
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)
        output = layers.GlobalAveragePooling2D()(output)
        output = self.prediction_layer(output, 11, l2_reg, dense_dropout)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        # Create model
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
