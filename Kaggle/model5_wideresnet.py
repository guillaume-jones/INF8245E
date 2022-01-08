import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 150+ (250-300 minutes)
# Good validation patience : 15-20
# Good fine-tuning lr : ?

model_number = 'model10'

class Model(kt.HyperModel):
    def residual_module(self, input, filters, stride=1, l2_reg=0, dropout=0.3):
        # Applies batch norm and relu to BOTH convolutions and shortcut
        bn_1 = layers.BatchNormalization()(input)
        relu_1 = layers.ReLU()(bn_1)

        # Convolution side 
        conv_1 = layers.Conv2D(
            filters, kernel_size=(3,3), strides=(stride, stride),
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=reg.l2(l2_reg), use_bias=False)(relu_1)
        dropout = layers.Dropout(dropout)(conv_1)
        bn_2 = layers.BatchNormalization()(dropout)
        relu_2 = layers.ReLU()(bn_2)
        conv_2 = layers.Conv2D(
            filters, kernel_size=(3,3), strides=(1,1),
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=reg.l2(l2_reg), use_bias=False)(relu_2)
        
        # Ensures shortcut is correct depth by adding a 1x1 convolution
        if input.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, kernel_size=(3,3), strides=(stride,stride),
                padding='same', kernel_initializer='he_normal',
                kernel_regularizer=reg.l2(l2_reg), use_bias=False)(relu_1)
        # If a convolution is not needed, no convolution is used for the shortcut
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
            l2_reg = hyperparameters.Float('l2_reg', 0.0001, 0.01, sampling='log')
            k = hyperparameters.Int('k', 10, 12, step=2)
            n = hyperparameters.Int('n', 2, 3, step=1)
            conv_dropout = hyperparameters.Float('conv_dropout', 0.4, 0.6, step=0.2)
        else:
            k=9
            n=2
            conv_dropout = 0.5
            l2_reg = 0.0001

        # Fixed hyperparameters
        dense_dropout = conv_dropout
        learning_rate = 0.0005

        input_layer = layers.Input(shape=(96, 96, 1))

        # WideResNet, conv group 1 (input)
        output = self.conv_layer(input_layer, 16, 2, l2_reg=l2_reg)

        # WideResNet, conv group 2
        output = self.residual_module(output, 16 * k, 2, l2_reg=l2_reg, dropout=conv_dropout/3)
        for i in range(n - 1):
            output = self.residual_module(output, 16 * k, l2_reg=l2_reg, dropout=conv_dropout/3)

        # WideResNet, conv group 3
        output = self.residual_module(output, 32 * k, 2, l2_reg=l2_reg, dropout=conv_dropout/2)
        for i in range(n - 1):
            output = self.residual_module(output, 32 * k, l2_reg=l2_reg, dropout=conv_dropout/2)
        
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
