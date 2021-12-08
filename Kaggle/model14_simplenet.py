import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 150+ (250-300 minutes)
# Good validation patience : 15-20
# Good fine-tuning lr : ?

model_number = 'model10'

class Model(kt.HyperModel):
    def conv_layer(self, input, filters, stride=1, l2_reg=0, dropout=0):
        bn = layers.BatchNormalization()(input)
        relu = layers.ReLU()(bn)
        conv = layers.Conv2D(
            filters, kernel_size=(3,3), strides=(stride,stride), padding='same', 
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_normal')(relu)
        if dropout != 0:
            conv = layers.SpatialDropout2D(dropout)(conv)
        return conv

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
            learning_rate = hyperparameters.Float('lr', 1E-5, 1E-2, sampling='log')
            #l2_reg = hyperparameters.Float('l2_reg', 0.0001, 0.01, sampling='log')
            #conv_dropout = hyperparameters.Float('conv_dropout', 0.4, 0.6, step=0.2)
        else:
            learning_rate = 0.001

        # Fixed hyperparameters
        l2_reg = 1E-6
        conv_dropout = 0
        dense_dropout = conv_dropout
        
        input_layer = layers.Input(shape=(96, 96, 1))

        # Hidden layers
        output = self.conv_layer(input_layer, 64, 2, l2_reg=l2_reg)
        output = self.conv_layer(input_layer, 128, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, 2, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, 2, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, 2, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(input_layer, 128, 2, l2_reg=l2_reg, dropout=conv_dropout)

        
        # Final average pool and prediction
        output = layers.GlobalAveragePooling2D()(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)
        output = self.prediction_layer(output, 11, l2_reg, dense_dropout)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        # Create model
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
