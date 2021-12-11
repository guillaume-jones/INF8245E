import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 
# Good validation patience : 
# Good fine-tuning lr : 

model_number = 'model14'

class Model(kt.HyperModel):
    def conv_layer(self, input, filters, stride=1, kernel=3, l2_reg=0, dropout=0):
        bn = layers.BatchNormalization()(input)
        relu = layers.ReLU()(bn)
        conv = layers.Conv2D(
            filters, kernel_size=(kernel,kernel), strides=(stride,stride), padding='same', 
            kernel_regularizer=reg.l2(l2_reg), bias_regularizer=reg.l2(l2_reg),
            kernel_initializer='he_normal')(relu)
        if dropout != 0:
            conv = layers.SpatialDropout2D(dropout)(conv)
        return conv

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
            learning_rate = hyperparameters.Float('lr', 1E-4, 1E-2, sampling='log')
            l2_reg = hyperparameters.Float('l2_reg', 1E-6, 1E-3, sampling='log')
            dense_dropout = hyperparameters.Float('dense_dropout', 0, 0.5)
        else:
            learning_rate = 0.0005
            l2_reg = 5E-4
            dense_dropout = 0.1

        # Fixed hyperparameters
        conv_dropout = 0.05
        
        input_layer = layers.Input(shape=(96, 96, 1))

        # Hidden layers
        output = self.conv_layer(input_layer, 64, 2, l2_reg=l2_reg)
        output = self.conv_layer(output, 144, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, 2, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, 2, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, 2, l2_reg=l2_reg, dropout=conv_dropout)
        output = self.conv_layer(output, 144, 2, l2_reg=l2_reg, dropout=conv_dropout)

        
        # Final average pool and prediction
        output = layers.GlobalAveragePooling2D()(output)
        output = self.dense_layer(output, 144, l2_reg=l2_reg, dropout=dense_dropout)
        output = self.dense_layer(
            output, 11, l2_reg=l2_reg, dropout=dense_dropout, activation='linear')

        model = keras.models.Model(inputs=input_layer, outputs=output)

        # Create model
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
