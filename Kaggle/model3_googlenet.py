import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import keras_tuner as kt

# Epochs for training : 200+
# Good validation patience : 20
# Good fine-tuning lr : 1E-5

class Model(kt.HyperModel):
    def inception_layer(self, input, f1, f3_in, f3_out, f5_in, f5_out, pool_out, dropout=0):
        conv1 = layers.Conv2D(
            f1, kernel_size=(1,1), padding='same', 
            activation='relu', kernel_initializer='he_uniform')(input)
        
        conv3 = layers.Conv2D(
            f3_in, (1,1), padding='same', 
            activation='relu', kernel_initializer='he_uniform')(input)
        conv3 = layers.Conv2D(
            f3_out, kernel_size=(3,3), padding='same', 
            activation='relu', kernel_initializer='he_uniform')(conv3)
        
        conv5 = layers.Conv2D(
            f5_in, (1,1), padding='same', 
            activation='relu', kernel_initializer='he_uniform')(input)
        conv5 = layers.Conv2D(
            f5_out, kernel_size=(5,5), padding='same', 
            activation='relu', kernel_initializer='he_uniform')(conv5)
        
        pool = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(input)
        pool = layers.Conv2D(
            pool_out, kernel_size=(1,1), padding='same', 
            activation='relu', kernel_initializer='he_uniform')(pool)
        
        layer_out = layers.concatenate([conv1, conv3, conv5, pool])

        if dropout == 0:
            return layer_out
        else:
            return layers.Dropout(dropout)(layer_out)
    
    def maxpool_layer(self, input):
        return layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(input)

    def conv_layer(self, input, filter, kernel, stride=1):
        return layers.Conv2D(
            filters=filter, kernel_size=(kernel,kernel), 
            strides=(stride, stride), padding='same',
            kernel_initializer='he_uniform')(input)

    def dense_layer(self, input, size, l2_reg, dropout):
        dense = layers.Dense(
            size, activation='relu', kernel_initializer='he_uniform',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            bias_regularizer=keras.regularizers.l2(l2_reg))(input)
        return layers.Dropout(dropout)(dense)

    def build(self, hyperparameters):
        # Fixed hyperparameters
        l2_reg = 0.0025
        conv_dropout = 0.05
        dense_dropout = 0.75

        input_layer = layers.Input(shape=(96, 96, 1))

        # Input convolution
        # Local respnorm and Spatial Dropout added
        output = self.conv_layer(input_layer, 64, 7, 2)
        output = layers.BatchNormalization()(output)

        # Second input convolution
        output = self.conv_layer(output, 128, 3, 2)
        output = layers.BatchNormalization()(output)

        # Inception layers, level 1
        output = self.inception_layer(output, 64, 96, 128, 16, 32, 32, dropout=conv_dropout) #3a
        output = layers.BatchNormalization()(output)
        output = self.inception_layer(output, 128, 128, 192, 32, 96, 64, dropout=conv_dropout) #3b
        output = self.maxpool_layer(output)
        output = layers.BatchNormalization()(output)

        # Inception layers, level 2
        output = self.inception_layer(output, 192, 96, 208, 16, 48, 64, dropout=conv_dropout) #4a

        # Auxilliary prediction layer
        output = layers.AveragePooling2D(pool_size=(5,5), strides=(2,2))(output)
        output = layers.BatchNormalization()(output)
        output = self.conv_layer(output, 128, 1)
        output = layers.Flatten()(output)
        output = self.dense_layer(output, 256, l2_reg=l2_reg, dropout=dense_dropout)

        # Final output
        output = layers.Dense(
            11, kernel_regularizer=keras.regularizers.l2(l2_reg),
            bias_regularizer=keras.regularizers.l2(l2_reg))(output)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        # Create model
        model.compile(
            optimizer=keras.optimizers.Nadam(0.0001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model
