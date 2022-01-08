import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import kerastuner as kt
from tensorflow.python.keras.backend import softmax
import tensorflow_addons as tfa

def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class Model(kt.HyperModel):

    def build(self, hyperparameters):
        # Tunable hyperparameters
        if hyperparameters is not None: 
            projection_dim = hyperparameters.Choice('projection_dim',[ 64, 96, 128])
        else:
            projection_dim = 64

        # Fixed hyperparameters
        learning_rate = 0.0001
        weight_decay = 0.0001
        batch_size = 256
        image_size = 96  # We'll resize input images to this size
        patch_size = 16  # Size of the patches to be extract from the input images
        num_patches = (image_size // patch_size) ** 2
        num_heads = 12
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        transformer_layers = 6
        mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
        
        input_layer = layers.Input(shape=(96, 96, 1))
        
        #Create Patches
        # Create patches.
        patches = Patches(patch_size)(input_layer)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        
         # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.5)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])
        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(11, activation = softmax)(features)
        
        model = keras.Model(inputs=input_layer, outputs=logits)
        #Create model
        model.compile(
                optimizer=keras.optimizers.Nadam(learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        return model
