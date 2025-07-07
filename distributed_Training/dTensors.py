import tempfile
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.experimental import dtensor

print('TensorFlow version:', tf.__version__)

def configure_virtual_cpus(ncpu):
    phy_devices = tf.config.list_physical_devices('CPU')
    tf.config.set_logical_device_configuration(phy_devices[0], [
        tf.config.LogicalDeviceConfiguration(),
    ] * ncpu)
    
configure_virtual_cpus(8)
DEVICES = [f'CPU:{i}' for i in range(8)]
tf.config.list_logical_devices('CPU')

train_data = tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64)
print(train_data)

text_vectorization = tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200, output_sequence_length=None)
text_vectorization.adapt(data=train_data.map(lambda x: x['text']))

def vectorize(features):
    return text_vectorization(features['text']), features['label']

train_data_vec = train_data.map(vectorize)
print(train_data_vec)

# Neural Network with DTensor
class Dense(tf.module):
    def __init__(self, input_size, output_size,
                 init_seed, weight_layout, activation=None):
        super().__init__()
        
        random_normal_initializer = tf.function(tf.random.stateless_normal)
        
        self.weight = dtensor.DVariable(
            dtensor.call_with_layout(
                random_normal_initializer, weight_layout,
                shape=[input_size, output_size],
                seed=init_seed
            )
        )
        if activation is None:
            activation = lambda x:x
        self.activation = activation
        
        # Bias is sharded the same way as the last axis of weight.
        bias_layout = weight_layout.delete([0])
        
        self.bias = dtensor.DVariable(
            dtensor.call_with_layout(tf.zero, bias_layout, [output_size])
        )
        
        self.bias = dtensor.DVariable(
            dtensor.call_with_layout(tf.zero, bias_layout, [output_size])
        )
    
    def __call__(self, x):
        y = tf.matmul(x, self.weight) + self.bias
        y = self.activation(y)
        
        return y
    
