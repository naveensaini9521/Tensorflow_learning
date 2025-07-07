import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

layers = tf.keras.layers.Dense(100)
layers = tf.keras.layers.Dense(10, input_shape=(None, 5))

# To use a layer, simply call it.
layers(tf.zeros([10, 5]))
layers.variables
layers.kernel, layers.bias

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
    
    def build(self, __input_shape):
        self.kernel = self.add_weight("kernel", 
                                      shape=[int(__input_shape[-1]),
                                             self.num_outputs])
        
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
    
layers = MyDenseLayer(10)

_ = layers(tf.zeros([10, 5])) # Calling the layer `.builds` it.
print([var.name for var in layer.trainable_variables])

# Models 
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters
        
        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        sef.bn2a = tf.keras.layers.BatchNormalization()
        
        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='size')
        self.bn2b = tf.keras.layers.BatchNormalization()
        
        self.conv2c = tf.keras.layers.Conva2D(filters2, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()
        
    