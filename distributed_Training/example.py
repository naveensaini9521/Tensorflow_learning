import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor

def configure_virtuel_cpus(ncpu):
    phy_devices = tf.config.list_physical_devices('CPU')
    tf.config.set_logical_device_configuration(
        phy_devices[0],
        [tf.config.LogicalDeviceConfiguration()] * ncpu
    )
    
configure_virtuel_cpus(8)
tf.config.list_logical_devices('CPU')

devices = [f'CPU:{i}' for i in range(8)]

#tf.keras.backend.experimental.enable_tf_random_generator()
tf.keras.utils.set_random_seed(1337)
tf.keras.utils.enable_tf_random_generator()

# Creating a Data Parallel Mesh
mesh = dtensor.create_mesh([("batch", 8)])

example_weight_layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED])
example_weight_layout = dtensor.Layout.replicated(mesh, rank=2)

example_data_layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh) 
example_data_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)

unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.Dense(128,
                   activation='relu',
                   name='d1',
                   kernel_layout=unsharded_layout_2d,
                   bias_layout=unsharded_layout_1d),
    tf.keras.layers.Dense(10,
                          name='d2',
                          kernel_layout=unsharded_layout_2d,
                          bias_layout=unsharded_layout_1d)
])

for weight in model.weights:
  print(f'Weight name: {weight.name} with layout: {weight.layout}')
  break
