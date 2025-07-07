import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub

mirrored_strategy = tf.distribute.MirroredStrategy()

# Load MNIST
(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

def get_data():
    datasets = tfds.load(name='mnist', as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']
    
    BUFFER_SIZE = 10000
    
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
    
    def scale(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, -1) 
        return image, label
    
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
    
    return train_dataset, eval_dataset


def get_model():
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model
# Train and save with MirroredStrategy    
model = get_model()
train_dataset, eval_dataset = get_data()
model.fit(train_dataset, epochs=2)
    
keras_model_path = '/tmp/keras_save.keras'
model.save(keras_model_path)
 
# Reload and continue training   
restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs=2)

# The tf.saved_model API
model = get_model()
saved_model_path = '/tmp/tf_save'
tf.saved_model.save(model, saved_model_path)

# Load SavedModel and prepare inference function
DEFAULT_FUNCTION_KEY = 'serving_default'
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

# Prepare dataset for inference
predict_dataset = eval_dataset.map(lambda image, label: image)
for batch in predict_dataset.take(1):
        input_key = list(inference_func.structured_inputs.keys())[0]
        print("Inference Result", inference_func({input_key: batch}))

# Distributed inference using another strategy    
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
    
    dist_predict_dataset = another_strategy.experimental_distribute_dataset(
        predict_dataset
    )
    
    # Calling the function in a distributed manner
    for batch in dist_predict_dataset:
        input_key = list(inference_func.structured_inputs.keys())[0]
        result = another_strategy.run(lambda x: inference_func({input_key: x}), args=(batch,))
        print("Distributed Inference Result:", result)
        break
    
def build_model(loaded_model):
    x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
    keras_layer = hub.KerasLayer(loaded_model, trainable=True)(x)
    model = tf.keras.Model(x, keras_layer)
    return model

another_strategy = tf.distribute.MirroredStrategy()
try:
    with another_strategy.scope():
        loaded = tf.saved_model.load(saved_model_path)
        model = build_model(loaded)

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        model.fit(train_dataset, epochs=2)

except Exception as e:
    print("Unable to load model with KerasLayer, Error:", e)