import numpy as np
import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)

print(df.head())
print(df.dtypes)

target = df.pop('target')

numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
print(numeric_features.head())

tf.convert_to_tensor(numeric_features)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(numeric_features))

normalizer(numeric_features.iloc[:3])

def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
    print(row)
    
numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

model = get_basic_model()
model.fit(numeric_batches, epochs=15)

numeric_features_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in dict(numeric_features).items()}
target_array = target.to_numpy()[:, tf.newaxis]

numeric_dict_ds = tf.data.Dataset.from_tensor_slices((numeric_features_dict, target_array))

len(numeric_features_dict)

for row in numeric_dict_ds.take(3):
    print(row)
    
inputs = {}
for name, column in numeric_features.items():
    inputs[name] = tf.keras.Input(
        shape=(1,), name=name, dtype=tf.float32
    )
print(inputs)

xs = [value for key, value in sorted(inputs.items())]

concat = tf.keras.layers.Concatenate(axis=1)
x = concat(xs)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.concatenate([value for key, value in sorted(numeric_features_dict.items())], axis=1))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)


tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True,  show_layer_names=True)