import os
import tensorflow as tf
from tensorflow import keras
import math

print("Version", tf.version.VERSION)


# Example 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define the model
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# Save checkpoints during training
checkpoint_path = "training_2/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32 
n_batches = math.ceil(len(train_images) / batch_size)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5 * math.ceil(len(train_images) / batch_size)
)

# Train the model with the new callback
model = create_model()
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback]) # Pass callback to training

os.listdir(checkpoint_dir)

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


checkpoint_dir = "training_2"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("Latest checkpoint:", latest)

# Load the weights
model = create_model()
# model.load_weights(latest)
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Checkpoint callback options

# Include the epoch in the file name (uses 'str.format')
checkpoint_path = "training_2/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate the number of batches per epoch
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches) # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*n_batches
)

# Create a new model instance
model = create_model()

# Save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          batch_size=batch_size,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0
          )

os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# Create a new model instance
model = create_model()

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a '.keras' zip archive.
model.save('my_model.keras')

new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
# print('Resotred model, accuracy: {:5.2f}%'.format(100 * acc))
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


print(new_model.predict(test_images).shape)

# SavedModel Format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model.save('saved_model/my_model')

# Load
new_model = tf.keras.models.load_model('saved_model/my_model')

# Evaluate
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.fromat(100 * acc))

predictions = new_model.predict(test_images)
print(predictions.shape)

new_model = tf.kerars.model.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

# HDF5 (Hierarchical Data Format version 5)

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extenstion indicates that the model should be saved to HDF5.
model.save('my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

# Check Accuracy
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

