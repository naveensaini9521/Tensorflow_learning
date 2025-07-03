import tensorflow as tf

print("Tensorflow version:", tf.__version__)

# Load Dataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


# Build Machine Learning Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(X_train[:1]).numpy()
print(predictions)

# tf.nn.softmax(predictions).numpy()
print("Probabilities:", tf.nn.softmax(predictions).numpy())


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()
print("Initial loss:", loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam', 
              loss=loss_fn,
              metrics=['accuracy'])

# Train and Evaluate model
model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
print("Predicted probabilities for first 5 test images:")
print(probability_model(X_test[:5]))