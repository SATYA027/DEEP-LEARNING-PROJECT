# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the IMDB dataset
# Keep only the top 10,000 most frequently occurring words
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Pad all sequences to the same length (200 words)
# This ensures that all reviews are the same shape for training
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Build the sequential neural network model
model = keras.Sequential([
    # Embedding layer: maps each word index to a 32-dimensional dense vector
    # input_dim = 10000 because we use the top 10,000 words
    # output_dim = 32 means each word will be represented by a 32-dim vector
    # input_length = 200 to match the padded input length
    layers.Embedding(input_dim=10000, output_dim=32, input_length=200),

    # Global average pooling layer: averages the word vectors across the sequence
    # This reduces the sequence of vectors to a single vector (of length 32)
    layers.GlobalAveragePooling1D(),

    # Dense hidden layer with 16 units and ReLU activation
    layers.Dense(16, activation='relu'),

    # Output layer with 1 unit and sigmoid activation for binary classification
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with:
# - Adam optimizer (good for most use cases)
# - Binary crossentropy loss (for binary classification)
# - Accuracy as the evaluation metric
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Split the training data into training and validation sets
# Use first 10,000 samples for validation
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model
# - Train on 15,000 samples
# - Validate on 10,000 samples
# - Run for 5 epochs
# - Use a batch size of 512
history = model.fit(partial_x_train, partial_y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model on test data and print accuracy
results = model.evaluate(x_test, y_test)
print("Test Accuracy:", results[1])
