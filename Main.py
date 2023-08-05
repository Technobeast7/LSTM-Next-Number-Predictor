# Created By Technobeast :)

# Importing Libraries
import numpy as np
import tensorflow as tf

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(3, 1)),  # LSTM layer with 64 memory units
    tf.keras.layers.Dense(1)                      # Output layer with 1 neuron
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Create some example data for the model to learn from
# Here, we're using a simple sequence of numbers
# Each sequence has three numbers, and the target is the next number in the sequence
data = np.array([[1, 2, 3],
                 [2, 3, 4],
                 [3, 4, 5],
                 [4, 5, 6],
                 [5, 6, 7],
                 [6, 7, 8]])

# The target for each sequence is the next number in the sequence
targets = np.array([4, 5, 6, 7, 8, 9])

# Reshape the data to fit the LSTM input shape
data = data.reshape(data.shape[0], data.shape[1], 1)

# Train the model
model.fit(data, targets, epochs=1000, verbose=0)

# Now, let's use the trained model to predict the next number in a new sequence
new_sequence = np.array([[7, 8, 9]])
new_sequence = new_sequence.reshape(new_sequence.shape[0], new_sequence.shape[1], 1)

# Lets Test the predictor!
predicted_number = model.predict(new_sequence)[0][0]
print("Predicted next number in the sequence:", predicted_number)
