import os
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
import sys

# Adjust the path to ensure the 'config.py' file can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

# Initialize Config
config = Config()

# Load preprocessed data using paths from config
x_train = np.load(os.path.join(config.processed_data_path, 'x_train.npy'))
x_test = np.load(os.path.join(config.processed_data_path, 'x_test.npy'))
y_train = np.load(os.path.join(config.processed_data_path, 'y_train.npy'))
y_test = np.load(os.path.join(config.processed_data_path, 'y_test.npy'))

# Ensure data has the correct shape
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Build a more complex LSTM model with two LSTM layers
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(33 * 3, activation='linear'))  # 33 keypoints with x, y, and z coordinates each

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(x_train, y_train, epochs=50, validation_split=0.3)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Model evaluation:\nLoss: {loss}\nAccuracy: {accuracy}")

# Save the model
model.save(config.model_path)
print(f"Model saved to {config.model_path}")
