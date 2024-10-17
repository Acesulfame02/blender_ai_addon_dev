import os
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense

# Define paths
BASE_PATH = os.path.join(os.getenv('APPDATA'), 'Blender Foundation', 'Blender', '4.2', 'scripts', 'addons', 'kma')
DATA_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'keypoints')
MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models', 'saved_models', 'action_model.keras')

# Load preprocessed data
x_train = np.load(os.path.join(DATA_PATH, 'x_train.npy'))
x_test = np.load(os.path.join(DATA_PATH, 'x_test.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

# Ensure data has the correct shape
print(f"x_train shape: {x_train.shape}")  # Should be (num_samples, sequence_length, num_features)
print(f"x_test shape: {x_test.shape}")  # Should be (num_samples, sequence_length, num_features)

# Build simplified LSTM model
model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(33 * 3, activation='linear'))  # 33 keypoints with x, y, and z coordinates each

# Compile the model (adjust loss and metrics based on your task)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Fit the model
model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_split=0.3,
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Model evaluation:\nLoss: {loss}\nAccuracy: {accuracy}")

# Save the model in the .keras format
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")