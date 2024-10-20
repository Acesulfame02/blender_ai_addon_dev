import os
import numpy as np
from keras import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
from keras.api.regularizers import l2
import sys
import matplotlib.pyplot as plt

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

# Print shapes to verify data
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Ensure shapes are correct
assert x_train.shape[1:] == (30, 18), f"Expected x_train shape to be (n, 30, 18), but got {x_train.shape}"
assert x_test.shape[1:] == (30, 18), f"Expected x_test shape to be (n, 30, 18), but got {x_test.shape}"
assert y_train.shape[1:] == (30, 18), f"Expected y_train shape to be (n, 30, 18), but got {y_train.shape}"
assert y_test.shape[1:] == (30, 18), f"Expected y_test shape to be (n, 30, 18), but got {y_test.shape}"

# Build the LSTM model for sequence-to-sequence prediction
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 18), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    LSTM(64, return_sequences=True, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(18, activation='linear')  # Output layer for regression (6 keypoints * 3 coordinates)
])

# Compile the model for regression
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# Save the model
model.save(config.model_path)
print(f"Model saved to {config.model_path}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(122)
plt.plot(history.history['mae'], label='Train MAE') # Update metric
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config.processed_data_path, 'training_history.png'))
plt.close()

model.summary()