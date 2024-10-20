import os
import numpy as np
from keras.models import load_model
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.optimizers import Adam
import matplotlib.pyplot as plt
import sys

# Adjust the path to ensure the 'config.py' file can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

# Initialize Config
config = Config()

# Load preprocessed data
x_train = np.load(os.path.join(config.processed_data_path, 'x_train.npy'))
x_test = np.load(os.path.join(config.processed_data_path, 'x_test.npy'))
y_train = np.load(os.path.join(config.processed_data_path, 'y_train.npy'))
y_test = np.load(os.path.join(config.processed_data_path, 'y_test.npy'))

# Load the existing model
model = load_model(config.model_path)

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Continue training the model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=200,  # Increase the number of epochs
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    initial_epoch=model.history.epoch[-1] if hasattr(model, 'history') else 0
)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# Save the retrained model
model.save(config.model_path)
print(f"Retrained model saved to {config.model_path}")

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
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config.processed_data_path, 'retraining_history.png'))
plt.close()

# Run more simulations to test the model's improved predictive skills
def run_simulation(model, input_sequence, num_steps=100):
    predictions = []
    current_sequence = input_sequence.copy()
    
    for _ in range(num_steps):
        next_step = model.predict(current_sequence[np.newaxis, :, :])[0, -1, :]
        predictions.append(next_step)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_step
    
    return np.array(predictions)

# Run a longer simulation
initial_sequence = x_test[0]  # Use the first test sequence as the initial sequence
predictions = run_simulation(model, initial_sequence, num_steps=500)

# Plot the simulation results
plt.figure(figsize=(12, 4))
plt.plot(predictions[:, 0], label='Predicted L Shoulder')
plt.plot(predictions[:, 1], label='Predicted L Elbow')
plt.plot(predictions[:, 2], label='Predicted L Wrist')
plt.plot(predictions[:, 3], label='Predicted R Shoulder')
plt.plot(predictions[:, 4], label='Predicted R Elbow')
plt.plot(predictions[:, 5], label='Predicted R Wrist')
plt.title('Simulation Results')
plt.xlabel('Time Step')
plt.ylabel('Keypoint Values')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config.processed_data_path, 'simulation_results.png'))
plt.close()