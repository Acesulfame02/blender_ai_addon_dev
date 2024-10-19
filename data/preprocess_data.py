import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.utils import to_categorical
import sys

# Adjust the path to ensure the 'config.py' file can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

# Initialize Config
config = Config()

# Parameters
sequence_length = 30
test_size = 0.05

# Label mapping based on actions in config
label_map = {label: num for num, label in enumerate(config.actions)}

# Load and preprocess data
sequences, labels = [], []
for action in config.actions:
    action_path = os.path.join(config.processed_data_path, action)
    if not os.path.exists(action_path):
        print(f"Action path does not exist: {action_path}")
        continue

    for sequence in os.listdir(action_path):
        sequence_path = os.path.join(action_path, sequence)
        if not os.path.isdir(sequence_path):
            continue

        window = []
        for frame_file in os.listdir(sequence_path):
            frame_file_path = os.path.join(sequence_path, frame_file)
            try:
                res = np.load(frame_file_path)
                window.append(res)
            except Exception as e:
                print(f"Error loading {frame_file_path}: {e}")
                continue

        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        elif len(window) > sequence_length:
            sequences.append(window[:sequence_length])
            labels.append(label_map[action])
        else:
            while len(window) < sequence_length:
                window.append(np.zeros_like(window[0]))
            sequences.append(window)
            labels.append(label_map[action])

# Convert to numpy arrays and save processed data
X = np.array(sequences)
if len(labels) == 0:
    print("No labels found. Exiting.")
    sys.exit(1)
Y = to_categorical(labels).astype(int)

# Reshape X to correct shape
X = X.reshape((X.shape[0], sequence_length, -1))

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# Save preprocessed data
np.save(os.path.join(config.processed_data_path, 'x_train.npy'), x_train)
np.save(os.path.join(config.processed_data_path, 'x_test.npy'), x_test)
np.save(os.path.join(config.processed_data_path, 'y_train.npy'), y_train)
np.save(os.path.join(config.processed_data_path, 'y_test.npy'), y_test)

print("Data preprocessing complete.")
