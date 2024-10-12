import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define paths and parameters
BASE_PATH = os.path.join(os.getenv('APPDATA'), 'Blender Foundation', 'Blender', '4.2', 'scripts', 'addons', 'kma')
DATA_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'keypoints')
actions = ['fighting']
sequence_length = 30
test_size = 0.05  # Adjusted test size

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Load and preprocess data
sequences, labels = [], []
print(f"Checking actions in: {DATA_PATH}")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Action path does not exist: {action_path}")
        continue

    print(f"Processing action: {action}")
    for sequence in os.listdir(action_path):
        sequence_path = os.path.join(action_path, sequence)
        if not os.path.isdir(sequence_path):
            print(f"Skipping non-directory: {sequence_path}")
            continue

        window = []
        print(f"Processing sequence: {sequence_path}")
        for frame_file in os.listdir(sequence_path):
            frame_file_path = os.path.join(sequence_path, frame_file)
            try:
                res = np.load(frame_file_path)
                window.append(res)
            except Exception as e:
                print(f"Failed to load file: {frame_file_path}, error: {e}")
                continue
        
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        elif len(window) > sequence_length:
            # Truncate sequences longer than 30 frames
            sequences.append(window[:sequence_length])
            labels.append(label_map[action])
        elif len(window) < sequence_length:
            # Pad sequences shorter than 30 frames
            while len(window) < sequence_length:
                window.append(np.zeros_like(window[0]))
            sequences.append(window)
            labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
if len(labels) == 0:
    print("No labels found. Exiting.")
    sys.exit(1)
Y = to_categorical(labels).astype(int)

# Reshape X to the correct shape (num_samples, sequence_length, num_features)
X = X.reshape((X.shape[0], sequence_length, -1))

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# Save preprocessed data
np.save(os.path.join(DATA_PATH, 'x_train.npy'), x_train)
np.save(os.path.join(DATA_PATH, 'x_test.npy'), x_test)
np.save(os.path.join(DATA_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(DATA_PATH, 'y_test.npy'), y_test)

print("Data preprocessing complete.")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")