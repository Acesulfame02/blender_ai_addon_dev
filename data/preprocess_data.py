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
print(f"Label map: {label_map}")

# Load and preprocess data
sequences, labels = [], []
for action in config.actions:
    action_path = os.path.join(config.processed_data_path, action)
    print(f"Processing action: {action}")
    print(f"Action path: {action_path}")
    
    if not os.path.exists(action_path):
        print(f"Action path does not exist: {action_path}")
        continue

    files_processed = 0
    for item in os.listdir(action_path):
        item_path = os.path.join(action_path, item)
        
        if os.path.isdir(item_path):
            print(f"Processing directory: {item_path}")
            for sequence_file in os.listdir(item_path):
                if sequence_file.endswith('.npy'):
                    sequence_path = os.path.join(item_path, sequence_file)
                    try:
                        sequence_data = np.load(sequence_path)
                        print(f"Loaded sequence data shape from {sequence_path}: {sequence_data.shape}")
                        
                        if sequence_data.shape == (sequence_length, 18):
                            sequences.append(sequence_data)
                            labels.append(label_map[action])
                            files_processed += 1
                        else:
                            print(f"Unexpected sequence data shape in {sequence_file}: {sequence_data.shape}")
                    except Exception as e:
                        print(f"Error loading {sequence_file} from {item_path}: {e}")
        
        elif item.endswith('.npy'):
            try:
                sequence_data = np.load(item_path)
                print(f"Loaded sequence data shape from {item_path}: {sequence_data.shape}")
                
                if sequence_data.shape == (sequence_length, 18):
                    sequences.append(sequence_data)
                    labels.append(label_map[action])
                    files_processed += 1
                else:
                    print(f"Unexpected sequence data shape in {item}: {sequence_data.shape}")
            except Exception as e:
                print(f"Error loading {item} from {action_path}: {e}")
    
    print(f"Processed {files_processed} files for action {action}")

print(f"Total sequences loaded: {len(sequences)}")
print(f"Total labels: {len(labels)}")

# Convert to numpy arrays and save processed data
if len(sequences) == 0 or len(labels) == 0:
    print("No valid sequences or labels found. Exiting.")
    sys.exit(1)

X = np.array(sequences)
# Instead of:
# Y = to_categorical(labels).astype(int)

# Use:
Y = np.array(sequences)

# Print information about the data
print(f"Final shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")
print(f"Number of sequences: {X.shape[0]}")
print(f"Sequence length: {X.shape[1]}")
print(f"Number of keypoints per frame: {X.shape[2]}")
# Print information about the data
print(f"Final shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")
print(f"Number of sequences: {X.shape[0]}")
print(f"Sequence length: {X.shape[1]}")
print(f"Number of keypoints per frame: {X.shape[2]}")
# Reshape X to correct shape (this step might not be necessary now, but we'll keep it for consistency)
X = X.reshape((X.shape[0], sequence_length, -1))
# Print information about the data
print(f"Final shape of X: {X.shape}")
# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# Save preprocessed data
np.save(os.path.join(config.processed_data_path, 'x_train.npy'), x_train)
np.save(os.path.join(config.processed_data_path, 'x_test.npy'), x_test)
np.save(os.path.join(config.processed_data_path, 'y_train.npy'), y_train)
np.save(os.path.join(config.processed_data_path, 'y_test.npy'), y_test)

print("Data preprocessing complete.")