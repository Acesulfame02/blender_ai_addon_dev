import os
import numpy as np
from keras.api.models import load_model

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
print(f"x_test shape: {x_test.shape}")    # Should be (num_samples, sequence_length, num_features)

# Load the model
model = load_model(MODEL_SAVE_PATH)
print(f"Model loaded from {MODEL_SAVE_PATH}")

# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=1)
print(f"Training data evaluation:\nLoss: {train_loss}\nAccuracy: {train_accuracy}")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test data evaluation:\nLoss: {test_loss}\nAccuracy: {test_accuracy}")

# Check for overfitting
if train_accuracy > test_accuracy:
    print("Warning: The model is overfitting. The accuracy on the training data is higher than on the test data.")
else:
    print("The model does not appear to be overfitting.")

# Check for underfitting
if train_accuracy < 0.7:  # Arbitrary threshold for underfitting
    print("Warning: The model may be underfitting. The accuracy on the training data is low.")
else:
    print("The model's training accuracy is within an acceptable range.")

# Provide insights based on evaluation
if train_loss < 0.1 and train_accuracy > 0.9:
    print("The model is performing very well on the training data.")
elif train_loss > 0.5 or train_accuracy < 0.7:
    print("The model is struggling to learn from the training data. Consider adjusting the model architecture or improving the data quality.")

if test_loss < 0.1 and test_accuracy > 0.9:
    print("The model is performing very well on the test data.")
elif test_loss > 0.5 or test_accuracy < 0.7:
    print("The model is struggling on the test data. This could indicate overfitting or that the test data is challenging.")

print("Evaluation complete.")
