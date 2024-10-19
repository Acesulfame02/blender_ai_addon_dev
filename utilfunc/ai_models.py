import tensorflow as tf
import os

def load_ai_model(self, model_path):
    print(f"Attempting to load model from: {model_path}")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        return None

    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except OSError as e:
        print(f"OSError: Unable to load model. Check if the model file is corrupted. Error: {e}")
        self.report({'OSError'}, f"Unable to load model. Check if the model file is corrupted. Error: {e}")
    except ValueError as e:
        print(f"ValueError: Model could not be interpreted. Ensure it was saved in a compatible format. Error: {e}")
        self.report({'ValueError:'}, f"Model could not be interpreted. Ensure it was saved in a compatible format. Error: {e}")
    except Exception as e:
        print(f"General Exception: Failed to load the model due to an unexpected error. Error: {e}")
        self.report({'General Exception:'}, f"Failed to load the model due to an unexpected error. Error: {e}")
    return None
