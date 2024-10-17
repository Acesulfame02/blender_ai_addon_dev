import cv2
import numpy as np
import tensorflow as tf
import os

# Path to the model
model_path = 'models/saved_models/action_model.keras'

# Load AI model
def load_ai_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def extract_efficient_keypoints(prediction):
    # Assuming prediction is shaped (99,) for 33 keypoints with x, y, z coordinates
    keypoints = prediction.reshape(-1, 3)
    
    # Define indices of key joints (adjust based on your model's output format)
    key_indices = [
        0, 1,  # Left and right shoulder
        2, 3,  # Left and right elbow
        4, 5,  # Left and right wrist
        6, 7   # Optional: thumb tips
    ]
    
    efficient_keypoints = keypoints[key_indices]
    return efficient_keypoints.flatten()

def visualize_keypoints(keypoints):
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Reshape keypoints to (8, 3) for x, y, z coordinates of 8 key points
    keypoints = keypoints.reshape(-1, 3)
    
    # Use only x and y coordinates for 2D visualization
    keypoints_2d = keypoints[:, :2]
    
    # Normalize keypoints to range [0, 1]
    keypoints_2d = (keypoints_2d - keypoints_2d.min()) / (keypoints_2d.max() - keypoints_2d.min())
    
    # Scale keypoints to fit the display
    scaled_keypoints = (keypoints_2d * 400 + 50).astype(int)
    
    # Define pairs of keypoints to connect
    skeleton_pairs = [
        (0, 1),  # Shoulder to shoulder
        (0, 2),  # Left shoulder to left elbow
        (2, 4),  # Left elbow to left wrist
        (1, 3),  # Right shoulder to right elbow
        (3, 5),  # Right elbow to right wrist
    ]
    
    # Define keypoint labels
    keypoint_labels = [
        "L Shoulder", "R Shoulder",
        "L Elbow", "R Elbow",
        "L Wrist", "R Wrist",
        "L Thumb", "R Thumb"
    ]
    
    # Draw bones
    for pair in skeleton_pairs:
        start_point = tuple(scaled_keypoints[pair[0]])
        end_point = tuple(scaled_keypoints[pair[1]])
        cv2.line(image, start_point, end_point, (255, 0, 0), 3)
    
    # Draw keypoints and labels
    for i, point in enumerate(scaled_keypoints):
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
        cv2.putText(image, keypoint_labels[i], (point[0]+10, point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image

def generate_sequence(model, num_frames=120):
    # Generate a sequence of random inputs
    input_sequences = np.random.rand(num_frames, 30, 4320)
    
    # Predict for each input
    predictions = model.predict(input_sequences)
    
    return predictions

def interpolate_predictions(predictions, factor=2):
    # Interpolate between predictions to create smoother transitions
    interpolated = []
    for i in range(len(predictions) - 1 ):
        start = predictions[i]
        end = predictions[i + 1]
        for j in range(factor):
            t = j / factor
            interpolated.append(start * (1 - t) + end * t)
    interpolated.append(predictions[-1])  # Add the last prediction
    return np.array(interpolated)

def main():
    model = load_ai_model(model_path)
    if model is None:
        return

    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    try:
        # Generate a sequence of predictions
        predictions = generate_sequence(model)
        # Interpolate between predictions for smoother transitions
        interpolated_predictions = interpolate_predictions(predictions, factor=4)  # Increased interpolation factor

        frame_index = 0
        while True:
            prediction = interpolated_predictions[frame_index]
            efficient_prediction = extract_efficient_keypoints(prediction)
            image = visualize_keypoints(efficient_prediction)
            cv2.imshow("AI Model Keypoints Visualization", image)
            
            # Increase wait time to 100ms (0.1 seconds)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

            # Move to next frame, looping back to start if necessary
            frame_index = (frame_index + 1) % len(interpolated_predictions)

            # Add a small delay between frames (50ms or 0.05 seconds)
            cv2.waitKey(50)

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()