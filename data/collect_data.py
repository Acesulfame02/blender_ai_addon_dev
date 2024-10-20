import cv2
import numpy as np
import os
import mediapipe as mp
import sys

# Adjust the path to ensure the 'config.py' file can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

# Initialize MediaPipe holistic model (which includes body, hands, and face landmarks)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize Config to handle paths
config = Config()

# Define sequence length for splitting video data into smaller sequences
sequence_length = 30

# Function to perform MediaPipe detection on the video frame
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw essential arm landmarks on the image
def draw_arm_landmarks(image, results):
    # Draw shoulders, elbows, and wrists (pose landmarks)
    if results.pose_landmarks:
        pose_landmark_idxs = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in pose_landmark_idxs:
            x = int(results.pose_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.pose_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Function to extract keypoints for essential body parts
def extract_arm_keypoints(results):
    def convert_to_normalized_coordinates(keypoints):
        """ Convert keypoints to a format normalized for Blender. """
        normalized_keypoints = []
        for kp in keypoints:
            x = np.clip((kp[0] - 0.5) * 0.1, -1, 1)
            y = np.clip((kp[2] - 0.5) * 0.1, -1, 1)  # Swap y and z
            z = np.clip((kp[1] - 0.5) * 0.1, -1, 1)
            normalized_keypoints.append((x, z, -y))
        return np.array(normalized_keypoints).flatten()

    # Initialize arrays for pose (shoulder, elbow, wrist)
    left_arm = np.zeros(9)  # Left shoulder, left elbow, left wrist
    right_arm = np.zeros(9)  # Right shoulder, right elbow, right wrist

    # Extract pose landmarks for shoulders, elbows, wrists
    if results.pose_landmarks:
        left_arm = convert_to_normalized_coordinates([
            (results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z),  # Left shoulder
            (results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[13].z),  # Left elbow
            (results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y, results.pose_landmarks.landmark[15].z)   # Left wrist
        ])
        right_arm = convert_to_normalized_coordinates([
            (results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z),  # Right shoulder
            (results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[14].z),  # Right elbow 
            (results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y, results.pose_landmarks.landmark[16].z)   # Right wrist
        ])

    # Return concatenated keypoints array (left arm, right arm)
    return np.concatenate([left_arm, right_arm])

# Create folders for data storage
for action in config.actions:
    action_path = os.path.join(config.raw_data_path, action)
    for video_file in os.listdir(action_path):
        video_output_path = os.path.join(config.processed_data_path, action, video_file.split('.')[0])
        os.makedirs(video_output_path, exist_ok=True)

# Process videos and extract keypoints for arm movements
for action in config.actions:
    action_path = os.path.join(config.raw_data_path, action)
    for video_file in os.listdir(action_path):
        cap = cv2.VideoCapture(os.path.join(action_path, video_file))
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            sequence = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection and draw landmarks
                image, results = mediapipe_detection(frame, holistic)
                draw_arm_landmarks(image, results)
                
                # Extract keypoints and append to the sequence
                keypoints = extract_arm_keypoints(results)
                sequence.append(keypoints)
                frame_count += 1
                
                # Once sequence length is reached, save the keypoints
                if len(sequence) == sequence_length:
                    npy_path = os.path.join(config.processed_data_path, action, video_file.split('.')[0], "{}.npy".format(len(os.listdir(os.path.join(config.processed_data_path, action, video_file.split('.')[0])))))
                    np.save(npy_path, np.array(sequence))
                    sequence = []  # Reset sequence after saving
                
                # Display the processed video
                cv2.imshow('Video', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Save remaining frames as a new sequence if sequence length is reached
            if len(sequence) == sequence_length:
                npy_path = os.path.join(config.processed_data_path, action, video_file.split('.')[0], "{}.npy".format(len(os.listdir(os.path.join(config.processed_data_path, action, video_file.split('.')[0])))))
                np.save(npy_path, np.array(sequence))

        cap.release()
        cv2.destroyAllWindows()