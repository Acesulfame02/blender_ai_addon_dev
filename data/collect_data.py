import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe holistic model (which includes body, hands, and face landmarks)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define paths and parameters
BASE_PATH = os.path.join(os.getenv('APPDATA'), 'Blender Foundation', 'Blender', '4.2', 'scripts', 'addons', 'kma')
RAW_VIDEO_PATH = os.path.join(BASE_PATH, 'data', 'raw', 'videos')
DATA_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'keypoints')
sequence_length = 30

# Ensure the raw video path exists
if not os.path.exists(RAW_VIDEO_PATH):
    raise FileNotFoundError(f"RAW_VIDEO_PATH does not exist: {RAW_VIDEO_PATH}")

# Function to perform MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks (shoulders, elbows, wrists, and hands)
def draw_arm_landmarks(image, results):
    # Draw left and right hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Draw shoulders, elbows, and wrists (pose landmarks)
    if results.pose_landmarks:
        pose_landmark_idxs = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in pose_landmark_idxs:
            x = int(results.pose_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.pose_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Function to extract keypoints from shoulders to fingers
def extract_arm_keypoints(results):
    def convert_to_blender_coordinates(keypoints):
        blender_keypoints = []
        for kp in keypoints:
            x = (kp[0] - 0.5) * 0.1
            y = (kp[2] - 0.5) * 0.1  # Swap y and z
            z = (kp[1] - 0.5) * 0.1
            blender_keypoints.append((x, z, -y))
        return np.array(blender_keypoints).flatten()

    # Initialize arrays for pose (shoulder, elbow, wrist) and hand keypoints
    left_arm = np.zeros(9)  # (left shoulder, left elbow, left wrist)
    right_arm = np.zeros(9)  # (right shoulder, right elbow, right wrist)
    lh = np.zeros(63)  # Left hand keypoints
    rh = np.zeros(63)  # Right hand keypoints

    # Extract pose landmarks for shoulders, elbows, wrists
    if results.pose_landmarks:
        left_arm = convert_to_blender_coordinates(
            [(results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z),  # Left shoulder
             (results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[13].z),  # Left elbow
             (results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y, results.pose_landmarks.landmark[15].z)]  # Left wrist
        )
        right_arm = convert_to_blender_coordinates(
            [(results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z),  # Right shoulder
             (results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[14].z),  # Right elbow
             (results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y, results.pose_landmarks.landmark[16].z)]  # Right wrist
        )

    # Extract hand landmarks (left and right hands)
    if results.left_hand_landmarks:
        lh = convert_to_blender_coordinates(
            [(res.x, res.y, res.z) for res in results.left_hand_landmarks.landmark]
        )
    if results.right_hand_landmarks:
        rh = convert_to_blender_coordinates(
            [(res.x, res.y, res.z) for res in results.right_hand_landmarks.landmark]
        )

    # Return concatenated array with arm and hand keypoints
    return np.concatenate([left_arm, right_arm, lh, rh])

# Dynamically identify action directories
actions = [d for d in os.listdir(RAW_VIDEO_PATH) if os.path.isdir(os.path.join(RAW_VIDEO_PATH, d))]

# Create folders for data storage
for action in actions:
    action_path = os.path.join(RAW_VIDEO_PATH, action)
    for video_file in os.listdir(action_path):
        video_output_path = os.path.join(DATA_PATH, action, video_file.split('.')[0])
        os.makedirs(video_output_path, exist_ok=True)

# Process videos and extract keypoints for arm movements and hands
for action in actions:
    action_path = os.path.join(RAW_VIDEO_PATH, action)
    for video_file in os.listdir(action_path):
        cap = cv2.VideoCapture(os.path.join(action_path, video_file))
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            sequence = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image, results = mediapipe_detection(frame, holistic)
                draw_arm_landmarks(image, results)
                keypoints = extract_arm_keypoints(results)
                sequence.append(keypoints)
                frame_count += 1
                
                if len(sequence) == sequence_length:
                    npy_path = os.path.join(DATA_PATH, action, video_file.split('.')[0], "{}.npy".format(len(os.listdir(os.path.join(DATA_PATH, action, video_file.split('.')[0])))))
                    np.save(npy_path, np.array(sequence))
                    sequence = []
                
                cv2.imshow('Video', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Save any remaining frames as a new sequence if they reach the desired sequence_length
            if len(sequence) == sequence_length:
                npy_path = os.path.join(DATA_PATH, action, video_file.split('.')[0], "{}.npy".format(len(os.listdir(os.path.join(DATA_PATH, action, video_file.split('.')[0])))))
                np.save(npy_path, np.array(sequence))

        cap.release()
        cv2.destroyAllWindows()
