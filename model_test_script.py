import bpy
import cv2
import mediapipe as mp
import threading
import mathutils
import os

# Name of the Rigify Human Meta-Rig
RIG_OBJECT_NAME = "rig"

# Setup Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Log file path
DOWNLOADS_PATH = os.path.join(os.path.expanduser('~'), 'Downloads')
LOG_FILE = os.path.join(DOWNLOADS_PATH, 'arm_control_names.txt')

# Function to log control names and scaling factors
def log_control_names(scale_factor, control_name):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Scale factor {scale_factor}: Applied to {control_name}\n")

# Function to calculate the arm length in Blender
def calculate_arm_length(rig, hand_bones):
    hand_pos = rig.matrix_world @ rig.pose.bones[hand_bones[0]].head
    shoulder_pos = rig.matrix_world @ rig.pose.bones[hand_bones[1]].head
    return (hand_pos - shoulder_pos).length

# Function to update the rig in Blender
def update_rig(hand_landmarks, hand_type, rig):
    try:
        # Specific bones for the hands in Rigify
        hand_bones = ['hand_ik.L', 'forearm_tweak.L', 'upper_arm_tweak.L.001'] if hand_type == "Left" else ['hand_ik.R', 'forearm_tweak.R', 'upper_arm_tweak.R.001']
        
        # Function to get world space coordinates of a bone
        def get_bone_world_position(bone_name):
            bone = rig.pose.bones.get(bone_name)
            if bone:
                return rig.matrix_world @ bone.head
            return None

        # Get initial positions of hand, forearm, and upper arm bones
        initial_hand_pos = get_bone_world_position(hand_bones[0])
        initial_forearm_pos = get_bone_world_position(hand_bones[1])
        initial_shoulder_pos = get_bone_world_position(hand_bones[2])

        if initial_hand_pos and initial_forearm_pos and initial_shoulder_pos:
            # Calculate initial arm length in Blender
            initial_arm_length = (initial_hand_pos - initial_shoulder_pos).length

            # Get Mediapipe hand landmarks for wrist and elbow (using wrist as proxy for elbow)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert Mediapipe coordinates to Blender world coordinates
            wrist_pos = mathutils.Vector([
                (wrist.x - 0.5) * 2,  # X-axis in Blender (adjusted)
                (wrist.z - 0.5) * 2,  # Z-axis (depth in Blender)
                (0.5 - wrist.y) * 2   # Y-axis in Blender (vertical)
            ])

            # Calculate new hand position with a fixed arm length
            direction_vector = wrist_pos.normalized()
            new_hand_pos = initial_shoulder_pos + direction_vector * initial_arm_length

            # Update the hand IK bone position
            hand_ik = rig.pose.bones.get(hand_bones[0])
            if hand_ik:
                new_local_pos = rig.matrix_world.inverted() @ new_hand_pos
                hand_ik.location = new_local_pos - hand_ik.head
                hand_ik.keyframe_insert(data_path="location")

                # Log the applied bone control name and scale factor
                log_control_names('Custom', hand_bones[0])

        # Update the Blender scene to apply the changes
        bpy.context.view_layer.update()

    except Exception as e:
        print(f"Error updating rig: {e}")

# Function to process video frames and update the rig
def process_video():
    rig = bpy.data.objects.get(RIG_OBJECT_NAME)
    if not rig:
        print(f"Error: Could not find rig '{RIG_OBJECT_NAME}' in Blender.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Iterate through detected hands and their landmarks
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = hand_info.classification[0].label
                # Update the rig in Blender using the detected hand landmarks
                bpy.app.timers.register(lambda: update_rig(hand_landmarks, hand_type, rig), first_interval=0.01)
        
        # Display the webcam feed
        cv2.imshow('Mediapipe Hands', frame)

        # Stop the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Start video processing in a separate thread so Blender stays responsive
thread = threading.Thread(target=process_video)
thread.start()
