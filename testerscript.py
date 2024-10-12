import bpy
import cv2
import mediapipe as mp
import threading

# Name of the Rigify Human Meta-Rig
RIG_OBJECT_NAME = "rig"

# Setup Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to update the rig in Blender
def update_rig(landmarks):
    try:
        rig = bpy.data.objects[RIG_OBJECT_NAME]

        # Specific bones for the hands in Rigify
        left_hand_bones = ['hand_fk.L', 'hand_ik.L', 'hand_tweak.L']
        right_hand_bones = ['hand_fk.R', 'hand_ik.R', 'hand_tweak.R']

        # Clamping function to prevent excessive movement
        def clamp(value, min_value, max_value):
            return max(min_value, min(value, max_value))

        # Function to update bone location based on Mediapipe landmarks
        def update_bone_location(bone, landmark):
            if bone:
                # Adjust axis mapping: Swap y and z, and invert as needed
                x = clamp((landmark[0] - 0.5) * 0.1, -0.1, 0.1)
                y = clamp((landmark[2] - 0.5) * 0.1, -0.1, 0.1)  # Swap y and z
                z = clamp((landmark[1] - 0.5) * 0.1, -0.1, 0.1)

                # Set the bone location to the adjusted values
                bone.location = (x, z, -y)
                bone.keyframe_insert(data_path="location")

        # Update left hand bones using Mediapipe landmark 15
        for bone_name in left_hand_bones:
            bone = rig.pose.bones.get(bone_name)
            if bone:
                update_bone_location(bone, landmarks[15])

        # Update right hand bones using Mediapipe landmark 16
        for bone_name in right_hand_bones:
            bone = rig.pose.bones.get(bone_name)
            if bone:
                update_bone_location(bone, landmarks[16])

        # Update the scene
        bpy.context.view_layer.update()

    except Exception as e:
        print(f"Error updating rig: {e}")

# Function to draw landmarks (shoulders, elbows, wrists, and hands)
def draw_arm_landmarks(image, results):
    # Draw left and right hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_pose.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_pose.HAND_CONNECTIONS)

    # Draw shoulders, elbows, and wrists (pose landmarks)
    if results.pose_landmarks:
        pose_landmark_idxs = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in pose_landmark_idxs:
            x = int(results.pose_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.pose_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Function to process video frames and update the rig
def process_video():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = [(lmk.x, lmk.y, lmk.z) for lmk in results.pose_landmarks.landmark]
            # Update the rig in the next available frame
            bpy.app.timers.register(lambda: update_rig(landmarks), first_interval=0.01)

        # Draw the arm landmarks on the video
        draw_arm_landmarks(frame, results)
        
        cv2.imshow('Mediapipe Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start video processing in a separate thread
thread = threading.Thread(target=process_video)
thread.start()
