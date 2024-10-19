import bpy
import mathutils
import os
import mediapipe as mp

# Setup Mediapipe Hands
mp_hands = mp.solutions.hands

# Log file path
DOWNLOADS_PATH = os.path.join(os.path.expanduser('~'), 'Downloads')
LOG_FILE = os.path.join(DOWNLOADS_PATH, 'arm_control_names.txt')

def log_control_names(scale_factor, control_name):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Scale factor {scale_factor}: Applied to {control_name}\n")

def calculate_arm_length(rig, hand_bones):
    hand_pos = rig.matrix_world @ rig.pose.bones[hand_bones[0]].head
    shoulder_pos = rig.matrix_world @ rig.pose.bones[hand_bones[1]].head
    return (hand_pos - shoulder_pos).length

def get_bone_world_position(rig, bone_name):
    bone = rig.pose.bones.get(bone_name)
    if bone:
        return rig.matrix_world @ bone.head
    return None

def update_rig(hand_landmarks, hand_type, rig):
    try:
        # Specific bones for the hands in Rigify
        hand_bones = ['hand_ik.L', 'forearm_tweak.L', 'upper_arm_tweak.L.001'] if hand_type == "Left" else ['hand_ik.R', 'forearm_tweak.R', 'upper_arm_tweak.R.001']
        
        # Get initial positions of hand, forearm, and upper arm bones
        initial_hand_pos = get_bone_world_position(rig, hand_bones[0])
        initial_forearm_pos = get_bone_world_position(rig, hand_bones[1])
        initial_shoulder_pos = get_bone_world_position(rig, hand_bones[2])

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

def apply_keypoints_to_rig(obj, keypoints, frame):
        keypoint_bone_map = {
        "DEF-thigh.L": keypoints[0:3],
        "DEF-shin.L": keypoints[3:6],
        "DEF-foot.L": keypoints[6:9],
        "DEF-thigh.R": keypoints[9:12],
        "DEF-shin.R": keypoints[12:15],
        "DEF-foot.R": keypoints[15:18],
        "DEF-upper_arm.L": keypoints[18:21],
        "DEF-forearm.L": keypoints[21:24],
        "DEF-hand.L": keypoints[24:27],
        "DEF-upper_arm.R": keypoints[27:30],
        "DEF-forearm.R": keypoints[30:33],
        "DEF-hand.R": keypoints[33:36],
    }
    
        for bone_name, keypoint in keypoint_bone_map.items():
            if bone_name in obj.pose.bones:
                bone = obj.pose.bones[bone_name]
                if len(keypoint) == 3:
                    bone.location = keypoint
                    bone.keyframe_insert(data_path="location", frame=frame)
                else:
                    print(f"Warning: Keypoint for {bone_name} does not contain exactly 3 elements: {keypoint}")

        print("Applied keypoints to rig")