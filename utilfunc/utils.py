import bpy
import mathutils
import mediapipe as mp
import os

# Log file path
DOWNLOADS_PATH = os.path.join(os.path.expanduser('~'), 'Downloads')
LOG_FILE = os.path.join(DOWNLOADS_PATH, 'arm_control_names.txt')

def log_control_names(scale_factor, control_name):
    """ Log control names and scale factors """
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Scale factor {scale_factor}: Applied to {control_name}\n")

def calculate_arm_length(rig, hand_bones):
    """ Calculate arm length based on rig bone positions """
    hand_pos = rig.matrix_world @ rig.pose.bones[hand_bones[0]].head
    shoulder_pos = rig.matrix_world @ rig.pose.bones[hand_bones[1]].head
    return (hand_pos - shoulder_pos).length

def update_rig_with_landmarks(rig, hand_landmarks, hand_type):
    """ Update the rig's arm bones using the hand landmarks """
    try:
        hand_bones = ['hand_ik.L', 'forearm_tweak.L', 'upper_arm_tweak.L.001'] if hand_type == "Left" else ['hand_ik.R', 'forearm_tweak.R', 'upper_arm_tweak.R.001']
        
        # Get the world coordinates of the hand, forearm, and upper arm bones
        def get_bone_world_position(bone_name):
            bone = rig.pose.bones.get(bone_name)
            if bone:
                return rig.matrix_world @ bone.head
            return None

        initial_hand_pos = get_bone_world_position(hand_bones[0])
        initial_forearm_pos = get_bone_world_position(hand_bones[1])
        initial_shoulder_pos = get_bone_world_position(hand_bones[2])

        if initial_hand_pos and initial_forearm_pos and initial_shoulder_pos:
            initial_arm_length = calculate_arm_length(rig, hand_bones)

            # Extract hand wrist coordinates from Mediapipe landmarks
            wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

            # Convert Mediapipe coordinates to Blender world coordinates
            wrist_pos = mathutils.Vector([
                (wrist.x - 0.5) * 2,
                (wrist.z - 0.5) * 2,
                (0.5 - wrist.y) * 2
            ])

            # Calculate the new hand position while preserving the arm length
            direction_vector = wrist_pos.normalized()
            new_hand_pos = initial_shoulder_pos + direction_vector * initial_arm_length

            # Update the rig's hand IK bone position
            hand_ik = rig.pose.bones.get(hand_bones[0])
            if hand_ik:
                new_local_pos = rig.matrix_world.inverted() @ new_hand_pos
                hand_ik.location = new_local_pos - hand_ik.head
                hand_ik.keyframe_insert(data_path="location")

                # Log the bone name and applied scaling factor
                log_control_names('Custom', hand_bones[0])

        bpy.context.view_layer.update()

    except Exception as e:
        print(f"Error updating rig: {e}")
