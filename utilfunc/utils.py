import bpy
import mathutils
import numpy as np
import mediapipe as mp
import os

def calculate_arm_length(rig, hand_bones):
    """ Calculate arm length based on rig bone positions """
    hand_pos = rig.matrix_world @ rig.pose.bones[hand_bones[0]].head
    shoulder_pos = rig.matrix_world @ rig.pose.bones[hand_bones[1]].head
    return (hand_pos - shoulder_pos).length

def get_processed_keypoints(rig, hand_type):
    """
    Returns the processed keypoints (after applying transformations) in world space for the rig.
    """
    if hand_type == "Left":
        hand_bones = ['hand_ik.L', 'forearm_tweak.L', 'upper_arm_tweak.L.001']
    else:
        hand_bones = ['hand_ik.R', 'forearm_tweak.R', 'upper_arm_tweak.R.001']

    keypoints = []
    
    # Retrieve the world-space positions of the hand, forearm, and upper arm bones
    for bone_name in hand_bones:
        bone = rig.pose.bones.get(bone_name)
        if bone:
            world_position = rig.matrix_world @ bone.head
            keypoints.append({
                'x': world_position.x,
                'y': world_position.y,
                'z': world_position.z
            })

    return keypoints

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

        bpy.context.view_layer.update()

    except Exception as e:
        print(f"Error updating rig: {e}")

def normalize_keypoints(predictions):
    """
    Normalize predictions to a usable range for Blender (between 0 and 1).
    Enforce a minimum value of 0.01 for visibility in Blender.
    """
    # Normalize keypoints to be between 0 and 1
    min_val = np.min(predictions)
    max_val = np.max(predictions)

    if max_val > min_val:
        normalized_predictions = (predictions - min_val) / (max_val - min_val)
    else:
        normalized_predictions = predictions

    # Enforce a minimum value of 0.01 for better visibility in Blender
    normalized_predictions = np.maximum(normalized_predictions, 0.01)

    # Round to 2 decimal places
    normalized_predictions = np.round(normalized_predictions, decimals=2)

    return normalized_predictions

def update_rig_with_model_keypoints(rig, keypoints):
    """
    Updates the rig's arm bones using model-generated keypoints.
    This version assumes the keypoints are provided in Blender's coordinate system.
    """
    keypoints = np.array(keypoints, dtype=float)
    
    if len(keypoints) < 18:
        raise ValueError(f"Not enough keypoints provided. Expected at least 18, got {len(keypoints)}")

    left_keypoints = keypoints[:9]   # First 9 values for the left arm
    right_keypoints = keypoints[9:]  # Last 9 values for the right arm

    # Update left and right arms
    update_arm(rig, left_keypoints, 'Left')
    update_arm(rig, right_keypoints, 'Right')

def update_arm(rig, keypoints, side):
    """
    Apply keypoints to the rig for a specific arm (left or right).
    Ensures the arm length remains constant.
    """
    if side == 'Left':
        hand_bones = ['hand_ik.L', 'forearm_tweak.L', 'upper_arm_tweak.L.001']
    else:
        hand_bones = ['hand_ik.R', 'forearm_tweak.R', 'upper_arm_tweak.R.001']

    # Extract shoulder, elbow, and wrist positions
    shoulder = mathutils.Vector(keypoints[0:3].tolist())
    elbow = mathutils.Vector(keypoints[3:6].tolist())
    wrist = mathutils.Vector(keypoints[6:9].tolist())

    if len(shoulder) != 3 or len(elbow) != 3 or len(wrist) != 3:
        raise ValueError(f"Keypoints have inconsistent dimensions: Shoulder = {shoulder}, Elbow = {elbow}, Wrist = {wrist}")

    # Calculate the original arm length
    arm_length = calculate_arm_length(rig, hand_bones)

    # Calculate the direction vector from shoulder to wrist
    direction = wrist - shoulder
    direction.normalize()

    # Calculate the new hand position while preserving the arm length
    new_hand_pos = shoulder + direction * arm_length

    # Update hand IK bone position
    hand_ik = rig.pose.bones.get(hand_bones[0])
    if hand_ik:
        new_local_pos = rig.matrix_world.inverted() @ new_hand_pos
        hand_ik.location = new_local_pos - hand_ik.head
        hand_ik.keyframe_insert(data_path="location")

    bpy.context.view_layer.update()