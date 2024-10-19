import numpy as np

def mediapipe_to_blender_coords(mp_coords):
    """Convert MediaPipe coordinates to Blender coordinates."""
    return [
        (mp_coords[0] - 0.5) * 2,     # x: [-1, 1]
        (mp_coords[2] - 0.5) * 2,     # y: [-1, 1] (MediaPipe z becomes Blender y)
        -(mp_coords[1] - 0.5) * 2     # z: [-1, 1] (inverted MediaPipe y becomes Blender z)
    ]

def blender_to_mediapipe_coords(blender_coords):
    """Convert Blender coordinates to MediaPipe coordinates."""
    return [
        (blender_coords[0] / 2) + 0.5,     # x: [0, 1]
        -(blender_coords[2] / 2) + 0.5,    # y: [0, 1] (inverted Blender z becomes MediaPipe y)
        (blender_coords[1] / 2) + 0.5      # z: [0, 1] (Blender y becomes MediaPipe z)
    ]