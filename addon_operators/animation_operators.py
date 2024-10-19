import bpy
import numpy as np
from ..utilfunc.utils import update_rig_with_landmarks
from .model_operators import global_model_store

class AI_OT_AnimateCharacter(bpy.types.Operator):
    bl_idname = "ai.animate_character"
    bl_label = "Animate Character"
    bl_description = "Animate the character using the AI model"

    def execute(self, context):
        # Load the AI model from the global store
        model = global_model_store.get('ai_model')
        if model is None:
            self.report({'ERROR'}, "AI model not loaded")
            return {'CANCELLED'}
        
        # Get the rig object from Blender
        obj = bpy.data.objects.get(context.scene.rig_object_name)
        if obj is None:
            self.report({'ERROR'}, f"Rig object '{context.scene.rig_object_name}' not found")
            return {'CANCELLED'}
        
        # Generate or load your actual input sequence (shape: [num_frames, 30, 4320])
        # Example: You can load real data from a file or generate based on your application
        input_sequence = np.random.rand(30, 30, 4320)  # Example random data, replace with actual input data
        
        # Predict using the AI model
        try:
            predictions = model.predict(input_sequence)  # Pass actual input here
        except Exception as e:
            self.report({'ERROR'}, f"Prediction failed: {str(e)}")
            return {'CANCELLED'}

        # Interpolate predictions for smoother animation
        interpolated_predictions = self.interpolate_predictions(predictions, factor=4)

        # Apply the keypoints frame by frame
        for frame_idx, prediction in enumerate(interpolated_predictions):
            efficient_keypoints = self.extract_efficient_keypoints(prediction)
            
            # Apply keypoints to the left and right arms
            update_rig_with_landmarks(obj, efficient_keypoints[:8], "Left")  # First 4 points for left side
            update_rig_with_landmarks(obj, efficient_keypoints[8:], "Right")  # Next 4 points for right side

            bpy.context.scene.frame_set(frame_idx)

        self.report({'INFO'}, "Character animated successfully")
        return {'FINISHED'}

    def extract_efficient_keypoints(self, prediction):
        """
        Extract the keypoints for shoulders, elbows, and wrists from the prediction.
        Assuming that the prediction shape is (99,) for 33 keypoints with x, y, z coordinates.
        """
        keypoints = prediction.reshape(-1, 3)  # Reshape into (33, 3)

        # Define indices of key joints (this may vary depending on the model output format)
        key_indices = [
            0, 1,  # Left and right shoulder
            2, 3,  # Left and right elbow
            4, 5,  # Left and right wrist
            6, 7   # Optional: thumb tips (if you want more detailed hand movements)
        ]

        # Extract only key joints
        efficient_keypoints = keypoints[key_indices]
        return efficient_keypoints.flatten()

    def interpolate_predictions(self, predictions, factor=2):
        """
        Interpolate between predictions to create smoother transitions for animation.
        """
        interpolated = []
        for i in range(len(predictions) - 1):
            start = predictions[i]
            end = predictions[i + 1]
            for j in range(factor):
                t = j / factor
                interpolated.append(start * (1 - t) + end * t)
        interpolated.append(predictions[-1])  # Add the last prediction
        return np.array(interpolated)

def register():
    bpy.utils.register_class(AI_OT_AnimateCharacter)

def unregister():
    bpy.utils.unregister_class(AI_OT_AnimateCharacter)
