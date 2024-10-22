import bpy
import numpy as np
import os
from ..utilfunc.utils import update_rig_with_model_keypoints, normalize_keypoints
from .model_operators import global_model_store

# Define path to log file
DOWNLOADS_PATH = os.path.join(os.path.expanduser('~'), 'Downloads')
LOG_FILE = os.path.join(DOWNLOADS_PATH, 'arm_control_names.txt')

class AI_OT_AnimateCharacter(bpy.types.Operator):
    bl_idname = "ai.animate_character"
    bl_label = "Animate Character"
    bl_description = "Animate the character using the AI model"

    def execute(self, context):
        model = global_model_store.get('ai_model')
        if model is None:
            self.report({'ERROR'}, "AI model not loaded")
            return {'CANCELLED'}
        
        rig = bpy.data.objects.get(context.scene.rig_object_name)
        if rig is None:
            self.report({'ERROR'}, f"Rig object '{context.scene.rig_object_name}' not found")
            return {'CANCELLED'}

        # Get model predictions and normalize them
        try:
            input_sequence = self.get_model_predictions(model)
            normalized_sequence = normalize_keypoints(input_sequence)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        # Log the raw and normalized predictions to a file for debugging
        self.log_predictions(input_sequence, normalized_sequence)

        # Apply keypoints for each frame
        for frame_idx, keypoints in enumerate(normalized_sequence[0]):  # [0] to get the first (and only) batch
            bpy.context.scene.frame_set(frame_idx + 1)  # +1 because Blender starts at frame 1
            update_rig_with_model_keypoints(rig, keypoints)

        self.report({'INFO'}, "Character animated successfully")
        return {'FINISHED'}

    def get_model_predictions(self, model):
        """
        Generate predictions from the AI model. Ensure the input has the correct shape (batch_size, time_steps, feature_size).
        """
        # Correctly shape the input to match the model's requirements: (batch_size, time_steps, feature_size)
        batch_size = 1
        time_steps = 8  # Adjusted time steps
        feature_size = 18  # Feature size per time step

        # Reshape input to (batch_size, time_steps, feature_size)
        input_data = np.random.rand(batch_size, time_steps, feature_size)  # Example input with correct shape

        # Make predictions
        predictions = model.predict(input_data)

        # Check if predictions are valid
        if predictions is None or predictions.shape[0] == 0:
            self.log_error("No keypoints returned by the model.")
            raise ValueError("No keypoints available. Please check the model.")

        if np.all(predictions == 0):
            self.log_error("Model returned all zero keypoints. Please consider retraining the model.")
            raise ValueError("Model returned zero keypoints. Retrain the model.")

        return predictions


    def log_predictions(self, raw_predictions, normalized_predictions):
        """
        Log both raw and normalized predictions to a text file for debugging.
        """
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Raw Model Predictions:\n{raw_predictions}\n")
            log_file.write(f"Normalized Model Predictions:\n{normalized_predictions}\n")
        print(f"Logged model predictions to {LOG_FILE}")

    def log_error(self, message):
        """
        Log errors to a file located in the Downloads directory.
        """
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Error: {message}\n")
        print(f"Logged error: {message}")

def register():
    bpy.utils.register_class(AI_OT_AnimateCharacter)

def unregister():
    bpy.utils.unregister_class(AI_OT_AnimateCharacter)
