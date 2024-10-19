import bpy
import numpy as np
from utilfunc.utils import apply_keypoints_to_rig
from .model_operators import global_model_store

class AI_OT_AnimateCharacter(bpy.types.Operator):
    bl_idname = "ai.animate_character"
    bl_label = "Animate Character"
    bl_description = "Animate the character using the AI model"

    def execute(self, context):
        model = global_model_store.get('ai_model')
        if model is None:
            self.report({'ERROR'}, "AI model not loaded")
            return {'CANCELLED'}
        
        obj = bpy.data.objects.get(context.scene.rig_object_name)
        if obj is None:
            self.report({'ERROR'}, f"Rig object '{context.scene.rig_object_name}' not found")
            return {'CANCELLED'}

        keypoints = np.random.rand(30, 144)  # Generate 30 frames of keypoints, 144 per frame

        for frame in range(keypoints.shape[0]):
            apply_keypoints_to_rig(obj, keypoints[frame], frame)
            bpy.context.scene.frame_set(frame)

        self.report({'INFO'}, "Character animated successfully")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(AI_OT_AnimateCharacter)

def unregister():
    bpy.utils.unregister_class(AI_OT_AnimateCharacter)
