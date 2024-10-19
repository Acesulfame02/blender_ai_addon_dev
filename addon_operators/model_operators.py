import bpy
import os

from utilfunc.ai_models import load_ai_model

global_model_store = {}

class AI_OT_LoadModel(bpy.types.Operator):
    bl_idname = "ai.load_model"
    bl_label = "Load AI Model"
    bl_description = "Load the trained AI model for character animation"

    def execute(self, context):
        addon_dir = os.path.join(os.getenv('APPDATA'), 'Blender Foundation', 'Blender', '4.2', 'scripts', 'addons', 'kma')
        model_path = os.path.join(addon_dir, 'models', 'saved_models', 'action_model.keras')

        print(f"Executing load model with path: {model_path}")
        
        if not os.path.exists(model_path):
            self.report({'ERROR'}, f"Model file not found at: {model_path}")
            return {'CANCELLED'}
        
        try:
            model = load_ai_model(self, model_path)
            if model is None:
                raise ValueError("Model loading returned None.")
            
            global_model_store['ai_model'] = model
            self.report({'INFO'}, "Model loaded successfully")
            return {'FINISHED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load model: {e}")
            return {'CANCELLED'}

def register():
    bpy.utils.register_class(AI_OT_LoadModel)

def unregister():
    bpy.utils.unregister_class(AI_OT_LoadModel)
