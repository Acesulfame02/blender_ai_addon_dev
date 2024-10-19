import bpy
from .addon_operators.animation_operators import AI_OT_AnimateCharacter
from .addon_operators.model_operators import AI_OT_LoadModel
from .addon_operators.webcam_operators import AI_OT_WebcamAnimate

def register():
    bpy.utils.register_class(AI_OT_LoadModel)
    bpy.utils.register_class(AI_OT_AnimateCharacter)
    bpy.utils.register_class(AI_OT_WebcamAnimate)

def unregister():
    bpy.utils.unregister_class(AI_OT_LoadModel)
    bpy.utils.unregister_class(AI_OT_AnimateCharacter)
    bpy.utils.unregister_class(AI_OT_WebcamAnimate)

if __name__ == "__main__":
    register()