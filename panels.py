import bpy

class AI_PT_AnimationPanel(bpy.types.Panel):
    bl_label = "AI Animation Panel"
    bl_idname = "AI_PT_animation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AI Animation'

    def draw(self, context):
        layout = self.layout

        layout.prop(context.scene, 'rig_object_name')
        layout.operator("ai.load_model", text="Load AI Model")
        layout.operator("ai.animate_character", text="Animate Character")
        layout.operator("ai.webcam_animate", text="Use Webcam for Animation")

def register():
    bpy.utils.register_class(AI_PT_AnimationPanel)
    bpy.types.Scene.rig_object_name = bpy.props.StringProperty(
        name="Rig Object Name",
        description="Name of the rigged object in the scene"
    )

def unregister():
    bpy.utils.unregister_class(AI_PT_AnimationPanel)
    del bpy.types.Scene.rig_object_name

if __name__ == "__main__":
    register()
