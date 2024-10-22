import bpy

class AI_PT_AnimationPanel(bpy.types.Panel):
    bl_label = "AI Animation Panel"
    bl_idname = "AI_PT_animation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AI Animation'

    def draw(self, context):
        layout = self.layout
        
        # Rig object dropdown
        row = layout.row()
        row.label(text="Rig Object:")
        row.prop_search(context.scene, 'rig_object_name', context.scene, 'objects', text="")
        
        # Animation dropdown
        layout.label(text="Animation:")
        col = layout.column(align=True)
        col.operator("ai.load_model", text="Load AI Model")
        col.operator("ai.animate_character", text="Animate Character")
        
        # Webcam dropdown
        layout.label(text="Webcam:")
        col = layout.column(align=True)
        col.operator("ai.webcam_get_keypoints", text="Get Keypoints (.json)")
        col.operator("ai.webcam_animate", text="Use Webcam for Animation")
        
        # Train dropdown (Future)
        layout.label(text="Train (Future Implementation):")
        col = layout.column(align=True)
        col.label(text="Future training options coming soon...")

# Registering the panel and properties
def register():
    bpy.utils.register_class(AI_PT_AnimationPanel)
    bpy.types.Scene.rig_object_name = bpy.props.StringProperty(
        name="Rig Object Name",
        description="Select the rigged object in the scene"
    )

def unregister():
    bpy.utils.unregister_class(AI_PT_AnimationPanel)
    del bpy.types.Scene.rig_object_name

if __name__ == "__main__":
    register()
