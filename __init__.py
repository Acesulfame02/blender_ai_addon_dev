import bpy
import importlib
import subprocess
import sys

bl_info = {
    "name": "AI-Driven Character Animation",
    "blender": (2, 80, 0),
    "category": "Animation",
}

from .operators import AI_OT_LoadModel, AI_OT_AnimateCharacter, AI_OT_WebcamAnimate
from .panels import AI_PT_AnimationPanel

REQUIRED_PACKAGES = ["tensorflow", "numpy", "mediapipe", "opencv-python", "scikit-learn"]

def check_and_install_packages(packages):
    total_packages = len(packages)
    bpy.context.window_manager.progress_begin(0, total_packages)
    for i, package in enumerate(packages):
        try:
            importlib.import_module(package)
            print(f"{package} is already installed")
        except ImportError:
            print(f"{package} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        bpy.context.window_manager.progress_update(i + 1)
    bpy.context.window_manager.progress_end()

def register():
    check_and_install_packages(REQUIRED_PACKAGES)
    
    bpy.utils.register_class(AI_OT_LoadModel)
    bpy.utils.register_class(AI_OT_AnimateCharacter)
    bpy.utils.register_class(AI_OT_WebcamAnimate)
    bpy.utils.register_class(AI_PT_AnimationPanel)

    bpy.types.Scene.rig_object_name = bpy.props.StringProperty(
        name="Rig Object Name",
        description="Name of the rigged object in the scene"
    )

def unregister():
    bpy.utils.unregister_class(AI_OT_LoadModel)
    bpy.utils.unregister_class(AI_OT_AnimateCharacter)
    bpy.utils.unregister_class(AI_OT_WebcamAnimate)
    bpy.utils.unregister_class(AI_PT_AnimationPanel)

    del bpy.types.Scene.rig_object_name

if __name__ == "__main__":
    register()
