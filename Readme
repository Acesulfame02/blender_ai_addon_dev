# AI-Driven Character Animation Addon for Blender

## Overview

The **AI-Driven Character Animation Addon** brings automated, AI-based animation to Blender. Using a pre-trained deep learning model, this addon predicts character poses and applies them to a rig, enabling game developers and animators to streamline character animation workflows. The addon supports both pre-trained model-based animation and live webcam-driven real-time animation.

## Project Goals

- Provide a Blender addon that automates character rig animation using AI.
- Allow users to animate rigs using live webcam input.
- Enhance productivity by automating repetitive animation tasks like hand and arm movements.
- Offer compatibility with a variety of rigs (such as Rigify) while allowing flexibility for custom rig structures.
- Improve the addon’s AI model to handle more complex animations and provide better accuracy.

## Features

- **AI-Based Pose Prediction**: Load pre-trained AI models to predict the character’s pose and animate the rig automatically.
- **Real-time Webcam Animation**: Capture webcam input and use pose detection to animate the rig in real-time.
- **Rig Compatibility**: Supports standard Blender rigs (e.g., Rigify), with the ability to adapt to custom rigs via bone mappings.
- **Modular Design**: Easy to extend the functionality, add new rig structures, or train with custom AI models.

---

## Installation

### Step 1: Download and Extract

1. Download the addon from the repository.
2. Extract the contents of the `.zip` file to a directory of your choice.

### Step 2: Install in Blender

1. Open Blender and go to the **Edit** menu, then select **Preferences**.
2. In the **Add-ons** tab, click **Install...** and navigate to the directory where you extracted the addon.
3. Select the addon’s `.py` file and click **Install**.
4. Enable the addon by checking the box next to its name.

### Step 3: Install Dependencies

The addon requires several Python libraries that are not bundled with Blender by default:
- **TensorFlow**
- **NumPy**
- **mediapipe**
- **opencv-python**

To install them:
1. Open the **Blender System Console** (from the **Window** menu, select **Toggle System Console**).
2. In the console, type:
   ```bash
   bpy.ops.wm.console_toggle()  # Open the system console within Blender
   ```

3. Install dependencies by running:
   ```bash
   import subprocess
   subprocess.check_call([bpy.app.binary_path_python, '-m', 'pip', 'install', 'tensorflow', 'numpy', 'mediapipe', 'opencv-python'])
   ```

---

## Requirements

- **Blender 2.80+**
- **Python 3.7+**
- The following Python packages must be installed inside Blender’s Python environment:
  - TensorFlow
  - NumPy
  - Mediapipe
  - OpenCV-Python

---

## Usage

### 1. **Loading the AI Model**

1. Open the Blender **System Console** (from the **Window** menu, select **Toggle System Console**).
2. Run the operator to load the AI model by typing:
   ```python
   bpy.ops.ai.load_model()
   ```
3. The model will load from the `models/saved_models/action_model.keras` file. If it is not found, you will see an error message in the console.

### 2. **Animating the Character**

1. Select the rigged character object in your scene.
2. Open the **System Console** to monitor progress.
3. Run the "Animate Character" operator by typing:
   ```python
   bpy.ops.ai.animate_character()
   ```
4. The addon will predict the character’s pose using the loaded AI model and apply it to the selected rig over multiple frames.

### 3. **Webcam Animation (Real-time)**

1. Select the rigged character object.
2. Open the **System Console**.
3. Run the "Webcam Animate Rig" operator by typing:
   ```python
   bpy.ops.ai.webcam_animate()
   ```
4. The webcam will capture your movement and animate the rig in real-time. You will see live feedback on the screen as well as the rig’s movement in Blender.

---

## Advanced Configuration

### Customizing Bone Mappings

If your rig uses custom bone names, you can modify the mappings in the script to match your rig’s structure. The default mappings work with Rigify rigs.

In the file where bones are applied, update the `left_arm_bones` and `right_arm_bones` arrays:
```python
left_arm_bones = ['shoulder.L', 'elbow.L', 'hand_fk.L']
right_arm_bones = ['shoulder.R', 'elbow.R', 'hand_fk.R']
```
Replace these bone names with the names used in your rig.

### Adjusting Clamping Values

To allow more freedom in motion, modify the clamping logic to adjust the range of movement. By default, bone movements are clamped to avoid excessive transformations:
```python
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))
```
You can adjust the scaling factor or remove the clamping to allow a wider range of movement.

---

## Troubleshooting

### 1. AI Model Fails to Load

If the AI model fails to load:
- Verify that the model file (`action_model.keras`) exists in the directory `models/saved_models/` inside the addon folder.
- Check for any error messages in the system console and ensure all required dependencies are installed.

### 2. Rig Not Updating Correctly

If the rig isn't animating correctly:
- Ensure the rig object is selected in the scene and the `rig_object_name` property is set correctly.
- Ensure bone names match between the rig and the script.

### 3. Webcam Animation Not Working

If webcam input doesn't animate the rig:
- Ensure that your webcam is accessible by OpenCV.
- Check that your system meets the hardware requirements to process real-time animation.
- Verify that the `mediapipe` library is properly installed and working in Blender’s Python environment.

---

## Known Issues

- The addon may not work with all rig types. Some rigs may require manual adjustments to the bone mappings.
- Webcam animation may not work smoothly on all systems due to hardware or performance limitations.

---

## Future Development

- **Improved AI Model Accuracy**: Enhance the AI model to better handle complex poses, such as walking, dancing, and emotional expressions.
- **Expanded Rig Compatibility**: Add support for more types of rigs and characters beyond the default Rigify setup.
- **Performance Optimizations**: Optimize the real-time webcam animation feature for smoother, higher-FPS performance on more systems.
- **UI Improvements**: Add a custom Blender UI panel to replace the need to use the system console for running operators.

---

## Contribution Guide

We welcome contributions to improve the addon:
- Add features such as support for additional AI models.
- Improve the animation quality and rig compatibility.
- Optimize the performance of the real-time animation feature.
  
Please fork the repository, create a feature branch, and submit a pull request with your changes.

---

## License

This addon is licensed under the MIT License. See the `LICENSE` file for details.

