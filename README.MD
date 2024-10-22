# AI-Powered Character Animation Addon for Blender

## Overview

The **AI-Powered Character Animation Addon** brings advanced, AI-driven animation capabilities to Blender. This addon integrates pre-trained deep learning models to automate the process of character animation, focusing on realistic hand movements with future goals to incorporate walking, talking, emotional expressions, and other complex behaviors. It supports both real-time animation via webcam input and pre-trained AI model-based animation, allowing game developers and animators to streamline their workflow.

## Project Objectives

### 1.3.1 Primary Objective

To develop an AI-driven Blender add-on that enables game developers to create realistic 3D character animations with a focus on hand movements. Future iterations will include walking, talking, and other complex animations.

### 1.3.2 Specific Objectives

- **AI Model Integration**: Integrate AI models for simulating character animations, with a focus on hand movements.
- **User-Friendly Interface**: Provide a seamless, easy-to-use interface for accessing hand animation features.
- **Real-Time Responsiveness**: Enable real-time hand movement animations through webcam input.
- **Data Collection**: Capture and preprocess diverse datasets of human hand movements for training AI models.
- **Evaluation**: Assess the effectiveness of the add-on in improving animation quality, usability, and workflow for hand animations.

## Achieved Objectives

- **Blender Add-on with AI Integration**: Successfully built the AI-driven add-on that can animate character hand movements.
- **User Interface Enhancements**: Created a user-friendly interface with dropdowns for easier interaction with rig and animation options.
- **Webcam Integration**: Implemented real-time hand animation via webcam input, allowing for immediate feedback and keypoint capture.
- **Data Export**: Ability to export hand movement keypoints captured through the webcam for future model training.

## Features

### 1.4.1 Key Features

- **AI-Driven Hand Animation**: Use a pre-trained AI model to animate character hands based on human hand movement data.
- **Webcam-Driven Real-Time Animation**: Capture movements through a webcam and animate rigged characters in real-time.
- **Keypoints Export**: Export captured keypoints to `.json` for further use in training or refining models.
- **User-Friendly Interface**: Includes dropdown menus for rig selection, animation options, and webcam features.
  
### 1.4.2 Interface Options

- **Object Rig Name Dropdown**: Select the rigged object to animate in Blender.
- **Animation Dropdown**:
  - **Load Model**: Load the pre-trained AI model for animation.
  - **Animate Character**: Animate the character using the loaded model.
- **Webcam Dropdown**:
  - **Get Keypoints**: Capture and save keypoints in `.json` format for future use.
  - **Use Webcam**: Animate the rig in real-time using webcam input.
- **Train Dropdown**: Placeholder for future training options.

## Installation

### 1.5.1 Step 1: Clone Repository

Clone the repository:
```bash
git clone https://github.com/Acesulfame02/blender_ai_addon_dev.git
```

### 1.5.2 Step 2: Install Addon in Blender

1. Open Blender and go to **Edit > Preferences**.
2. In the **Add-ons** tab, click **Install...** and navigate to the directory containing the `.py` files.
3. Select the add-on file and click **Install Add-on**.
4. Enable the add-on by checking the box next to its name.

### 1.5.3 Step 3: Install Python Dependencies

Ensure you have installed the following dependencies in Blender’s Python environment:
```bash
pip install tensorflow numpy mediapipe opencv-python
```

## Usage

### 1.6.1 Loading the AI Model

1. Select the rigged character in your Blender scene.
2. Open the **Animation** dropdown and click **Load Model**.
3. The model will load from the `models/saved_models/action_model.keras` file.

### 1.6.2 Animating the Character

1. After loading the model, go to the **Animation** dropdown and click **Animate Character**.
2. The AI model will predict hand movements and animate the rigged character accordingly.

### 1.6.3 Webcam-Driven Real-Time Animation

1. Select the rigged character.
2. Use the **Webcam** dropdown to choose **Use Webcam**.
3. The webcam captures movements and animates the rig in real-time.

### 1.6.4 Exporting Keypoints

1. Use the **Webcam** dropdown and select **Get Keypoints**.
2. Captured keypoints are saved in `.json` format in `data/json_coordinates_for_blender`, useful for training future AI models.

## Advanced Configuration

### 1.7.1 Customizing Bone Mappings

If your rig uses custom bone names, modify the `left_arm_bones` and `right_arm_bones` arrays in the script to match your rig structure.

```python
left_arm_bones = ['shoulder.L', 'elbow.L', 'hand_fk.L']
right_arm_bones = ['shoulder.R', 'elbow.R', 'hand_fk.R']
```

### 1.7.2 Adjusting Keypoint Scaling

The captured keypoints are normalized for better visibility in Blender. You can modify the scaling factors if necessary by changing the `normalize_keypoints()` function in `utils.py`.

## Troubleshooting

### 1.8.1 Common Errors

- **AI Model Not Loaded**: Ensure the `action_model.keras` file is located in the `models/saved_models` directory.
- **Webcam Input Issues**: Verify your webcam is accessible by OpenCV and ensure all dependencies are installed.

### 1.8.2 Keypoints Not Being Captured

If keypoints are not captured:
- Ensure the webcam is working and that `mediapipe` is correctly processing the input.

## Known Issues

- The add-on is currently optimized for hand movements. Future iterations will expand to include full-body animations.
- Real-time webcam input may lag on lower-end systems.

## Future Development

- **Full-Body Animation**: Expand the model to handle walking, talking, and emotional expressions.
- **Improved AI Models**: Refine the models for smoother animations and better accuracy.
- **Training Interface**: Provide an interface for users to train their own models within Blender.

## Contribution Guide

We welcome contributions to the project! To contribute:

1. **Fork the Repository** on GitHub.
2. Create a **Feature Branch** for your changes.
3. Submit a **Pull Request** with a detailed explanation.

Please **do not directly commit to the main branch**.

## License

This add-on is licensed under the MIT License. See the `LICENSE` file for details.