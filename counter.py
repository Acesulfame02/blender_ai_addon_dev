import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
from PIL import Image

# Path to the model
model_path = 'd:/school/practical_visual_studio/blender_ai_addon_dev/models/saved_models/action_model.keras'

def load_ai_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_sequence(model, num_frames=1):
    input_sequences = np.random.rand(num_frames, 30, 18)
    predictions = model.predict(input_sequences)
    return predictions

def interpret_output(predictions):
    return predictions[0].reshape(30, 6, 3)

def visualize_keypoints(keypoints, frame, output_dir):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale factor to make the points more visible
    scale = 100
    
    # Left arm (red)
    ax.plot(keypoints[frame, :3, 0]*scale, keypoints[frame, :3, 1]*scale, keypoints[frame, :3, 2]*scale, 'ro-', label='Left Arm', linewidth=2, markersize=8)
    
    # Right arm (blue)
    ax.plot(keypoints[frame, 3:, 0]*scale, keypoints[frame, 3:, 1]*scale, keypoints[frame, 3:, 2]*scale, 'bo-', label='Right Arm', linewidth=2, markersize=8)
    
    # Labels for each keypoint
    labels = ['L Shoulder', 'L Elbow', 'L Wrist', 'R Shoulder', 'R Elbow', 'R Wrist']
    for i, label in enumerate(labels):
        ax.text(keypoints[frame, i, 0]*scale, keypoints[frame, i, 1]*scale, keypoints[frame, i, 2]*scale, label)
    
    # Set axis limits based on the scaled data
    ax.set_xlim(np.min(keypoints[frame,:,0]*scale)-1, np.max(keypoints[frame,:,0]*scale)+1)
    ax.set_ylim(np.min(keypoints[frame,:,1]*scale)-1, np.max(keypoints[frame,:,1]*scale)+1)
    ax.set_zlim(np.min(keypoints[frame,:,2]*scale)-1, np.max(keypoints[frame,:,2]*scale)+1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Frame {frame + 1}')
    
    # Adjust the view angle for better visibility
    ax.view_init(elev=20, azim=45)
    
    filename = os.path.join(output_dir, f'frame_{frame+1:02d}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved image: {filename}")
    
    # Print keypoint values for this frame
    print(f"\nKeypoint values for Frame {frame + 1}:")
    for i, label in enumerate(labels):
        print(f"{label}: {keypoints[frame, i]}")

def display_frames_slowly(output_dir):
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    for frame in frames:
        img = Image.open(os.path.join(output_dir, frame))
        plt.imshow(img)
        plt.axis('off')
        plt.title(frame)
        plt.show()
        time.sleep(1)  # Pause for 1 second between frames
        plt.close()

def main():
    model = load_ai_model(model_path)
    if model is None:
        return
    
    print(f"Model input shape: {model.input_shape }")
    print(f"Model output shape: {model.output_shape}")
    
    model.summary()
    
    predictions = generate_sequence(model, num_frames=1)
    keypoints = interpret_output(predictions)
    
    output_dir = 'keypoint_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    for frame in range(30):
        visualize_keypoints(keypoints, frame, output_dir)

    print(f"\nAll images have been saved in the directory: {os.path.abspath(output_dir)}")
    
    # Display the frames slowly
    print("\nDisplaying frames...")
    display_frames_slowly(output_dir)

if __name__ == "__main__":
    main()