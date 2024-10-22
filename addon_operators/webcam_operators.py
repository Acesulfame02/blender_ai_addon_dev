import bpy
import cv2
import mediapipe as mp
import json
import os
from ..utilfunc.utils import update_rig_with_landmarks, get_processed_keypoints
from ..config import Config

class AI_OT_WebcamAnimate(bpy.types.Operator):
    bl_idname = "ai.webcam_animate"
    bl_label = "Webcam Animate Rig"
    bl_description = "Animate the rig using webcam input and save processed keypoints to .json"

    _timer = None
    _cap = None
    _running = False

    def execute(self, context):
        if not AI_OT_WebcamAnimate._running:
            AI_OT_WebcamAnimate._cap = cv2.VideoCapture(0)
            AI_OT_WebcamAnimate._timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)
            AI_OT_WebcamAnimate._running = True
            
            # Prepare to store processed keypoints for saving
            self.keypoints_data = []  
            return {'RUNNING_MODAL'}
        else:
            self.report({'ERROR'}, "Webcam animation is already running.")
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'TIMER' and AI_OT_WebcamAnimate._running:
            ret, frame = AI_OT_WebcamAnimate._cap.read()
            if not ret:
                self.cancel(context)
                return {'CANCELLED'}
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = mp.solutions.hands.Hands()
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = hand_info.classification[0].label
                    rig = bpy.data.objects.get(context.scene.rig_object_name)
                    if rig:
                        # Update the rig and get the processed keypoints
                        update_rig_with_landmarks(rig, hand_landmarks, hand_type)
                        processed_keypoints = get_processed_keypoints(rig, hand_type)  # Get the transformed keypoints
                        self.collect_processed_keypoints(processed_keypoints, hand_type)  # Collect for saving
                    else:
                        self.report({'ERROR'}, f"Rig object '{context.scene.rig_object_name}' not found")

            cv2.imshow('Mediapipe Hands', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cancel(context)
                return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def collect_processed_keypoints(self, processed_keypoints, hand_type):
        """
        Collect the processed keypoints for saving to .json.
        """
        self.keypoints_data.append({
            'hand_type': hand_type,
            'keypoints': processed_keypoints
        })

    def save_keypoints_to_json(self):
        """
        Save the processed keypoints to a JSON file.
        """
        config = Config()  # Get paths from config
        json_directory = os.path.join(config.base_path, 'data', 'json_coordinates_for_blender')
        os.makedirs(json_directory, exist_ok=True)
        json_file_path = os.path.join(json_directory, 'processed_hand_keypoints.json')

        # Save keypoints to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(self.keypoints_data, json_file, indent=4)
        
        print(f"Keypoints saved to {json_file_path}")
        self.report({'INFO'}, f"Keypoints saved to {json_file_path}")

    def cancel(self, context):
        AI_OT_WebcamAnimate._running = False
        AI_OT_WebcamAnimate._cap.release()
        cv2.destroyAllWindows()
        context.window_manager.event_timer_remove(AI_OT_WebcamAnimate._timer)

        # Save collected processed keypoints to JSON when the animation is canceled
        self.save_keypoints_to_json()
        return {'CANCELLED'}

# Registration functions
def register():
    bpy.utils.register_class(AI_OT_WebcamAnimate)

def unregister():
    bpy.utils.unregister_class(AI_OT_WebcamAnimate)