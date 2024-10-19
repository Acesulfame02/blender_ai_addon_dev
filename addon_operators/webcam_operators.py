import bpy
import cv2
import mediapipe as mp
import threading

from utilfunc.utils import update_rig

class AI_OT_WebcamAnimate(bpy.types.Operator):
    bl_idname = "ai.webcam_animate"
    bl_label = "Webcam Animate Rig"
    bl_description = "Animate the rig using webcam input"

    _timer = None
    _cap = None
    _running = False
    _thread = None

    def execute(self, context):
        if not AI_OT_WebcamAnimate._running:
            AI_OT_WebcamAnimate._cap = cv2.VideoCapture(0)
            AI_OT_WebcamAnimate._timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)
            AI_OT_WebcamAnimate._running = True
            AI_OT_WebcamAnimate._thread = threading.Thread(target=self.process_video)
            AI_OT_WebcamAnimate._thread.start()
            return {'RUNNING_MODAL'}
        else:
            self.report({'ERROR'}, "Webcam animation is already running.")
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        AI_OT_WebcamAnimate._running = False
        if AI_OT_WebcamAnimate._cap:
            AI_OT_WebcamAnimate._cap.release()
        cv2.destroyAllWindows()
        context.window_manager.event_timer_remove(AI_OT_WebcamAnimate._timer)
        if AI_OT_WebcamAnimate._thread:
            AI_OT_WebcamAnimate._thread.join()
        return {'CANCELLED'}

    def process_video(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        
        rig = bpy.data.objects.get(bpy.context.scene.rig_object_name)
        if not rig:
            print(f"Error: Could not find rig '{bpy.context.scene.rig_object_name}' in Blender.")
            return

        while AI_OT_WebcamAnimate._running and AI_OT_WebcamAnimate._cap.isOpened():
            ret, frame = AI_OT_WebcamAnimate._cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = hand_info.classification[0].label
                    bpy.app.timers.register(lambda: update_rig(hand_landmarks, hand_type, rig), first_interval=0.01)
            
            cv2.imshow('Mediapipe Hands', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        AI_OT_WebcamAnimate._cap.release()
        cv2.destroyAllWindows()

def register():
    bpy.utils.register_class(AI_OT_WebcamAnimate)

def unregister():
    bpy.utils.unregister_class(AI_OT_WebcamAnimate)